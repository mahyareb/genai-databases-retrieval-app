# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import uuid
from datetime import date
from typing import Any, Dict, List, Optional

from aiohttp import ClientSession, TCPConnector
from fastapi import HTTPException
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent import AgentExecutor
from langchain.globals import set_verbose  # type: ignore
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate
from langchain.tools import StructuredTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_vertexai import VertexAI

from ..orchestrator import BaseOrchestrator, classproperty
from .tools import initialize_tools

set_verbose(bool(os.getenv("DEBUG", default=False)))
BASE_HISTORY = {
    "type": "ai",
    "data": {"content": "I am an SFO Airport Assistant, ready to assist you."},
}


class UserAgent:
    client: ClientSession
    agent: AgentExecutor

    def __init__(self, client: ClientSession, agent: AgentExecutor):
        self.client = client
        self.agent = agent

    @classmethod
    def initialize_agent(
        cls,
        client: ClientSession,
        tools: List[StructuredTool],
        history: List[BaseMessage],
        prompt: ChatPromptTemplate,
    ) -> "UserAgent":
        llm = VertexAI(max_output_tokens=512, model_name="gemini-pro")
        memory = ConversationBufferMemory(
            chat_memory=ChatMessageHistory(messages=history),
            memory_key="chat_history",
            input_key="input",
            output_key="output",
        )
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=memory,
            handle_parsing_errors=True,
            max_iterations=3,
            early_stopping_method="generate",
            return_intermediate_steps=True,
        )
        agent.agent.llm_chain.prompt = prompt  # type: ignore
        return UserAgent(client, agent)

    async def close(self):
        await self.client.close()

    async def invoke(self, prompt: str) -> Dict[str, Any]:
        try:
            response = await self.agent.ainvoke({"input": prompt})
        except Exception as err:
            raise HTTPException(status_code=500, detail=f"Error invoking agent: {err}")
        return response


class LangChainToolsOrchestrator(BaseOrchestrator):
    _user_sessions: Dict[str, UserAgent] = {}
    # aiohttp context
    connector = None

    @classproperty
    def kind(cls):
        return "langchain-tools"

    def user_session_exist(self, uuid: str) -> bool:
        return uuid in self._user_sessions

    async def user_session_create(self, session: dict[str, Any]):
        """Create and load an agent executor with tools and LLM."""
        print("Initializing agent..")
        if "uuid" not in session:
            session["uuid"] = str(uuid.uuid4())
        id = session["uuid"]
        if "history" not in session:
            session["history"] = [BASE_HISTORY]
        history = self.parse_messages(session["history"])
        client = await self.create_client_session()
        tools = await initialize_tools(client)
        prompt = self.create_prompt_template(tools)
        agent = UserAgent.initialize_agent(client, tools, history, prompt)
        self._user_sessions[id] = agent

    async def user_session_invoke(self, uuid: str, prompt: str) -> str:
        user_session = self.get_user_session(uuid)
        # Send prompt to LLM
        response = await user_session.invoke(prompt)
        return response["output"]

    async def user_session_reset(self, uuid: str):
        user_session = self.get_user_session(uuid)
        await user_session.close()
        del self._user_sessions[uuid]

    def get_user_session(self, uuid: str) -> UserAgent:
        return self._user_sessions[uuid]

    async def get_connector(self) -> TCPConnector:
        if self.connector is None:
            self.connector = TCPConnector(limit=100)
        return self.connector

    async def create_client_session(self) -> ClientSession:
        return ClientSession(
            connector=await self.get_connector(),
            connector_owner=False,
            headers={},
            raise_for_status=True,
        )

    def create_prompt_template(self, tools: List[StructuredTool]) -> ChatPromptTemplate:
        # Create new prompt template
        tool_strings = "\n".join(
            [f"> {tool.name}: {tool.description}" for tool in tools]
        )
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = FORMAT_INSTRUCTIONS.format(
            tool_names=tool_names,
        )
        today_date = date.today().strftime("%Y-%m-%d")
        today = f"Today is {today_date}."
        template = "\n\n".join(
            [PREFIX, tool_strings, format_instructions, SUFFIX, today]
        )
        human_message_template = "{input}\n\n{agent_scratchpad}"
        prompt = ChatPromptTemplate.from_messages(
            [("system", template), ("human", human_message_template)]
        )
        return prompt

    def parse_messages(self, datas: List[Any]) -> List[BaseMessage]:
        messages: List[BaseMessage] = []
        for data in datas:
            if data["type"] == "human":
                messages.append(HumanMessage(content=data["data"]["content"]))
            if data["type"] == "ai":
                messages.append(AIMessage(content=data["data"]["content"]))
        return messages

    def close_clients(self):
        close_client_tasks = [
            asyncio.create_task(a.close()) for a in self._user_sessions.values()
        ]
        asyncio.gather(*close_client_tasks)


PREFIX = """SFO Airport Assistant helps travelers find their way at the airport.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to
complex multi-query questions that require passing results from one query to another. As a language model, Assistant is
able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding
conversations and provide responses that are coherent and relevant to the topic at hand.

Overall, Assistant is a powerful tool that can help answer a wide range of questions pertaining to the San
Francisco Airport. SFO Airport Assistant is here to assist. It currently does not have access to user info.

TOOLS:
------

Assistant has access to the following tools:"""

FORMAT_INSTRUCTIONS = """Use a json blob to specify a tool by providing an action key (tool name)
and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}}}
```"""

SUFFIX = """Begin! Use tools if necessary. Respond directly if appropriate.
If using a tool, reminder to ALWAYS respond with a valid json blob of a single action.
Format is Action:```$JSON_BLOB```then Observation:.
Thought:

Previous conversation history:
{chat_history}
"""
