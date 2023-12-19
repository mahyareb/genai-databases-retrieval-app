import csv
from langchain.callbacks import FileCallbackHandler
from agent import init_agent

questions = [
    # Flights
    "What gate is flight 118 at?",
    "What is the status of flight 118? ",
    "What is the status of flight 188? ",
    "What gate is flight UA 1532 at?",
    "What is the status of flight UA 1532? ",
    "What time is my flight leaving today?",
    "What flights are departing SFO?",
    "what flights leave SEA on 11-01-2023?",
    "What flights are leaving from SFO today?",
    "What flights are leaving from SFO on 2023-11-01?",
    "What flights land at SFO today?",
    "What flights arrive at SFO today?",
    "What flights arrive at SFO tomorrow?",
    "Find flights that leave SFO and arrive at SEA",
    "Where does flight 118 land?",
    "What gate does flight 118 land?",
    "What gate does flight UA 1532 land?",
    # Amenities
    "Where can I get coffee near gate A6?",
    "Where can I get a snack near the gate for flight 457?",
    "Where can I get a snack near the gate for flight UA1739?",
    "I need a gift",
    "Where is Starbucks?",
    "What are the hours of Amy's Drive Thru?",
    "Where can I get a salad in Terminal 1?",
    "I need headphones",
    "Are there restaurants open at midnight?",
    "Where can I buy a luxury bag?",
    # Airports
    "Where is SFO?",
    "Where is the san Prancisco airport?",
    "What is the code for San Francisco airport?",
    "What is the YWG airport?",
    # Extra
    "hi",
    "How can you help me?",
    "Where are the restrooms?",
    "what are airport hours?",
    "Where is TSA pre-check?",
]

async def run():
    EXPERIMENT = "gemini"
    logfile = "output.log"
    results = []
    for question in questions:
        agent = init_agent()  # Init here for no chat history
        # agent.callbacks = [FileCallbackHandler(logfile)]
        # Clear log
        # open(logfile, "w").close()
        try:
            response = await agent.ainvoke({"input": question})
            output = response["output"]
            steps = response["intermediate_steps"]
            print(question)
            print(response["output"])
            results.append([question, output, steps])
        except Exception as e:
            output = f"Error: {e}"
            steps = "Error"

    #     with open(logfile, "r") as f:
    #         stdout = f.read()
    #     results.append([question, output, stdout, steps])

    # # Write to CSV
    with open(f"experiment-{EXPERIMENT}.csv", "w") as f:
        col_names = ["question", "response", "stdout", "steps"]
        writer = csv.writer(f, delimiter=",")
        writer.writerow(col_names)
        for result in results:
            writer.writerow(result)


import asyncio
asyncio.run(run())