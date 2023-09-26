# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from ipaddress import IPv4Address, IPv6Address
from typing import Any, Dict, Literal, List, Tuple

import asyncpg
from pydantic import BaseModel
from numpy import float32
from google.cloud.alloydb.connector import Connector
import sqlalchemy
from sqlalchemy.ext.asyncio import create_async_engine

import models
from .. import datastore


POSTGRES_IDENTIFIER = "alloydb-postgres"


class Config(BaseModel, datastore.AbstractConfig):
    kind: Literal["alloydb-postgres"]
    driver: str = "pg8000"
    project: str
    region: str
    cluster: str
    instance: str
    user: str
    password: str
    database: str


class Client(datastore.Client):
    __pool: sqlalchemy.pool.Pool

    @classmethod
    @property
    def kind(cls):
        return "alloydb-postgres"

    def __init__(self, pool: asyncpg.Pool):
        self.__pool = pool

    @classmethod
    async def create(cls, config: Config) -> "Client":
        def getconn(config):
            with Connector() as connector:
                url = f"projects/{config.project}/locations/{config.region}/clusters/{config.cluster}/instances/{config.instance}"
                conn = connector.connect(
                    url,
                    config.driver,
                    user=config.user,
                    password=config.password,
                    db=config.database
                )
            return conn

        pool = create_async_engine(
              "postgresql+asyncpg://",
              creator=getconn(config),
        )
        if pool is None:
            raise TypeError("pool not instantiated")
        return cls(pool)

    async def initialize_data(
        self, toys: List[models.Toy], embeddings: List[models.Embedding]
    ) -> None:
        async with self.__pool.connect() as conn:
            # If the table already exists, drop it to avoid conflicts
            await conn.execute("DROP TABLE IF EXISTS products CASCADE")
            # Create a new table
            await conn.execute(
                """
                CREATE TABLE products(
                  product_id VARCHAR(1024) PRIMARY KEY,
                  product_name TEXT,
                  description TEXT,
                  list_price NUMERIC
                )
                """
            )
            # Insert all the data
            await conn.executemany(
                """INSERT INTO products VALUES ($1, $2, $3, $4)""",
                [
                    (t.product_id, t.product_name, t.description, t.list_price)
                    for t in toys
                ],
            )

            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute("DROP TABLE IF EXISTS product_embeddings")
            await conn.execute(
                """
                CREATE TABLE product_embeddings(
                    product_id VARCHAR(1024) NOT NULL REFERENCES products(product_id),
                    content TEXT,
                    embedding vector(768))
                """
            )
            # Insert all the data
            await conn.executemany(
                """INSERT INTO product_embeddings VALUES ($1, $2, $3)""",
                [(e.product_id, e.content, e.embedding) for e in embeddings],
            )

    async def export_data(self) -> Tuple[List[models.Toy], List[models.Embedding]]:
        toy_task = asyncio.create_task(self.__pool.fetch("""SELECT * FROM products"""))
        emb_task = asyncio.create_task(
            self.__pool.fetch("""SELECT * FROM product_embeddings""")
        )

        toys = [models.Toy.model_validate(dict(t)) for t in await toy_task]
        embeddings = [models.Embedding.model_validate(dict(v)) for v in await emb_task]

        return toys, embeddings

    async def semantic_similiarity_search(
        self, query_embedding: List[float32], similarity_theshold: float, top_k: int
    ) -> List[Dict[str, Any]]:
        results = await self.__pool.fetch(
            """
                WITH vector_matches AS (
                    SELECT product_id, 1 - (embedding <=> $1) AS similarity
                    FROM product_embeddings
                    WHERE 1 - (embedding <=> $1) > $2
                    ORDER BY similarity DESC
                    LIMIT $3
                )
                SELECT
                    product_name,
                    list_price,
                    description
                FROM products
                WHERE product_id IN (SELECT product_id FROM vector_matches)
            """,
            query_embedding,
            similarity_theshold,
            top_k,
            timeout=10,
        )

        results = [dict(r) for r in results]
        return results

    async def close(self):
        await self.__pool.close()
