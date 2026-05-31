import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence

import psycopg
from openai import OpenAI
from psycopg import sql


@dataclass(frozen=True)
class RetrievedMemory:
    role: str
    content: str
    similarity: float
    created_at: datetime


@dataclass(frozen=True)
class RetrievedUserMemory:
    type: str
    content: str
    confidence: float
    similarity: float
    created_at: datetime


class ConversationMemory:
    def __init__(
        self,
        *,
        database_url: str,
        table_name: str,
        embedding_model: str,
        embedding_dimensions: int,
        top_k: int,
    ) -> None:
        self.database_url = database_url
        self.table_name = table_name
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.top_k = top_k

    @classmethod
    def from_env(cls) -> Optional["ConversationMemory"]:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return None

        return cls(
            database_url=database_url,
            table_name=os.getenv("PGVECTOR_TABLE", "conversation_memories"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1536")),
            top_k=int(os.getenv("MEMORY_TOP_K", "5")),
        )

    def setup(self) -> None:
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agents (
                        id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                        user_id text NOT NULL,
                        slug text NOT NULL DEFAULT 'default',
                        name text NOT NULL DEFAULT 'Milo',
                        created_at timestamptz NOT NULL DEFAULT now(),
                        updated_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS agents_user_slug_idx
                    ON agents (user_id, slug)
                    """
                )
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {table} (
                            id bigserial PRIMARY KEY,
                            agent_id uuid REFERENCES agents(id) ON DELETE CASCADE,
                            conversation_id text NOT NULL,
                            role text NOT NULL CHECK (role IN ('user', 'assistant')),
                            content text NOT NULL,
                            embedding vector({dimensions}) NOT NULL,
                            response_id text,
                            created_at timestamptz NOT NULL DEFAULT now()
                        )
                        """
                    ).format(
                        table=sql.Identifier(self.table_name),
                        dimensions=sql.Literal(self.embedding_dimensions),
                    )
                )
                cur.execute(
                    sql.SQL(
                        "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS agent_id uuid"
                    ).format(table=sql.Identifier(self.table_name))
                )
                cur.execute(
                    sql.SQL(
                        """
                        DO $$
                        BEGIN
                            IF NOT EXISTS (
                                SELECT 1
                                FROM pg_constraint
                                WHERE conname = {constraint_name}
                            ) THEN
                                ALTER TABLE {table}
                                ADD CONSTRAINT {constraint_ident}
                                FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE;
                            END IF;
                        END $$;
                        """
                    ).format(
                        table=sql.Identifier(self.table_name),
                        constraint_name=sql.Literal(
                            f"{self.table_name}_agent_id_fkey"
                        ),
                        constraint_ident=sql.Identifier(
                            f"{self.table_name}_agent_id_fkey"
                        ),
                    )
                )
                cur.execute(
                    sql.SQL(
                        """
                        CREATE INDEX IF NOT EXISTS {index}
                        ON {table}
                        USING hnsw (embedding vector_cosine_ops)
                        """
                    ).format(
                        index=sql.Identifier(f"{self.table_name}_embedding_hnsw_idx"),
                        table=sql.Identifier(self.table_name),
                    )
                )
                cur.execute(
                    sql.SQL("DROP INDEX IF EXISTS {index}").format(
                        index=sql.Identifier(
                            f"{self.table_name}_conversation_created_idx"
                        )
                    )
                )
                cur.execute(
                    sql.SQL(
                        """
                        CREATE INDEX IF NOT EXISTS {index}
                        ON {table} (agent_id, conversation_id, created_at DESC)
                        """
                    ).format(
                        index=sql.Identifier(
                            f"{self.table_name}_agent_conversation_created_idx"
                        ),
                        table=sql.Identifier(self.table_name),
                    )
                )
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS user_memories (
                            id bigserial PRIMARY KEY,
                            agent_id uuid REFERENCES agents(id) ON DELETE CASCADE,
                            user_id text NOT NULL,
                            type text NOT NULL CHECK (type IN ('preference', 'project', 'fact', 'instruction')),
                            content text NOT NULL,
                            confidence double precision NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                            embedding vector({dimensions}) NOT NULL,
                            source_conversation_id text,
                            created_at timestamptz NOT NULL DEFAULT now(),
                            updated_at timestamptz NOT NULL DEFAULT now()
                        )
                        """
                    ).format(dimensions=sql.Literal(self.embedding_dimensions))
                )
                cur.execute(
                    "ALTER TABLE user_memories ADD COLUMN IF NOT EXISTS agent_id uuid"
                )
                cur.execute(
                    """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1
                            FROM pg_constraint
                            WHERE conname = 'user_memories_agent_id_fkey'
                        ) THEN
                            ALTER TABLE user_memories
                            ADD CONSTRAINT user_memories_agent_id_fkey
                            FOREIGN KEY (agent_id)
                            REFERENCES agents(id) ON DELETE CASCADE;
                        END IF;
                    END $$;
                    """
                )
                cur.execute("DROP INDEX IF EXISTS user_memories_user_created_idx")
                cur.execute("DROP INDEX IF EXISTS user_memories_unique_content_idx")
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS user_memories_embedding_hnsw_idx
                    ON user_memories
                    USING hnsw (embedding vector_cosine_ops)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS user_memories_agent_created_idx
                    ON user_memories (agent_id, created_at DESC)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS user_memories_user_agent_created_idx
                    ON user_memories (user_id, agent_id, created_at DESC)
                    """
                )
                cur.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS user_memories_agent_unique_content_idx
                    ON user_memories (agent_id, type, content)
                    """
                )

    def get_or_create_default_agent(self, *, user_id: str) -> str:
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agents (user_id, slug, name)
                    VALUES (%s, 'default', 'Milo')
                    ON CONFLICT (user_id, slug)
                    DO UPDATE SET updated_at = agents.updated_at
                    RETURNING id
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
        if row is None:
            raise RuntimeError("Could not resolve default agent")
        return str(row[0])

    def remember(
        self,
        client: OpenAI,
        *,
        agent_id: str,
        conversation_id: str,
        role: str,
        content: str,
        response_id: Optional[str] = None,
    ) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        if not content.strip():
            return

        embedding = self._embed(client, content)
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {table}
                            (agent_id, conversation_id, role, content, embedding, response_id)
                        VALUES
                            (%s::uuid, %s, %s, %s, %s::vector, %s)
                        """
                    ).format(table=sql.Identifier(self.table_name)),
                    (
                        agent_id,
                        conversation_id,
                        role,
                        content,
                        self._vector_literal(embedding),
                        response_id,
                    ),
                )

    def retrieve(
        self,
        client: OpenAI,
        *,
        agent_id: str,
        conversation_id: str,
        query: str,
    ) -> List[RetrievedMemory]:
        if not query.strip():
            return []

        embedding = self._embed(client, query)
        vector = self._vector_literal(embedding)
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT role, content, 1 - (embedding <=> %s::vector) AS similarity, created_at
                        FROM {table}
                        WHERE agent_id = %s::uuid
                            AND conversation_id = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """
                    ).format(table=sql.Identifier(self.table_name)),
                    (vector, agent_id, conversation_id, vector, self.top_k),
                )
                rows = cur.fetchall()

        return [
            RetrievedMemory(
                role=row[0],
                content=row[1],
                similarity=float(row[2]),
                created_at=row[3],
            )
            for row in rows
        ]

    def retrieve_user_memories(
        self,
        client: OpenAI,
        *,
        agent_id: str,
        user_id: str,
        query: str,
    ) -> List[RetrievedUserMemory]:
        if not query.strip():
            return []

        embedding = self._embed(client, query)
        vector = self._vector_literal(embedding)
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT type, content, confidence, 1 - (embedding <=> %s::vector) AS similarity, created_at
                    FROM user_memories
                    WHERE agent_id = %s::uuid
                        AND user_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (
                        vector,
                        agent_id,
                        user_id,
                        vector,
                        int(os.getenv("USER_MEMORY_TOP_K", "5")),
                    ),
                )
                rows = cur.fetchall()

        return [
            RetrievedUserMemory(
                type=row[0],
                content=row[1],
                confidence=float(row[2]),
                similarity=float(row[3]),
                created_at=row[4],
            )
            for row in rows
        ]

    def _embed(self, client: OpenAI, text: str) -> Sequence[float]:
        response = client.embeddings.create(
            model=self.embedding_model,
            input=text,
            dimensions=self.embedding_dimensions,
        )
        return response.data[0].embedding

    @staticmethod
    def _vector_literal(embedding: Sequence[float]) -> str:
        return json.dumps(list(embedding))
