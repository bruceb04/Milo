import os
from typing import Iterable

import psycopg
from dotenv import load_dotenv
from psycopg import sql


DEFAULT_TABLE = "conversation_memories"
DEFAULT_DIMENSIONS = 1536
DEFAULT_AGENT_SLUG = "default"
DEFAULT_AGENT_NAME = "Milo"


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is required. Use the Supabase Postgres connection string."
        )
    return database_url


def get_dev_user_id() -> str:
    return os.getenv("DEV_AUTH_USER_ID", "dev-user")


def create_extensions(cur: psycopg.Cursor) -> None:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")


def create_agents_table(cur: psycopg.Cursor) -> None:
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


def create_memory_table(
    cur: psycopg.Cursor,
    *,
    table_name: str,
    embedding_dimensions: int,
) -> None:
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
            table=sql.Identifier(table_name),
            dimensions=sql.Literal(embedding_dimensions),
        )
    )
    cur.execute(
        sql.SQL(
            "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS agent_id uuid"
        ).format(table=sql.Identifier(table_name))
    )


def create_user_memories_table(
    cur: psycopg.Cursor,
    *,
    embedding_dimensions: int,
) -> None:
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
        ).format(dimensions=sql.Literal(embedding_dimensions))
    )
    cur.execute("ALTER TABLE user_memories ADD COLUMN IF NOT EXISTS agent_id uuid")


def ensure_agent_foreign_keys(cur: psycopg.Cursor, *, table_name: str) -> None:
    cur.execute(
        sql.SQL(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conname = {conversation_fk}
                ) THEN
                    ALTER TABLE {table}
                    ADD CONSTRAINT {conversation_fk_ident}
                    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE;
                END IF;
            END $$;
            """
        ).format(
            table=sql.Identifier(table_name),
            conversation_fk=sql.Literal(f"{table_name}_agent_id_fkey"),
            conversation_fk_ident=sql.Identifier(f"{table_name}_agent_id_fkey"),
        )
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
                FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE;
            END IF;
        END $$;
        """
    )


def create_default_agent(cur: psycopg.Cursor, *, user_id: str) -> None:
    cur.execute(
        """
        INSERT INTO agents (user_id, slug, name)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_id, slug) DO NOTHING
        """,
        (user_id, DEFAULT_AGENT_SLUG, DEFAULT_AGENT_NAME),
    )


def backfill_default_agents(
    cur: psycopg.Cursor,
    *,
    table_name: str,
    dev_user_id: str,
) -> None:
    create_default_agent(cur, user_id=dev_user_id)
    cur.execute(
        """
        INSERT INTO agents (user_id, slug, name)
        SELECT DISTINCT user_id, 'default', 'Milo'
        FROM user_memories
        WHERE user_id IS NOT NULL AND btrim(user_id) <> ''
        ON CONFLICT (user_id, slug) DO NOTHING
        """
    )
    cur.execute(
        sql.SQL(
            """
            INSERT INTO agents (user_id, slug, name)
            SELECT DISTINCT split_part(conversation_id, ':', 1), 'default', 'Milo'
            FROM {table}
            WHERE position(':' in conversation_id) > 0
                AND split_part(conversation_id, ':', 1) <> ''
            ON CONFLICT (user_id, slug) DO NOTHING
            """
        ).format(table=sql.Identifier(table_name))
    )


def backfill_agent_ids(
    cur: psycopg.Cursor,
    *,
    table_name: str,
    dev_user_id: str,
) -> None:
    cur.execute(
        sql.SQL(
            """
            UPDATE {table} AS memories
            SET agent_id = agents.id,
                conversation_id = substr(
                    memories.conversation_id,
                    position(':' in memories.conversation_id) + 1
                )
            FROM agents
            WHERE memories.agent_id IS NULL
                AND position(':' in memories.conversation_id) > 0
                AND agents.user_id = split_part(memories.conversation_id, ':', 1)
                AND agents.slug = 'default'
            """
        ).format(table=sql.Identifier(table_name))
    )
    cur.execute(
        sql.SQL(
            """
            UPDATE {table} AS memories
            SET agent_id = agents.id
            FROM agents
            WHERE memories.agent_id IS NULL
                AND agents.user_id = %s
                AND agents.slug = 'default'
            """
        ).format(table=sql.Identifier(table_name)),
        (dev_user_id,),
    )
    cur.execute(
        """
        UPDATE user_memories AS memories
        SET agent_id = agents.id
        FROM agents
        WHERE memories.agent_id IS NULL
            AND agents.user_id = memories.user_id
            AND agents.slug = 'default'
        """
    )
    cur.execute(
        """
        UPDATE user_memories AS memories
        SET agent_id = agents.id,
            user_id = %s
        FROM agents
        WHERE memories.agent_id IS NULL
            AND agents.user_id = %s
            AND agents.slug = 'default'
        """,
        (dev_user_id, dev_user_id),
    )


def require_agent_ids(cur: psycopg.Cursor, *, table_name: str) -> None:
    cur.execute(
        sql.SQL("ALTER TABLE {table} ALTER COLUMN agent_id SET NOT NULL").format(
            table=sql.Identifier(table_name)
        )
    )
    cur.execute("ALTER TABLE user_memories ALTER COLUMN agent_id SET NOT NULL")


def drop_legacy_indexes(cur: psycopg.Cursor, *, table_name: str) -> None:
    legacy_indexes = (
        f"{table_name}_conversation_created_idx",
        "user_memories_user_created_idx",
        "user_memories_unique_content_idx",
    )
    for index_name in legacy_indexes:
        cur.execute(
            sql.SQL("DROP INDEX IF EXISTS {index}").format(
                index=sql.Identifier(index_name)
            )
        )


def create_indexes(cur: psycopg.Cursor, *, table_name: str) -> None:
    indexes: Iterable[tuple[str, sql.SQL]] = (
        (
            f"{table_name}_embedding_hnsw_idx",
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {index}
                ON {table}
                USING hnsw (embedding vector_cosine_ops)
                """
            ),
        ),
        (
            f"{table_name}_agent_conversation_created_idx",
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {index}
                ON {table} (agent_id, conversation_id, created_at DESC)
                """
            ),
        ),
        (
            f"{table_name}_response_id_idx",
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {index}
                ON {table} (response_id)
                WHERE response_id IS NOT NULL
                """
            ),
        ),
    )

    for index_name, statement in indexes:
        cur.execute(
            statement.format(
                index=sql.Identifier(index_name),
                table=sql.Identifier(table_name),
            )
        )


def create_user_memory_indexes(cur: psycopg.Cursor) -> None:
    indexes: Iterable[tuple[str, sql.SQL]] = (
        (
            "user_memories_embedding_hnsw_idx",
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {index}
                ON user_memories
                USING hnsw (embedding vector_cosine_ops)
                """
            ),
        ),
        (
            "user_memories_agent_created_idx",
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {index}
                ON user_memories (agent_id, created_at DESC)
                """
            ),
        ),
        (
            "user_memories_user_agent_created_idx",
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {index}
                ON user_memories (user_id, agent_id, created_at DESC)
                """
            ),
        ),
        (
            "user_memories_agent_unique_content_idx",
            sql.SQL(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS {index}
                ON user_memories (agent_id, type, content)
                """
            ),
        ),
    )

    for index_name, statement in indexes:
        cur.execute(statement.format(index=sql.Identifier(index_name)))


def enable_rls_for_direct_client_safety(
    cur: psycopg.Cursor,
    *,
    table_name: str,
) -> None:
    cur.execute(
        sql.SQL("ALTER TABLE {table} ENABLE ROW LEVEL SECURITY").format(
            table=sql.Identifier(table_name)
        )
    )


def create_rls_policies(cur: psycopg.Cursor, *, table_name: str) -> None:
    cur.execute(
        """
        CREATE POLICY agents_owner_access
        ON agents
        FOR ALL
        USING (user_id = auth.uid()::text)
        WITH CHECK (user_id = auth.uid()::text)
        """
    )
    cur.execute(
        sql.SQL(
            """
            CREATE POLICY {policy}
            ON {table}
            FOR ALL
            USING (
                EXISTS (
                    SELECT 1
                    FROM agents
                    WHERE agents.id = agent_id
                        AND agents.user_id = auth.uid()::text
                )
            )
            WITH CHECK (
                EXISTS (
                    SELECT 1
                    FROM agents
                    WHERE agents.id = agent_id
                        AND agents.user_id = auth.uid()::text
                )
            )
            """
        ).format(
            policy=sql.Identifier(f"{table_name}_owner_access"),
            table=sql.Identifier(table_name),
        )
    )
    cur.execute(
        """
        CREATE POLICY user_memories_owner_access
        ON user_memories
        FOR ALL
        USING (
            user_id = auth.uid()::text
            AND EXISTS (
                SELECT 1
                FROM agents
                WHERE agents.id = user_memories.agent_id
                    AND agents.user_id = auth.uid()::text
            )
        )
        WITH CHECK (
            user_id = auth.uid()::text
            AND EXISTS (
                SELECT 1
                FROM agents
                WHERE agents.id = user_memories.agent_id
                    AND agents.user_id = auth.uid()::text
            )
        )
        """
    )


def recreate_rls_policies(cur: psycopg.Cursor, *, table_name: str) -> None:
    policy_names = (
        ("agents", "agents_owner_access"),
        (table_name, f"{table_name}_owner_access"),
        ("user_memories", "user_memories_owner_access"),
    )
    for policy_table, policy_name in policy_names:
        cur.execute(
            sql.SQL("DROP POLICY IF EXISTS {policy} ON {table}").format(
                policy=sql.Identifier(policy_name),
                table=sql.Identifier(policy_table),
            )
        )
    create_rls_policies(cur, table_name=table_name)


def setup_database() -> None:
    load_dotenv()

    table_name = os.getenv("PGVECTOR_TABLE", DEFAULT_TABLE)
    embedding_dimensions = env_int("EMBEDDING_DIMENSIONS", DEFAULT_DIMENSIONS)
    dev_user_id = get_dev_user_id()

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            create_extensions(cur)
            create_agents_table(cur)
            create_memory_table(
                cur,
                table_name=table_name,
                embedding_dimensions=embedding_dimensions,
            )
            create_user_memories_table(
                cur,
                embedding_dimensions=embedding_dimensions,
            )
            ensure_agent_foreign_keys(cur, table_name=table_name)
            backfill_default_agents(
                cur,
                table_name=table_name,
                dev_user_id=dev_user_id,
            )
            backfill_agent_ids(
                cur,
                table_name=table_name,
                dev_user_id=dev_user_id,
            )
            require_agent_ids(cur, table_name=table_name)
            drop_legacy_indexes(cur, table_name=table_name)
            create_indexes(cur, table_name=table_name)
            create_user_memory_indexes(cur)
            enable_rls_for_direct_client_safety(cur, table_name="agents")
            enable_rls_for_direct_client_safety(cur, table_name=table_name)
            enable_rls_for_direct_client_safety(cur, table_name="user_memories")
            recreate_rls_policies(cur, table_name=table_name)

    print(
        "Supabase database setup complete: "
        f"table={table_name}, embedding_dimensions={embedding_dimensions}, "
        f"default_dev_user={dev_user_id}"
    )


if __name__ == "__main__":
    setup_database()
