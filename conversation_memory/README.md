# Conversation Memory

This directory contains the pgvector-backed long-term memory layer for `chat.py`.

Set these values in `.env` to enable it:

```dotenv
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/friend_memory
PGVECTOR_TABLE=conversation_memories
CONVERSATION_ID=default
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
MEMORY_TOP_K=5
```

The app creates the `vector` and `pgcrypto` extensions, an `agents` table, memory tables, and cosine HNSW indexes on startup. If `DATABASE_URL` is not set, the chatbot runs without long-term memory.

You can provision the Supabase database ahead of time with:

```bash
env/bin/python scripts/setup_supabase_database.py
```

Each authenticated user gets a default agent. Conversation memories and durable user memories are scoped to that agent, while `CONVERSATION_ID` stays as the raw conversation key. The setup script migrates older rows that used scoped IDs such as `user_123:default` by assigning them to `user_123`'s default agent and rewriting the stored conversation id to `default`.

Durable user memories are stored in `user_memories` with both `user_id` and `agent_id`. After enough raw user turns accumulate, the `summarize-memories` Supabase Edge Function extracts durable memories, saves them with embeddings, and deletes the raw conversation rows it processed.

Deploy the function with:

```bash
supabase functions deploy summarize-memories
```

Set these Supabase function secrets:

```bash
supabase secrets set OPENAI_API_KEY=... SUPABASE_SERVICE_ROLE_KEY=... MEMORY_FUNCTION_SECRET=...
```
