# Milo

Milo is a local AI assistant workspace built around a FastAPI-powered chatbot with a browser UI, file-aware prompts, optional Supabase authentication, and pgvector-backed long-term memory. The project demonstrates a practical assistant architecture that combines OpenAI's Responses API, retrieval-augmented conversation memory, local file handling, and a lightweight frontend.

## Highlights

- FastAPI backend with typed request and response models.
- OpenAI Responses API integration with configurable model, system prompt, and web search tools.
- Local browser UI for chatting with Milo, uploading files, selecting file context, and downloading generated outputs.
- Supabase authentication endpoints for signup, login, and bearer-token protected chat.
- pgvector memory layer for semantic retrieval of prior conversation turns.
- Durable user memory extraction through a Supabase Edge Function.
- CLI chat client with text and optional voice transcription modes.
- Unit tests covering memory context assembly and agent-scoped persistence behavior.

## Tech Stack

- **Backend:** Python, FastAPI, Pydantic, Uvicorn
- **AI:** OpenAI Responses API, embeddings, optional web search, audio transcription
- **Database:** PostgreSQL with pgvector, Supabase
- **Frontend:** HTML, CSS, vanilla JavaScript
- **Testing:** Python `unittest`

## Architecture

```text
Browser UI / CLI
      |
      v
FastAPI app (`chat.py`)
      |
      +-- OpenAI Responses API
      +-- Local upload/output storage (`.local_assistant/`)
      +-- Supabase Auth
      +-- ConversationMemory (`conversation_memory/store.py`)
              |
              v
        PostgreSQL + pgvector
              |
              v
Supabase Edge Function (`summarize-memories`)
```

The application can run as a local assistant without Supabase authentication by enabling development auth bypass. When `DATABASE_URL` is configured, the memory layer creates the required tables, pgvector indexes, and agent-scoped memory records automatically.

## Getting Started

### 1. Create a virtual environment

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-5.5
DEV_AUTH_DISABLED=true

# Optional: Supabase/Postgres memory
DATABASE_URL=postgresql://...
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key
PGVECTOR_TABLE=conversation_memories
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
MEMORY_TOP_K=5
```

### 3. Run the local app

```bash
env/bin/uvicorn chat:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The local UI stores uploaded files and assistant-generated outputs under `.local_assistant/`, which is ignored by git.

## Memory Setup

To provision the Supabase/Postgres schema manually, run:

```bash
env/bin/python scripts/setup_supabase_database.py
```

The setup script creates:

- `agents` for user-scoped assistant identities.
- `conversation_memories` for embedded raw conversation turns.
- `user_memories` for durable preferences, facts, projects, and instructions.
- HNSW cosine indexes for semantic retrieval.
- Row-level security policies for direct-client safety.

Durable memory extraction is handled by the Supabase Edge Function in `supabase/functions/summarize-memories`. Deploy it with:

```bash
supabase functions deploy summarize-memories
supabase secrets set OPENAI_API_KEY=... SUPABASE_SERVICE_ROLE_KEY=... MEMORY_FUNCTION_SECRET=...
```

## API Overview

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | `GET` | Reports service, auth, memory, and web search status. |
| `/auth/signup` | `POST` | Creates a Supabase-backed user account. |
| `/auth/login` | `POST` | Authenticates a user and returns a session. |
| `/auth/me` | `GET` | Returns the authenticated user. |
| `/chat` | `POST` | Authenticated chat endpoint. |
| `/local/chat` | `POST` | Development/local chat endpoint with file context support. |
| `/local/files` | `GET` | Lists local uploads and assistant outputs. |
| `/local/files/upload` | `POST` | Uploads a local file for prompt context. |
| `/local/files/{kind}/{file_id}/download` | `GET` | Downloads an uploaded or generated file. |
| `/local/files/{kind}/{file_id}` | `DELETE` | Deletes an uploaded or generated file. |

## CLI Usage

Run the terminal client:

```bash
env/bin/python cli_chat.py
```

Optional CLI settings:

```dotenv
CLI_INPUT_MODE=text
CLI_CONVERSATION_ID=default
CLI_SHOW_TIMING=true
VOICE_TRANSCRIPTION_MODEL=gpt-4o-transcribe
```

Set `CLI_INPUT_MODE=voice` to record microphone input and transcribe it before sending the turn to Milo.

## Configuration Reference

| Variable | Default | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | Required | API key used for chat, embeddings, and transcription. |
| `OPENAI_MODEL` | `gpt-5.5` | Model used for assistant responses. |
| `SYSTEM_PROMPT` | Built in | Overrides Milo's default assistant instructions. |
| `WEB_SEARCH_ENABLED` | `true` | Enables OpenAI web search tools. |
| `WEB_SEARCH_CONTEXT_SIZE` | `medium` | Search context size: `low`, `medium`, or `high`. |
| `WEB_SEARCH_MAX_TOOL_CALLS` | `3` | Maximum web search tool calls per response. |
| `DEV_AUTH_DISABLED` | `false` | Uses a development user instead of bearer-token auth. |
| `DEV_AUTH_USER_ID` | `dev-user` | User id used when development auth is enabled. |
| `DATABASE_URL` | Optional | Enables pgvector-backed conversation memory. |
| `PGVECTOR_TABLE` | `conversation_memories` | Table used for raw conversation memories. |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for semantic search. |
| `EMBEDDING_DIMENSIONS` | `1536` | Embedding vector dimensions. |
| `MEMORY_TOP_K` | `5` | Number of conversation memories retrieved per turn. |
| `USER_MEMORY_TOP_K` | `5` | Number of durable user memories retrieved per turn. |
| `MEMORY_SUMMARY_ENABLED` | `true` | Enables durable memory extraction calls. |
| `MEMORY_SUMMARY_THRESHOLD` | `20` | Raw user-turn count required before summarization. |
| `LOCAL_FILE_DIR` | `.local_assistant` | Local upload/output storage directory. |
| `LOCAL_UPLOAD_MAX_BYTES` | `26214400` | Maximum local upload size in bytes. |

## Testing

Run the unit tests with:

```bash
env/bin/python -m unittest
```

Current tests validate that retrieved conversation and durable user memories are included in the model instructions, and that memory records are scoped to the correct agent and conversation id.

## Project Structure

```text
.
├── chat.py                              # FastAPI app, auth, local files, chat orchestration
├── cli_chat.py                          # Terminal chat client with text/voice input
├── conversation_memory/                 # pgvector memory package
├── frontend/                            # Browser UI assets
├── scripts/setup_supabase_database.py   # Supabase/Postgres provisioning script
├── supabase/functions/summarize-memories/
│   └── index.ts                         # Durable memory extraction Edge Function
├── tests/                               # Unit tests
└── requirements.txt                     # Python dependencies
```