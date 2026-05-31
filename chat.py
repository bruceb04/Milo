import base64
import binascii
import mimetypes
import os
import re
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional, Sequence

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field
from supabase import Client as SupabaseClient
from supabase import create_client

from conversation_memory import ConversationMemory, RetrievedMemory, RetrievedUserMemory


DEFAULT_SYSTEM_PROMPT = """
You are Milo, a helpful, concise chatbot.
Answer clearly, ask clarifying questions when needed, and keep a friendly tone.
""".strip()

load_dotenv()
auth_scheme = HTTPBearer(auto_error=False)
openai_client: Optional[OpenAI] = None
supabase_client: Optional[SupabaseClient] = None
conversation_memory: Optional[ConversationMemory] = None


def build_client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def build_supabase_client() -> SupabaseClient:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")
    return create_client(supabase_url, supabase_key)


def get_system_prompt() -> str:
    return os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_web_search_tools() -> list[dict[str, Any]]:
    if not env_bool("WEB_SEARCH_ENABLED", True):
        return []

    search_context_size = os.getenv("WEB_SEARCH_CONTEXT_SIZE", "medium").strip().lower()
    if search_context_size not in {"low", "medium", "high"}:
        search_context_size = "medium"

    return [
        {
            "type": "web_search",
            "search_context_size": search_context_size,
        }
    ]


def get_web_search_max_tool_calls() -> int:
    try:
        return int(os.getenv("WEB_SEARCH_MAX_TOOL_CALLS", "3"))
    except ValueError:
        return 3


def get_memory_summary_threshold() -> int:
    try:
        return int(os.getenv("MEMORY_SUMMARY_THRESHOLD", "20"))
    except ValueError:
        return 20


def build_system_prompt(
    system_prompt: str,
    retrieved_memories: Sequence[RetrievedMemory],
    user_memories: Sequence[RetrievedUserMemory] = (),
) -> str:
    if not retrieved_memories and not user_memories:
        return system_prompt

    memory_lines = ["Use the following retrieved memory only when it helps answer the user."]
    if user_memories:
        memory_lines.append("Relevant durable user memories:")
        for memory in user_memories:
            memory_lines.append(
                f"- {memory.type} (confidence {memory.confidence:.2f}, similarity {memory.similarity:.2f}): {memory.content}"
            )

    if retrieved_memories:
        memory_lines.append("Relevant previous conversation details:")
    for memory in retrieved_memories:
        created_at = memory.created_at.strftime("%Y-%m-%d %H:%M")
        memory_lines.append(
            f"- {memory.role} at {created_at} (similarity {memory.similarity:.2f}): {memory.content}"
        )

    return f"{system_prompt}\n\n" + "\n".join(memory_lines)


def ask_openai(
    client: OpenAI,
    user_message: str,
    *,
    system_prompt: str,
    model: str,
    previous_response_id: Optional[str] = None,
    tools: Optional[Sequence[dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    max_tool_calls: Optional[int] = None,
):
    request = {
        "model": model,
        "instructions": system_prompt,
        "input": user_message,
    }
    if previous_response_id:
        request["previous_response_id"] = previous_response_id
    if tools:
        request["tools"] = tools
    if tool_choice:
        request["tool_choice"] = tool_choice
    if max_tool_calls:
        request["max_tool_calls"] = max_tool_calls

    return client.responses.create(**request)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: str = Field(default="default", min_length=1)
    previous_response_id: Optional[str] = None


class LocalFileRecord(BaseModel):
    id: str
    filename: str
    kind: str
    size_bytes: int
    created_at: str
    download_url: str
    content_type: str = "application/octet-stream"


class ChatResponse(BaseModel):
    message: str
    conversation_id: str
    response_id: str
    output_file: Optional[LocalFileRecord] = None
    memory_saved: bool = False
    memory_error: Optional[str] = None


class LocalUploadRequest(BaseModel):
    filename: str = Field(..., min_length=1)
    content_base64: str = Field(..., min_length=1)


class LocalFilesResponse(BaseModel):
    uploads: list[LocalFileRecord]
    outputs: list[LocalFileRecord]


class LocalChatRequest(ChatRequest):
    file_ids: list[str] = Field(default_factory=list)


class AuthCredentials(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)


class AuthenticatedUser(BaseModel):
    id: str
    email: Optional[str] = None


def get_dev_user() -> AuthenticatedUser:
    return AuthenticatedUser(
        id=os.getenv("DEV_AUTH_USER_ID", "dev-user"),
        email=os.getenv("DEV_AUTH_EMAIL", "dev@example.local"),
    )


class AuthSession(BaseModel):
    access_token: str
    refresh_token: str
    expires_in: int
    expires_at: Optional[int] = None
    token_type: str


class AuthResult(BaseModel):
    user: Optional[AuthenticatedUser] = None
    session: Optional[AuthSession] = None
    message: str


def build_auth_result(auth_response: Any, message: str) -> AuthResult:
    user = getattr(auth_response, "user", None)
    session = getattr(auth_response, "session", None)

    return AuthResult(
        user=AuthenticatedUser(
            id=getattr(user, "id"),
            email=getattr(user, "email", None),
        )
        if user and getattr(user, "id", None)
        else None,
        session=AuthSession(
            access_token=getattr(session, "access_token"),
            refresh_token=getattr(session, "refresh_token"),
            expires_in=getattr(session, "expires_in"),
            expires_at=getattr(session, "expires_at", None),
            token_type=getattr(session, "token_type"),
        )
        if session
        else None,
        message=message,
    )


def get_required_openai_client() -> OpenAI:
    if openai_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI client is not configured",
        )
    return openai_client


def get_required_supabase_client() -> SupabaseClient:
    if supabase_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase client is not configured",
        )
    return supabase_client


def get_authenticated_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(auth_scheme),
) -> AuthenticatedUser:
    if env_bool("DEV_AUTH_DISABLED", False):
        return get_dev_user()

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    supabase = get_required_supabase_client()
    try:
        auth_response = supabase.auth.get_user(credentials.credentials)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from error

    user = getattr(auth_response, "user", None)
    user_id = getattr(user, "id", None)
    if not user or not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return AuthenticatedUser(id=user_id, email=getattr(user, "email", None))


def get_local_file_root() -> Path:
    return Path(os.getenv("LOCAL_FILE_DIR", ".local_assistant"))


def get_local_file_dir(kind: str) -> Path:
    if kind not in {"uploads", "outputs"}:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unknown file area",
        )
    path = get_local_file_root() / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(filename: str) -> str:
    clean_name = Path(filename).name.strip()
    clean_name = re.sub(r"[^A-Za-z0-9._ -]+", "_", clean_name)
    clean_name = re.sub(r"\s+", " ", clean_name).strip(" .")
    return clean_name or "file"


def local_file_record(kind: str, path: Path) -> LocalFileRecord:
    stat = path.stat()
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return LocalFileRecord(
        id=path.name,
        filename=path.name,
        kind=kind,
        size_bytes=stat.st_size,
        created_at=datetime.fromtimestamp(stat.st_ctime).isoformat(timespec="seconds"),
        download_url=f"/local/files/{kind}/{path.name}/download",
        content_type=content_type,
    )


def list_local_files(kind: str) -> list[LocalFileRecord]:
    directory = get_local_file_dir(kind)
    records = []
    for path in directory.iterdir():
        if path.is_file():
            records.append(local_file_record(kind, path))
    return sorted(records, key=lambda record: record.created_at, reverse=True)


def resolve_local_file(kind: str, file_id: str) -> Path:
    if "/" in file_id or "\\" in file_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file id",
        )
    path = get_local_file_dir(kind) / file_id
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found",
        )
    return path


def decode_text_excerpt(path: Path, max_chars: int) -> Optional[str]:
    raw = path.read_bytes()
    if not raw:
        return ""

    text: Optional[str] = None
    for encoding in ("utf-8", "utf-16"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        content_type = mimetypes.guess_type(path.name)[0] or ""
        text_extensions = {
            ".csv",
            ".json",
            ".log",
            ".md",
            ".py",
            ".sql",
            ".text",
            ".toml",
            ".ts",
            ".txt",
            ".yaml",
            ".yml",
        }
        if not content_type.startswith("text/") and path.suffix.lower() not in text_extensions:
            return None
        text = raw.decode("latin-1", errors="ignore")

    printable = sum(1 for char in text[:4000] if char.isprintable() or char.isspace())
    sample_size = min(len(text), 4000)
    if sample_size and printable / sample_size < 0.85:
        return None

    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[File excerpt truncated.]"
    return text


def build_uploaded_file_context(file_ids: Sequence[str]) -> str:
    if not file_ids:
        return ""

    max_total_chars = int(os.getenv("LOCAL_FILE_CONTEXT_CHARS", "120000"))
    max_per_file = int(os.getenv("LOCAL_FILE_CONTEXT_CHARS_PER_FILE", "40000"))
    remaining = max_total_chars
    sections = []

    for file_id in file_ids:
        if remaining <= 0:
            sections.append("[Additional files omitted because the file context limit was reached.]")
            break

        path = resolve_local_file("uploads", file_id)
        excerpt = decode_text_excerpt(path, min(max_per_file, remaining))
        if excerpt is None:
            sections.append(
                f"--- {path.name} ---\n"
                "[This uploaded file appears to be binary, so its contents were not included.]"
            )
            continue

        remaining -= len(excerpt)
        sections.append(f"--- {path.name} ---\n{excerpt}")

    if not sections:
        return ""
    return "Uploaded local file context:\n\n" + "\n\n".join(sections)


def save_assistant_output(message: str) -> LocalFileRecord:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    filename = f"milo-{timestamp}.txt"
    path = get_local_file_dir("outputs") / filename
    path.write_text(message, encoding="utf-8")
    return local_file_record("outputs", path)


def user_requested_output_file(message: str) -> bool:
    normalized = message.strip().lower()
    if not normalized:
        return False

    file_request_patterns = [
        r"\b(create|make|generate|produce|write|save|export)\b.{0,80}\b(file|download|document)\b",
        r"\b(file|download|document)\b.{0,80}\b(create|make|generate|produce|write|save|export)\b",
        r"\b(save|export)\s+(this|that|it|the response|your response)\s+as\b",
        r"\bdownloadable\s+(file|document|copy|version)\b",
        r"\b(make|create|generate)\s+(this|that|it|the response|your response)\s+downloadable\b",
        r"\b(as|in)\s+(a\s+)?(txt|text|md|markdown|csv|json|html|pdf)\s+file\b",
    ]
    return any(re.search(pattern, normalized) for pattern in file_request_patterns)


def complete_chat_turn(
    *,
    user_message: str,
    conversation_id: str,
    previous_response_id: Optional[str],
    user: AuthenticatedUser,
    client: OpenAI,
    file_context: str = "",
    save_output_file: bool = False,
    require_memory_save: bool = False,
) -> ChatResponse:
    if not user_message:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message cannot be empty",
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    system_prompt = get_system_prompt()
    tools = build_web_search_tools()
    tool_choice = "auto" if tools else None
    max_tool_calls = get_web_search_max_tool_calls() if tools else None

    agent_id = None
    retrieved_memories = []
    user_memories = []
    if conversation_memory:
        try:
            agent_id = conversation_memory.get_or_create_default_agent(user_id=user.id)
            user_memories = conversation_memory.retrieve_user_memories(
                client,
                agent_id=agent_id,
                user_id=user.id,
                query=user_message,
            )
            retrieved_memories = conversation_memory.retrieve(
                client,
                agent_id=agent_id,
                conversation_id=conversation_id,
                query=user_message,
            )
        except Exception as error:
            print(f"Memory retrieval skipped: {error}")

    openai_input = user_message
    if file_context:
        openai_input = f"{user_message}\n\n{file_context}"

    response = ask_openai(
        client,
        openai_input,
        system_prompt=build_system_prompt(
            system_prompt,
            retrieved_memories,
            user_memories,
        ),
        model=model,
        previous_response_id=previous_response_id,
        tools=tools,
        tool_choice=tool_choice,
        max_tool_calls=max_tool_calls,
    )

    memory_saved = False
    memory_error = None
    if conversation_memory and agent_id:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        try:
            conversation_memory.remember(
                client,
                agent_id=agent_id,
                conversation_id=conversation_id,
                role="user",
                content=f"[{timestamp}] {user_message}",
                response_id=response.id,
            )
            conversation_memory.remember(
                client,
                agent_id=agent_id,
                conversation_id=conversation_id,
                role="assistant",
                content=f"[{timestamp}] {response.output_text}",
                response_id=response.id,
            )
            memory_saved = True
        except Exception as error:
            memory_error = str(error)
            print(f"Memory save skipped: {error}")
    else:
        memory_error = "Supabase memory is not configured"
        if conversation_memory and not agent_id:
            memory_error = "Default agent could not be resolved"

    if require_memory_save and not memory_saved:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Supabase memory save failed: {memory_error}",
        )

    if agent_id:
        invoke_memory_summary_function(
            supabase_client,
            agent_id=agent_id,
            user_id=user.id,
            conversation_id=conversation_id,
        )

    output_file = save_assistant_output(response.output_text) if save_output_file else None
    return ChatResponse(
        message=response.output_text,
        conversation_id=conversation_id,
        response_id=response.id,
        output_file=output_file,
        memory_saved=memory_saved,
        memory_error=memory_error,
    )


def invoke_memory_summary_function(
    supabase: Optional[SupabaseClient],
    *,
    agent_id: str,
    user_id: str,
    conversation_id: str,
) -> None:
    if not env_bool("MEMORY_SUMMARY_ENABLED", True):
        return
    if supabase is None:
        return

    function_name = os.getenv("MEMORY_SUMMARY_FUNCTION", "summarize-memories")
    supabase_url = os.getenv("SUPABASE_URL")
    if not supabase_url:
        print("Memory summary function skipped: SUPABASE_URL is not configured")
        return

    secret = os.getenv("MEMORY_FUNCTION_SECRET")
    headers = {"Content-Type": "application/json"}
    supabase_key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
    if supabase_key:
        headers["Authorization"] = f"Bearer {supabase_key}"
    if secret:
        headers["x-memory-function-secret"] = secret

    payload = {
        "agent_id": agent_id,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "threshold": get_memory_summary_threshold(),
    }
    function_url = f"{supabase_url.rstrip('/')}/functions/v1/{function_name}"
    timeout = float(os.getenv("MEMORY_SUMMARY_TIMEOUT_SECONDS", "30"))

    try:
        response = httpx.post(
            function_url,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        if response.status_code >= 400:
            print(
                "Memory summary function skipped: "
                f"HTTP {response.status_code} {response.text[:500]}"
            )
    except Exception as error:
        print(f"Memory summary function skipped: {error}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global openai_client, supabase_client, conversation_memory

    openai_client = build_client()
    try:
        supabase_client = build_supabase_client()
    except RuntimeError as error:
        if not env_bool("DEV_AUTH_DISABLED", False):
            raise
        print(f"Supabase disabled for development auth bypass: {error}")
        supabase_client = None

    conversation_memory = ConversationMemory.from_env()
    if conversation_memory:
        try:
            conversation_memory.setup()
        except Exception as error:
            print(f"Memory disabled: {error}")
            conversation_memory = None

    get_local_file_dir("uploads")
    get_local_file_dir("outputs")

    yield


app = FastAPI(title="Milo Chatbot API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, bool]:
    return {
        "ok": True,
        "dev_auth_disabled": env_bool("DEV_AUTH_DISABLED", False),
        "memory_enabled": conversation_memory is not None,
        "web_search_enabled": bool(build_web_search_tools()),
    }


@app.post("/auth/signup", response_model=AuthResult)
def signup(
    credentials: AuthCredentials,
    supabase: SupabaseClient = Depends(get_required_supabase_client),
) -> AuthResult:
    try:
        auth_response = supabase.auth.sign_up(
            {
                "email": credentials.email,
                "password": credentials.password,
            }
        )
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not create user",
        ) from error

    message = "User created"
    if not getattr(auth_response, "session", None):
        message = "User created. Check your email to confirm the account."
    return build_auth_result(auth_response, message)


@app.post("/auth/login", response_model=AuthResult)
def login(
    credentials: AuthCredentials,
    supabase: SupabaseClient = Depends(get_required_supabase_client),
) -> AuthResult:
    try:
        auth_response = supabase.auth.sign_in_with_password(
            {
                "email": credentials.email,
                "password": credentials.password,
            }
        )
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        ) from error

    return build_auth_result(auth_response, "Logged in")


@app.get("/auth/me", response_model=AuthenticatedUser)
def me(user: AuthenticatedUser = Depends(get_authenticated_user)) -> AuthenticatedUser:
    return user


@app.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    user: AuthenticatedUser = Depends(get_authenticated_user),
    client: OpenAI = Depends(get_required_openai_client),
) -> ChatResponse:
    return complete_chat_turn(
        user_message=payload.message.strip(),
        conversation_id=payload.conversation_id,
        previous_response_id=payload.previous_response_id,
        user=user,
        client=client,
    )


@app.get("/local/files", response_model=LocalFilesResponse)
def local_files() -> LocalFilesResponse:
    return LocalFilesResponse(
        uploads=list_local_files("uploads"),
        outputs=list_local_files("outputs"),
    )


@app.post("/local/files/upload", response_model=LocalFileRecord)
def local_upload(payload: LocalUploadRequest) -> LocalFileRecord:
    try:
        content_base64 = payload.content_base64.split(",", 1)[-1]
        content = base64.b64decode(content_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file content must be valid base64",
        ) from error

    max_bytes = int(os.getenv("LOCAL_UPLOAD_MAX_BYTES", str(25 * 1024 * 1024)))
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file is larger than {max_bytes} bytes",
        )

    safe_name = sanitize_filename(payload.filename)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = get_local_file_dir("uploads") / f"{timestamp}-{safe_name}"
    path.write_bytes(content)
    return local_file_record("uploads", path)


@app.get("/local/files/{kind}/{file_id}/download")
def local_download(kind: str, file_id: str) -> FileResponse:
    path = resolve_local_file(kind, file_id)
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, filename=path.name, media_type=media_type)


@app.delete("/local/files/{kind}/{file_id}")
def local_delete(kind: str, file_id: str) -> dict[str, bool]:
    path = resolve_local_file(kind, file_id)
    path.unlink()
    return {"ok": True}


@app.post("/local/chat", response_model=ChatResponse)
def local_chat(
    payload: LocalChatRequest,
    client: OpenAI = Depends(get_required_openai_client),
) -> ChatResponse:
    user_message = payload.message.strip()
    return complete_chat_turn(
        user_message=user_message,
        conversation_id=payload.conversation_id,
        previous_response_id=payload.previous_response_id,
        user=get_dev_user(),
        client=client,
        file_context=build_uploaded_file_context(payload.file_ids),
        save_output_file=user_requested_output_file(user_message),
        require_memory_save=True,
    )


frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("chat:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
