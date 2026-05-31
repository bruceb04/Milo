import os
import tempfile
import wave
from datetime import datetime
from pathlib import Path
from threading import Event
from time import perf_counter
from typing import Optional

import sounddevice as sd
from conversation_memory import ConversationMemory
from openai import OpenAI

from chat import (
    ask_openai,
    build_client,
    build_system_prompt,
    build_web_search_tools,
    build_supabase_client,
    env_bool,
    get_dev_user,
    get_system_prompt,
    get_web_search_max_tool_calls,
    invoke_memory_summary_function,
)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def setup_memory() -> Optional[ConversationMemory]:
    memory = ConversationMemory.from_env()
    if not memory:
        return None

    try:
        memory.setup()
    except Exception as error:
        print(f"Memory disabled: {error}")
        return None

    return memory


def get_input_mode() -> str:
    input_mode = os.getenv("CLI_INPUT_MODE", "text").strip().lower()
    if input_mode not in {"voice", "text"}:
        return "text"
    return input_mode


def print_timing(show_timing: bool, label: str, elapsed_seconds: float) -> None:
    if show_timing:
        print(f"[timing] {label}: {elapsed_seconds:.2f}s")


def record_audio_file() -> Path:
    sample_rate = env_int("VOICE_SAMPLE_RATE", 16000)
    channels = env_int("VOICE_CHANNELS", 1)
    stop_recording = Event()
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    print("\nPress Enter to start recording.")
    input()
    print("Recording... press Enter to stop.")

    with wave.open(str(temp_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        def callback(indata, frames, time, status):
            if status:
                print(f"Audio warning: {status}")
            wav_file.writeframes(indata.copy().tobytes())

        with sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
            callback=callback,
        ):
            input()
            stop_recording.set()

    if not stop_recording.is_set():
        raise RuntimeError("Recording did not stop cleanly")

    return temp_path


def transcribe_audio(client: OpenAI, audio_path: Path) -> str:
    transcription_model = os.getenv("VOICE_TRANSCRIPTION_MODEL", "gpt-4o-transcribe")
    with audio_path.open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=transcription_model,
            file=audio_file,
        )
    return getattr(transcription, "text", str(transcription)).strip()


def get_voice_input(client: OpenAI) -> str:
    audio_path = record_audio_file()
    try:
        print("Transcribing...")
        return transcribe_audio(client, audio_path)
    finally:
        try:
            audio_path.unlink()
        except FileNotFoundError:
            pass


def get_user_message(client: OpenAI, input_mode: str) -> str:
    if input_mode == "text":
        return input("\nYou: ").strip()

    transcript = get_voice_input(client)
    if transcript:
        print(f"\nYou: {transcript}")
    return transcript


def main() -> None:
    client = build_client()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    system_prompt = get_system_prompt()
    conversation_id = os.getenv(
        "CLI_CONVERSATION_ID",
        os.getenv("CONVERSATION_ID", "default"),
    )
    memory = setup_memory()
    dev_user = get_dev_user()
    agent_id = None
    if memory:
        try:
            agent_id = memory.get_or_create_default_agent(user_id=dev_user.id)
        except Exception as error:
            print(f"Memory disabled: could not resolve default agent: {error}")
            memory = None
    supabase_client = None
    try:
        supabase_client = build_supabase_client()
    except RuntimeError:
        pass
    tools = build_web_search_tools()
    tool_choice = "auto" if tools else None
    max_tool_calls = get_web_search_max_tool_calls() if tools else None
    input_mode = get_input_mode()
    show_timing = env_bool("CLI_SHOW_TIMING", True)
    previous_response_id = None

    print("Chatbot CLI ready. Type 'exit' or 'quit' to stop in text mode.")
    if input_mode == "voice":
        print("Voice mode is active. Press Ctrl+C to stop.")

    while True:
        turn_start = perf_counter()
        input_start = perf_counter()
        user_message = get_user_message(client, input_mode)
        print_timing(
            show_timing,
            "input/transcription",
            perf_counter() - input_start,
        )
        if user_message.lower() in {"exit", "quit"}:
            break
        if not user_message:
            print("No input detected. Try again.")
            print_timing(show_timing, "total turn", perf_counter() - turn_start)
            continue

        memory_retrieval_start = perf_counter()
        retrieved_memories = []
        user_memories = []
        if memory:
            try:
                user_memories = memory.retrieve_user_memories(
                    client,
                    agent_id=agent_id,
                    user_id=dev_user.id,
                    query=user_message,
                )
                retrieved_memories = memory.retrieve(
                    client,
                    agent_id=agent_id,
                    conversation_id=conversation_id,
                    query=user_message,
                )
            except Exception as error:
                print(f"Memory retrieval skipped: {error}")
        print_timing(
            show_timing,
            "memory retrieval",
            perf_counter() - memory_retrieval_start,
        )

        response_start = perf_counter()
        response = ask_openai(
            client,
            user_message,
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
        print_timing(show_timing, "OpenAI response", perf_counter() - response_start)
        previous_response_id = response.id

        print(f"\nAssistant: {response.output_text}")

        memory_save_start = perf_counter()
        if memory:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            try:
                memory.remember(
                    client,
                    agent_id=agent_id,
                    conversation_id=conversation_id,
                    role="user",
                    content=f"[{timestamp}] {user_message}",
                    response_id=response.id,
                )
                memory.remember(
                    client,
                    agent_id=agent_id,
                    conversation_id=conversation_id,
                    role="assistant",
                    content=f"[{timestamp}] {response.output_text}",
                    response_id=response.id,
                )
            except Exception as error:
                print(f"Memory save skipped: {error}")
        print_timing(show_timing, "memory save", perf_counter() - memory_save_start)
        if agent_id:
            invoke_memory_summary_function(
                supabase_client,
                agent_id=agent_id,
                user_id=dev_user.id,
                conversation_id=conversation_id,
            )
        print_timing(show_timing, "total turn", perf_counter() - turn_start)


if __name__ == "__main__":
    main()
