import asyncio
import os
import json
import re
import subprocess
import tempfile

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

LLM_PROVIDER    = os.getenv("LLM_PROVIDER",    "ollama")
OLLAMA_URL      = os.getenv("OLLAMA_URL",       "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",     "llama3.2:3b")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY",   "")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL",     "gpt-4o-mini")
WHISPER_MODEL   = os.getenv("WHISPER_MODEL",    "base")
LANGUAGE        = os.getenv("LANGUAGE",         "es")
SILENCE_TIMEOUT = float(os.getenv("SILENCE_TIMEOUT", "2.5"))
PROMPT_FILE     = os.getenv("PROMPT_FILE", os.path.join(os.path.dirname(__file__), "prompt.txt"))

def _load_prompt() -> str:
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"[prompt] could not load {PROMPT_FILE}: {e}")
        return "Eres un asistente experto. Responde en máximo 2 oraciones, directo al punto."

SYSTEM_PROMPT = _load_prompt()
print(f"[prompt] loaded from {PROMPT_FILE} ({len(SYSTEM_PROMPT)} chars)")

_whisper_model = None


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel  # type: ignore
        print(f"[whisper] loading model={WHISPER_MODEL} …")
        _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        print("[whisper] ready")
    return _whisper_model


def _audio_extension(data: bytes) -> str:
    if len(data) > 8 and data[4:8] == b"ftyp":
        return ".mp4"
    if len(data) > 4 and data[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"
    return ".webm"


def _save_chunk(audio_bytes: bytes) -> str:
    ext = _audio_extension(audio_bytes)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(audio_bytes)
        return f.name


def _delete(path: str) -> None:
    try:
        os.unlink(path)
    except Exception:
        pass


_SILENCE_DB = float(os.getenv("SILENCE_DB", "-35"))


def is_voice_chunk(path: str) -> bool:
    if os.path.getsize(path) < 3_000:
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", path, "-af", "volumedetect",
             "-vn", "-sn", "-dn", "-f", "null", "/dev/null"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stderr.split("\n"):
            if "mean_volume" in line:
                m = re.search(r"mean_volume: ([\-\d.]+) dB", line)
                if m:
                    db = float(m.group(1))
                    print(f"[vad] mean_volume={db:.1f} dB  threshold={_SILENCE_DB} dB")
                    return db > _SILENCE_DB
        return True
    except Exception as e:
        print(f"[vad] error: {e}")
        return True


def transcribe_file(path: str) -> str:
    model = get_whisper()
    lang = None if LANGUAGE == "auto" else LANGUAGE
    try:
        segments, info = model.transcribe(
            path,
            language=lang,
            beam_size=2,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.45,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 400,
            },
            no_speech_threshold=0.5,
            condition_on_previous_text=False,
        )
        text = " ".join(
            s.text.strip() for s in segments if s.avg_logprob > -1.0
        ).strip()
        print(f"[whisper] lang={info.language}({info.language_probability:.2f}) text={text!r}")
        return text
    except Exception as e:
        print(f"[whisper] error: {e}")
        return ""


def concat_and_transcribe(chunk_paths: list[str]) -> str:
    if not chunk_paths:
        return ""

    if len(chunk_paths) == 1:
        return transcribe_file(chunk_paths[0])

    ext      = os.path.splitext(chunk_paths[0])[1] or ".webm"
    list_txt = tempfile.mktemp(suffix=".txt")
    out_file = tempfile.mktemp(suffix=ext)

    try:
        with open(list_txt, "w") as f:
            for p in chunk_paths:
                f.write(f"file '{p}'\n")

        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", list_txt, "-c", "copy", out_file],
            capture_output=True, timeout=30, check=True,
        )
        return transcribe_file(out_file)

    except Exception as e:
        print(f"[concat] error: {e} — falling back to last chunk")
        return transcribe_file(chunk_paths[-1])

    finally:
        _delete(list_txt)
        _delete(out_file)


_SENTENCE_END = re.compile(r"[.!?]\s*$")


def extract_sentences(text: str) -> tuple[list[str], str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    complete: list[str] = []
    for part in parts[:-1]:
        part = part.strip()
        if part:
            complete.append(part)
    remainder = parts[-1].strip() if parts else ""
    if _SENTENCE_END.search(remainder) and remainder:
        complete.append(remainder)
        remainder = ""
    return complete, remainder


MAX_HISTORY = int(os.getenv("MAX_HISTORY", "6"))


def _build_messages(sentence: str, history: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": sentence})
    return messages


async def ask_llm(sentence: str, history: list[dict]) -> str:
    messages = _build_messages(sentence, history)

    if LLM_PROVIDER == "ollama":
        import httpx  # type: ignore
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model":    OLLAMA_MODEL,
                    "messages": messages,
                    "stream":   False,
                    "options":  {"temperature": 0.3, "num_predict": 120},
                },
            )
            r.raise_for_status()
            return r.json()["message"]["content"].strip()

    if LLM_PROVIDER == "openai":
        import openai  # type: ignore
        client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        r = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=120,
            temperature=0.3,
        )
        return r.choices[0].message.content.strip()

    return "(LLM_PROVIDER not configured)"


app = FastAPI(title="voice-conscience")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    print("[startup] loading Whisper …")
    await loop.run_in_executor(None, get_whisper)
    print("[startup] warming up Ollama …")
    asyncio.create_task(_warmup_ollama())


async def _warmup_ollama():
    try:
        await ask_llm("hola", [])
        print("[startup] Ollama ready")
    except Exception as e:
        print(f"[startup] Ollama warmup failed (will retry on first request): {e}")


@app.get("/")
async def root():
    return {"status": "ok", "provider": LLM_PROVIDER, "whisper": WHISPER_MODEL}


@app.get("/config")
async def get_config():
    return {
        "llm_provider":    LLM_PROVIDER,
        "llm_model":       OLLAMA_MODEL if LLM_PROVIDER == "ollama" else OPENAI_MODEL,
        "ollama_url":      OLLAMA_URL,
        "whisper_model":   WHISPER_MODEL,
        "language":        LANGUAGE,
        "silence_timeout": SILENCE_TIMEOUT,
    }


@app.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    await websocket.accept()

    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=20)
    loop  = asyncio.get_event_loop()

    chunk_files: list[str] = []
    history:     list[dict] = []

    async def send(obj: dict) -> None:
        try:
            await websocket.send_text(json.dumps(obj, ensure_ascii=False))
        except Exception:
            pass

    async def process_utterance() -> None:
        if not chunk_files:
            return

        files = chunk_files.copy()
        chunk_files.clear()

        print(f"[ws] processing utterance from {len(files)} file(s)")
        text = await loop.run_in_executor(None, concat_and_transcribe, files)
        for p in files:
            _delete(p)

        if not text:
            print("[ws] transcription empty, skipping")
            return

        print(f"[ws] transcription: {text!r}")
        await send({"type": "transcription", "text": text})

        if len(text.split()) >= 4 and not queue.full():
            await send({"type": "queued", "text": text, "position": queue.qsize() + 1})
            await queue.put(text)
            await send({"type": "status", "queue_size": queue.qsize()})

    async def llm_worker() -> None:
        while True:
            sentence = await queue.get()
            try:
                print(f"[llm] querying: {sentence!r}")
                await send({"type": "processing", "text": sentence})
                response = await ask_llm(sentence, history)
                print(f"[llm] response: {response!r}")
                await send({"type": "response", "input": sentence, "output": response})
                history.append({"role": "user",      "content": sentence})
                history.append({"role": "assistant",  "content": response})
                if len(history) > MAX_HISTORY * 2:
                    history[:] = history[-(MAX_HISTORY * 2):]
            except Exception as e:
                await send({"type": "error", "msg": str(e)})
            finally:
                queue.task_done()
                await send({"type": "status", "queue_size": queue.qsize()})

    worker = asyncio.create_task(llm_worker())

    try:
        while True:
            msg = await websocket.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            if msg.get("text") == "flush":
                asyncio.ensure_future(process_utterance())
                continue

            audio_bytes = msg.get("bytes")
            if not audio_bytes:
                continue

            path = await loop.run_in_executor(None, _save_chunk, audio_bytes)
            chunk_files.append(path)
            print(f"[ws] received audio chunk {len(audio_bytes)} bytes → {path}")

    except WebSocketDisconnect:
        pass
    finally:
        worker.cancel()
        for p in chunk_files:
            _delete(p)
