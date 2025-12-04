import os
import sys
import contextlib
import asyncio
import threading
import time
from typing import Optional
import torch
import whisperx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel


@contextlib.contextmanager
def suppress_stdout_stderr():
    class _DevNull:
        def write(self, s):
            pass
        def flush(self):
            pass
    devnull = _DevNull()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        try:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except Exception:
            pass


def _get_whisper_device():
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return f"cuda:{local_rank}"
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        return "cuda:0"
    try:
        current_device = torch.cuda.current_device()
        return f"cuda:{current_device}"
    except Exception:
        return "cuda:0"


whisper_model = None
batch_size = 1
transcribe_lock = threading.Lock()
active_transcribe_count = 0

def _maybe_load_model():
    global whisper_model
    if whisper_model is not None:
        return
    device = _get_whisper_device()
    compute_type = "int8"
    import logging
    logging.getLogger("whisperx").setLevel(logging.ERROR)
    if device.startswith("cuda:"):
        device_id = int(device.split(":")[1])
        whisper_model = whisperx.load_model("large-v3", device="cuda", device_index=device_id, compute_type=compute_type, language="en")
    else:
        whisper_model = whisperx.load_model("large-v3", device=device, compute_type=compute_type, language="en")


app = FastAPI(title="Molmo-R1 Transcribe API")


@app.get("/health")
async def health():
    """Health endpoint that responds quickly without blocking"""
    global active_transcribe_count
    
    # Quick health check - don't wait for locks or heavy operations
    if whisper_model is None:
        return JSONResponse(status_code=503, content={"status": "starting"})
    
    # Return status with current load information
    return {
        "status": "ok",
        "active_transcriptions": active_transcribe_count,
        "model_loaded": whisper_model is not None
    }


class TranscribeRequest(BaseModel):
    filepath: str
    timestamp_start: Optional[str] = None
    timestamp_end: Optional[str] = None

_maybe_load_model()

@app.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    """Transcribe endpoint with proper concurrency control"""
    global active_transcribe_count
    
    filepath = req.filepath
    if not isinstance(filepath, str) or not filepath:
        raise HTTPException(status_code=400, detail="Invalid filepath")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"File not found: {filepath}")
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=400, detail=f"Not a file: {filepath}")

    # Use lock to prevent concurrent transcribe operations that could block health endpoint
    with transcribe_lock:
        active_transcribe_count += 1
        try:
            # WhisperX transcribe
            with suppress_stdout_stderr():
                result = whisper_model.transcribe(filepath, batch_size=batch_size)
            return result
        finally:
            active_transcribe_count -= 1


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = sys.argv[1]
    uvicorn.run("sage.serve.transcribe_api:app", host=host, port=port, reload=False)


