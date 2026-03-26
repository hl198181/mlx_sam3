"""
FastAPI backend for SAM3 GPU segmentation model.
Provides endpoints for image upload, text prompts, box prompts, and segmentation results.
"""

import asyncio
import io
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# Add parent directory to path to import sam3_gpu
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sam3_gpu
from sam3_gpu import build_sam3_image_model
from sam3_gpu.model.sam3_image_processor import Sam3Processor

# Global model and processor
model = None
processor = None

# Device detection
device = "cuda" if torch.cuda.is_available() else "cpu"

# Session storage for processing states
sessions: dict = {}

# Concurrency and resource limits
MAX_SESSIONS = 5
SESSION_TTL = 1800  # 30 minutes
inference_lock = threading.Lock()


def _get_peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0


def _remove_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _evict_oldest_session():
    if not sessions:
        return
    oldest_id = min(sessions, key=lambda sid: sessions[sid].get("last_active", 0))
    _remove_session(oldest_id)


async def _cleanup_expired_sessions():
    while True:
        await asyncio.sleep(60)
        now = time.time()
        expired = [
            sid for sid, s in sessions.items()
            if now - s.get("last_active", 0) > SESSION_TTL
        ]
        for sid in expired:
            print(f"Session {sid} expired, removing.")
            _remove_session(sid)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor

    print(f"Using device: {device}")
    model = build_sam3_image_model()
    if device == "cuda":
        model.to("cuda")
    processor = Sam3Processor(model)
    print("SAM3 GPU model loaded successfully!")

    cleanup_task = asyncio.create_task(_cleanup_expired_sessions())

    yield

    cleanup_task.cancel()
    sessions.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="SAM3 GPU Segmentation API",
    description="API for interactive image segmentation using SAM3 model (GPU)",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextPromptRequest(BaseModel):
    session_id: str
    prompt: str


class BoxPromptRequest(BaseModel):
    session_id: str
    box: list[float]
    label: bool


class ConfidenceRequest(BaseModel):
    session_id: str
    threshold: float


class SessionRequest(BaseModel):
    session_id: str


def mask_to_rle(mask: np.ndarray) -> dict:
    flat = mask.flatten()
    diff = np.diff(flat)
    change_indices = np.where(diff != 0)[0] + 1
    run_starts = np.concatenate([[0], change_indices])
    run_ends = np.concatenate([change_indices, [len(flat)]])
    run_lengths = (run_ends - run_starts).tolist()
    if flat[0] == 1:
        run_lengths = [0] + run_lengths
    return {"counts": run_lengths, "size": list(mask.shape)}


def serialize_state(state: dict) -> dict:
    result = {
        "original_width": state.get("original_width"),
        "original_height": state.get("original_height"),
    }

    if "masks" in state:
        masks = state["masks"]
        boxes = state["boxes"]
        scores = state["scores"]

        masks_list = []
        boxes_list = []
        scores_list = []

        for i in range(len(scores)):
            mask_t = masks[i]
            mask_np = mask_t.cpu().numpy() if isinstance(mask_t, torch.Tensor) else np.array(mask_t)
            box_t = boxes[i]
            box_np = box_t.cpu().numpy() if isinstance(box_t, torch.Tensor) else np.array(box_t)
            score_val = scores[i]
            if isinstance(score_val, torch.Tensor):
                score_val = score_val.cpu().item()
            else:
                score_val = float(score_val)

            mask_binary = (mask_np > 0.5).astype(np.uint8)
            if mask_binary.ndim == 3:
                mask_binary = mask_binary[0]

            rle = mask_to_rle(mask_binary)
            masks_list.append(rle)
            boxes_list.append(box_np.tolist())
            scores_list.append(score_val)

        result["masks"] = masks_list
        result["boxes"] = boxes_list
        result["scores"] = scores_list

    if "prompted_boxes" in state:
        result["prompted_boxes"] = state["prompted_boxes"]

    return result


# ---- Synchronous endpoints (FastAPI runs them in threadpool automatically) ----

@app.get("/")
async def root():
    return {"message": "SAM3 GPU Segmentation API", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None, "device": device}


@app.post("/upload")
def upload_image(file: UploadFile = File(...)):
    """Upload an image and initialize a session."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

    while len(sessions) >= MAX_SESSIONS:
        _evict_oldest_session()

    with inference_lock:
        try:
            session_id = str(uuid.uuid4())

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            start_time = time.perf_counter()
            with torch.no_grad():
                state = processor.set_image(image)
            processing_time_ms = (time.perf_counter() - start_time) * 1000

            sessions[session_id] = {
                "state": state,
                "image_size": image.size,
                "last_active": time.time(),
            }

            return {
                "session_id": session_id,
                "width": image.size[0],
                "height": image.size[1],
                "message": "Image uploaded and processed successfully",
                "processing_time_ms": round(processing_time_ms, 2),
                "peak_memory_mb": round(_get_peak_memory_mb(), 2)
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/segment/text")
def segment_with_text(request: TextPromptRequest):
    """Segment image using text prompt."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    with inference_lock:
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            start_time = time.perf_counter()
            with torch.no_grad():
                state = processor.set_text_prompt(request.prompt, session["state"])
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            session["state"] = state
            session["last_active"] = time.time()
            results = serialize_state(state)

            return {
                "session_id": request.session_id,
                "prompt": request.prompt,
                "results": results,
                "processing_time_ms": round(processing_time_ms, 2),
                "peak_memory_mb": round(_get_peak_memory_mb(), 2)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during segmentation: {str(e)}")


@app.post("/segment/box")
def add_box_prompt(request: BoxPromptRequest):
    """Add a box prompt (positive or negative) and re-segment."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    with inference_lock:
        try:
            state = session["state"]

            if "prompted_boxes" not in state:
                state["prompted_boxes"] = []

            img_w = state["original_width"]
            img_h = state["original_height"]
            cx, cy, w, h = request.box
            x_min = (cx - w / 2) * img_w
            y_min = (cy - h / 2) * img_h
            x_max = (cx + w / 2) * img_w
            y_max = (cy + h / 2) * img_h

            state["prompted_boxes"].append({
                "box": [x_min, y_min, x_max, y_max],
                "label": request.label
            })

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            start_time = time.perf_counter()
            with torch.no_grad():
                state = processor.add_geometric_prompt(request.box, request.label, state)
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            session["state"] = state
            session["last_active"] = time.time()

            return {
                "session_id": request.session_id,
                "box_type": "positive" if request.label else "negative",
                "results": serialize_state(state),
                "processing_time_ms": round(processing_time_ms, 2),
                "peak_memory_mb": round(_get_peak_memory_mb(), 2)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error adding box prompt: {str(e)}")


@app.post("/reset")
def reset_prompts(request: SessionRequest):
    """Reset all prompts for a session."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        state = session["state"]
        session["last_active"] = time.time()

        start_time = time.perf_counter()
        processor.reset_all_prompts(state)
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        if "prompted_boxes" in state:
            del state["prompted_boxes"]

        return {
            "session_id": request.session_id,
            "message": "All prompts reset",
            "results": serialize_state(state),
            "processing_time_ms": round(processing_time_ms, 2),
            "peak_memory_mb": round(_get_peak_memory_mb(), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting prompts: {str(e)}")


@app.post("/confidence")
def set_confidence(request: ConfidenceRequest):
    """Update confidence threshold."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    processor.confidence_threshold = request.threshold

    return {
        "session_id": request.session_id,
        "threshold": request.threshold,
        "message": "Confidence threshold updated. Re-run segmentation to apply."
    }


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Delete a session and free memory."""
    if session_id in sessions:
        _remove_session(session_id)
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
