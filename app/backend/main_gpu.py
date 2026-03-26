"""
FastAPI backend for SAM3 GPU segmentation model.
Provides endpoints for image upload, text prompts, box prompts, and segmentation results.
"""

import asyncio
import io
import os
import sys
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
INFERENCE_TIMEOUT = 30  # seconds to wait in queue before returning 503
inference_semaphore = asyncio.Semaphore(1)


def _get_peak_memory_mb() -> float:
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0


def _remove_session(session_id: str):
    """Remove a session and release GPU memory."""
    if session_id in sessions:
        del sessions[session_id]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _evict_oldest_session():
    """Remove the least recently active session."""
    if not sessions:
        return
    oldest_id = min(sessions, key=lambda sid: sessions[sid].get("last_active", 0))
    _remove_session(oldest_id)


async def _cleanup_expired_sessions():
    """Background task to remove expired sessions."""
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
    """Load model on startup."""
    global model, processor

    print(f"Using device: {device}")
    model = build_sam3_image_model()
    if device == "cuda":
        model.to("cuda")
    processor = Sam3Processor(model)
    print("SAM3 GPU model loaded successfully!")

    cleanup_task = asyncio.create_task(_cleanup_expired_sessions())

    yield

    # Cleanup
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

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextPromptRequest(BaseModel):
    session_id: str
    prompt: str


class BoxPromptRequest(BaseModel):
    session_id: str
    box: list[float]  # [center_x, center_y, width, height] normalized
    label: bool  # True for positive, False for negative


class ConfidenceRequest(BaseModel):
    session_id: str
    threshold: float


class SessionRequest(BaseModel):
    session_id: str


def mask_to_rle(mask: np.ndarray) -> dict:
    """
    Encode a binary mask to RLE (Run-Length Encoding) format.

    Args:
        mask: 2D binary numpy array (H, W) with values 0 or 1

    Returns:
        dict with 'counts' (list of run lengths) and 'size' [H, W]
    """
    # Flatten the mask in row-major (C) order
    flat = mask.flatten()

    # Find where values change
    diff = np.diff(flat)
    change_indices = np.where(diff != 0)[0] + 1

    # Build run lengths
    run_starts = np.concatenate([[0], change_indices])
    run_ends = np.concatenate([change_indices, [len(flat)]])
    run_lengths = (run_ends - run_starts).tolist()

    # If mask starts with 1, prepend a 0-length run for background
    if flat[0] == 1:
        run_lengths = [0] + run_lengths

    return {
        "counts": run_lengths,
        "size": list(mask.shape)  # [H, W]
    }


def serialize_state(state: dict) -> dict:
    """Convert state arrays to JSON-serializable format."""
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
            mask_np = np.array(masks[i].cpu() if isinstance(masks[i], torch.Tensor) else masks[i])
            box_np = np.array(boxes[i].cpu() if isinstance(boxes[i], torch.Tensor) else boxes[i])
            score_val = scores[i]
            if isinstance(score_val, torch.Tensor):
                score_val = score_val.cpu().item()
            else:
                score_val = float(np.array(score_val))

            # Convert mask to binary and get the 2D mask (handle [1, H, W] shape)
            mask_binary = (mask_np > 0.5).astype(np.uint8)
            if mask_binary.ndim == 3:
                mask_binary = mask_binary[0]  # Take first channel

            # Encode as RLE
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


@app.get("/")
async def root():
    return {"message": "SAM3 GPU Segmentation API", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None, "device": device}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image and initialize a session."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Read and validate image before acquiring the lock
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

    # Evict oldest session if at capacity
    while len(sessions) >= MAX_SESSIONS:
        _evict_oldest_session()

    try:
        await asyncio.wait_for(inference_semaphore.acquire(), timeout=INFERENCE_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Server busy, try again later")
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
    finally:
        inference_semaphore.release()


@app.post("/segment/text")
async def segment_with_text(request: TextPromptRequest):
    """Segment image using text prompt."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        await asyncio.wait_for(inference_semaphore.acquire(), timeout=INFERENCE_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Server busy, try again later")
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.perf_counter()
        with torch.no_grad():
            state = processor.set_text_prompt(request.prompt, session["state"])
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        session["state"] = state
        session["last_active"] = time.time()
        start = time.perf_counter()
        results = serialize_state(state)
        end = time.perf_counter()
        print(f"Serialization took {end - start:.4f} seconds")

        return {
            "session_id": request.session_id,
            "prompt": request.prompt,
            "results": results,
            "processing_time_ms": round(processing_time_ms, 2),
            "peak_memory_mb": round(_get_peak_memory_mb(), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during segmentation: {str(e)}")
    finally:
        inference_semaphore.release()


@app.post("/segment/box")
async def add_box_prompt(request: BoxPromptRequest):
    """Add a box prompt (positive or negative) and re-segment."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        await asyncio.wait_for(inference_semaphore.acquire(), timeout=INFERENCE_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Server busy, try again later")
    try:
        state = session["state"]

        # Store prompted box for display
        if "prompted_boxes" not in state:
            state["prompted_boxes"] = []

        # Convert from normalized cxcywh to pixel xyxy for display
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
    finally:
        inference_semaphore.release()


@app.post("/reset")
async def reset_prompts(request: SessionRequest):
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
async def set_confidence(request: ConfidenceRequest):
    """Update confidence threshold (note: requires re-running inference)."""
    if processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Update processor threshold
    processor.confidence_threshold = request.threshold

    return {
        "session_id": request.session_id,
        "threshold": request.threshold,
        "message": "Confidence threshold updated. Re-run segmentation to apply."
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and free memory."""
    if session_id in sessions:
        _remove_session(session_id)
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
