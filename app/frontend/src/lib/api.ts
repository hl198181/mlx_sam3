const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface UploadResponse {
  session_id: string;
  width: number;
  height: number;
  message: string;
  processing_time_ms: number;
}

export interface RLEMask {
  counts: number[];  // Run-length encoded counts
  size: [number, number];  // [height, width]
}

export interface SegmentationResult {
  original_width: number;
  original_height: number;
  masks?: RLEMask[];      // RLE-encoded masks
  boxes?: number[][];     // [N, 4] as [x0, y0, x1, y1]
  scores?: number[];      // [N]
  prompted_boxes?: { box: number[]; label: boolean }[];
}

export interface SegmentResponse {
  session_id: string;
  prompt?: string;
  box_type?: string;
  results: SegmentationResult;
  processing_time_ms: number;
}

export interface ResetResponse {
  session_id: string;
  message: string;
  results: SegmentationResult;
  processing_time_ms: number;
}

// Helper to fetch with error handling
async function apiFetch<T>(fetchFn: () => Promise<Response>): Promise<T> {
  const response = await fetchFn();
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Request failed");
  }
  
  return response.json();
}

async function compressImage(file: File, maxSize = 2048, quality = 0.85): Promise<File> {
  return new Promise((resolve) => {
    // Skip if already small enough
    if (file.size < 500 * 1024) {
      resolve(file);
      return;
    }

    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      let { width, height } = img;

      // Scale down if larger than maxSize
      if (width > maxSize || height > maxSize) {
        const scale = maxSize / Math.max(width, height);
        width = Math.round(width * scale);
        height = Math.round(height * scale);
      }

      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0, width, height);

      canvas.toBlob(
        (blob) => {
          if (blob) {
            resolve(new File([blob], file.name.replace(/\.\w+$/, ".jpg"), { type: "image/jpeg" }));
          } else {
            resolve(file);
          }
        },
        "image/jpeg",
        quality
      );
    };
    img.onerror = () => resolve(file);
    img.src = URL.createObjectURL(file);
  });
}

export async function uploadImage(file: File): Promise<UploadResponse> {
  const compressed = await compressImage(file);
  const formData = new FormData();
  formData.append("file", compressed);

  return apiFetch<UploadResponse>(() =>
    fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    })
  );
}

export async function segmentWithText(
  sessionId: string,
  prompt: string
): Promise<SegmentResponse> {
  return apiFetch<SegmentResponse>(() =>
    fetch(`${API_BASE}/segment/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, prompt }),
    })
  );
}

export async function addBoxPrompt(
  sessionId: string,
  box: number[],
  label: boolean
): Promise<SegmentResponse> {
  return apiFetch<SegmentResponse>(() =>
    fetch(`${API_BASE}/segment/box`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, box, label }),
    })
  );
}

export async function resetPrompts(sessionId: string): Promise<ResetResponse> {
  return apiFetch<ResetResponse>(() =>
    fetch(`${API_BASE}/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    })
  );
}

export async function checkHealth(): Promise<{ status: string; model_loaded: boolean }> {
  return apiFetch<{ status: string; model_loaded: boolean }>(() =>
    fetch(`${API_BASE}/health`)
  );
}

