"""
app.py — FastAPI wrapper cho tuyen_sinh_AI.py
Render chạy file này để khởi động web server.

Endpoints:
  GET  /          → serve giao diện chat (index.html)
  GET  /health    → kiểm tra server còn sống
  POST /hoi       → gửi câu hỏi, nhận câu trả lời
  POST /reset     → xóa lịch sử hội thoại của session
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from tuyen_sinh_AI import TuVanTuyenSinh

# ── Session store ─────────────────────────────────────────────────────────────
sessions: dict[str, TuVanTuyenSinh] = {}

def lay_bot(session_id: str) -> TuVanTuyenSinh:
    if session_id not in sessions:
        sessions[session_id] = TuVanTuyenSinh()
    return sessions[session_id]


# ── Khởi động app ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    lay_bot("default")
    yield

app = FastAPI(
    title="AI Tư Vấn Tuyển Sinh",
    description="Hỏi về điểm chuẩn, ngành học, định hướng nghề nghiệp",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bắt lỗi 413 từ Render proxy — trả về JSON thân thiện
@app.exception_handler(413)
async def request_too_large(request: Request, exc):
    return JSONResponse(
        status_code=413,
        content={"detail": "File quá lớn. Vui lòng gửi ảnh nhỏ hơn 5MB."},
    )


# ── Schemas ───────────────────────────────────────────────────────────────────

class CauHoiRequest(BaseModel):
    session_id: str = "default"
    cau_hoi: str = ""
    image_base64: Optional[str] = None
    image_type: Optional[str] = "image/jpeg"

class TraLoiResponse(BaseModel):
    session_id: str
    tra_loi: str

class ResetRequest(BaseModel):
    session_id: str = "default"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok", "message": "AI Tư Vấn Tuyển Sinh đang chạy 🎓"}

@app.api_route("/", methods=["GET", "HEAD"])
def serve_frontend():
    """Serve giao diện chat."""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
    return FileResponse(html_path, media_type="text/html")


@app.post("/hoi", response_model=TraLoiResponse)
def hoi(body: CauHoiRequest):
    if not body.cau_hoi.strip() and not body.image_base64:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")
    try:
        bot = lay_bot(body.session_id)
        if body.image_base64:
            tra_loi = bot.hoi_voi_anh(
                body.cau_hoi or "(Xem ảnh đính kèm)",
                body.image_base64,
                body.image_type or "image/jpeg",
            )
        else:
            tra_loi = bot.hoi(body.cau_hoi)
        return TraLoiResponse(session_id=body.session_id, tra_loi=tra_loi)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
def reset(body: ResetRequest):
    if body.session_id in sessions:
        sessions[body.session_id].reset_lich_su()
    return {"status": "ok", "message": f"Đã reset session '{body.session_id}'"}


# ── Chạy trực tiếp (dev) ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        h11_max_incomplete_event_size=5 * 1024 * 1024,
    )
