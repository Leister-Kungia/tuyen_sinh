"""
app.py — FastAPI wrapper cho tuyen_sinh_AI.py
"""

import os
from contextlib import asynccontextmanager
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────

class CauHoiRequest(BaseModel):
    session_id: str = "default"
    cau_hoi: str

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
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
    return FileResponse(html_path, media_type="text/html")

@app.post("/hoi", response_model=TraLoiResponse)
def hoi(body: CauHoiRequest):
    if not body.cau_hoi.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")
    try:
        bot = lay_bot(body.session_id)
        tra_loi = bot.hoi(body.cau_hoi)
        return TraLoiResponse(session_id=body.session_id, tra_loi=tra_loi)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
def reset(body: ResetRequest):
    if body.session_id in sessions:
        sessions[body.session_id].reset_lich_su()
    return {"status": "ok", "message": f"Đã reset session '{body.session_id}'"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
