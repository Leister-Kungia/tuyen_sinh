"""
app.py — FastAPI + Supabase auth + tuyen_sinh_AI
"""

import os
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from tuyen_sinh_AI import TuVanTuyenSinh

# Supabase Python client để verify JWT
try:
    from supabase import create_client, Client as SupabaseClient
    _SUPABASE_URL  = os.getenv("SUPABASE_URL", "")
    _SUPABASE_ANON = os.getenv("SUPABASE_ANON_KEY", "")
    sb: SupabaseClient = create_client(_SUPABASE_URL, _SUPABASE_ANON) if _SUPABASE_URL else None
except ImportError:
    sb = None

# ── Session store ─────────────────────────────────────────────────────────────
sessions: dict[str, TuVanTuyenSinh] = {}

def lay_bot(session_id: str) -> TuVanTuyenSinh:
    if session_id not in sessions:
        sessions[session_id] = TuVanTuyenSinh()
    return sessions[session_id]

@asynccontextmanager
async def lifespan(app: FastAPI):
    lay_bot("default")
    yield

app = FastAPI(title="Magerok AI", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth helper ───────────────────────────────────────────────────────────────
def get_user_id(authorization: Optional[str] = Header(default=None)) -> str:
    """
    Lấy user_id từ Supabase JWT trong header Authorization: Bearer <token>
    Nếu chưa cấu hình Supabase → fallback về 'anonymous' (dev mode)
    """
    if not sb or not _SUPABASE_URL:
        return "anonymous"  # dev mode — không cần auth
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Chưa đăng nhập.")
    token = authorization.split(" ", 1)[1]
    try:
        user = sb.auth.get_user(token)
        return user.user.id
    except Exception:
        raise HTTPException(status_code=401, detail="Token không hợp lệ hoặc đã hết hạn.")

# ── Schemas ───────────────────────────────────────────────────────────────────
class CauHoiRequest(BaseModel):
    cau_hoi: str = ""
    image_base64: Optional[str] = None
    image_type: Optional[str] = "image/jpeg"
    history_id: Optional[str] = None   # dùng để ghép lịch sử (frontend gửi lên, hiện tại bỏ qua — bot tự giữ state)

class TraLoiResponse(BaseModel):
    tra_loi: str
    anh: list[str] = []   # list base64 PNG — rỗng nếu không có hình vẽ
    history_id: Optional[str] = None   # trả về để frontend lưu vào sidebar
    user_id: str

class ResetRequest(BaseModel):
    pass  # user_id lấy từ token

# ── Serve static ──────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
_static = os.path.join(_BASE, "static")
if os.path.exists(_static):
    app.mount("/static", StaticFiles(directory=_static), name="static")

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    for p in [os.path.join(_BASE, "static", "login.html"),
              os.path.join(_BASE, "login.html")]:
        if os.path.exists(p):
            return FileResponse(p, media_type="text/html")
    return {"status": "ok", "message": "Magerok AI 🎓"}

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}

# ── Chat endpoints ────────────────────────────────────────────────────────────
@app.post("/hoi", response_model=TraLoiResponse)
def hoi(body: CauHoiRequest, user_id: str = Depends(get_user_id)):
    if not body.cau_hoi.strip() and not body.image_base64:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")
    try:
        bot = lay_bot(user_id)  # mỗi user có bot riêng
        if body.image_base64:
            ket_qua = bot.hoi_voi_anh(
                cau_hoi=body.cau_hoi or "(Xem ảnh đính kèm)",
                image_base64=body.image_base64,
                image_type=body.image_type or "image/jpeg",
            )
            if isinstance(ket_qua, str):
                ket_qua = {"tra_loi": ket_qua, "anh": []}
        else:
            ket_qua = bot.hoi(body.cau_hoi)
            if isinstance(ket_qua, str):
                ket_qua = {"tra_loi": ket_qua, "anh": []}
        return TraLoiResponse(
            tra_loi=ket_qua["tra_loi"],
            anh=ket_qua.get("anh", []),
            history_id=body.history_id or user_id,   # trả lại để frontend dùng làm conv id
            user_id=user_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── History endpoints (lịch sử hội thoại — lưu in-memory theo user) ──────────
# Nếu bạn muốn lưu vào Supabase thật, thay các hàm này bằng Supabase client

_histories: dict[str, list[dict]] = {}   # user_id → list of {id, title, updated_at}
_history_msgs: dict[str, list[dict]] = {}  # history_id → list of {role, content}

@app.get("/histories")
def get_histories(user_id: str = Depends(get_user_id)):
    """Trả danh sách lịch sử chat của user."""
    return _histories.get(user_id, [])

@app.get("/histories/{history_id}")
def get_history_messages(history_id: str, user_id: str = Depends(get_user_id)):
    """Trả danh sách tin nhắn của một cuộc trò chuyện."""
    msgs = _history_msgs.get(history_id, [])
    return msgs

@app.delete("/histories/{history_id}")
def delete_history(history_id: str, user_id: str = Depends(get_user_id)):
    """Xóa một cuộc trò chuyện."""
    _history_msgs.pop(history_id, None)
    if user_id in _histories:
        _histories[user_id] = [h for h in _histories[user_id] if h["id"] != history_id]
    # Xóa bot session tương ứng
    sessions.pop(history_id, None)
    return {"status": "ok"}

@app.post("/reset")
def reset(user_id: str = Depends(get_user_id)):
    if user_id in sessions:
        sessions[user_id].reset_lich_su()
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
