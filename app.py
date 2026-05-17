"""
app.py — Magerok AI v5
Routes:
  /                → landing page
  /demo            → chat không cần đăng nhập
  /login           → đăng nhập / đăng ký
  /chat            → chat có auth + lịch sử Supabase
  /reset-password  → đặt lại mật khẩu (từ link email)
  /verify          → xác nhận email đăng ký
  /health          → health check

API:
  POST /demo/hoi         → chat không auth
  POST /hoi              → chat có auth, lưu lịch sử Supabase
  POST /reset            → reset session bot
  GET  /histories        → danh sách lịch sử (auth)
  GET  /histories/{id}   → messages của 1 cuộc trò chuyện (auth)
  DELETE /histories/{id} → xóa cuộc trò chuyện (auth)
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

# ── Supabase ──────────────────────────────────────────────────────────────────
try:
    from supabase import create_client, Client as SupabaseClient
    _SUPABASE_URL  = os.getenv("SUPABASE_URL", "")
    _SUPABASE_ANON = os.getenv("SUPABASE_ANON_KEY", "")
    sb: SupabaseClient = create_client(_SUPABASE_URL, _SUPABASE_ANON) if _SUPABASE_URL else None
except ImportError:
    sb = None

# ── Session store (in-memory) ─────────────────────────────────────────────────
sessions: dict[str, TuVanTuyenSinh] = {}

def lay_bot(session_id: str) -> TuVanTuyenSinh:
    if session_id not in sessions:
        sessions[session_id] = TuVanTuyenSinh()
    return sessions[session_id]

@asynccontextmanager
async def lifespan(app: FastAPI):
    lay_bot("demo")
    yield

app = FastAPI(title="Magerok AI", version="5.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth helpers ──────────────────────────────────────────────────────────────
def get_user_id(authorization: Optional[str] = Header(default=None)) -> str:
    if not sb or not _SUPABASE_URL:
        return "anonymous"
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Chưa đăng nhập.")
    token = authorization.split(" ", 1)[1]
    try:
        return sb.auth.get_user(token).user.id
    except Exception:
        raise HTTPException(status_code=401, detail="Token không hợp lệ hoặc đã hết hạn.")

def get_user_optional(authorization: Optional[str] = Header(default=None)) -> Optional[str]:
    if not sb or not authorization or not authorization.startswith("Bearer "):
        return None
    try:
        return sb.auth.get_user(authorization.split(" ", 1)[1]).user.id
    except Exception:
        return None

# ── Schemas ───────────────────────────────────────────────────────────────────
class CauHoiRequest(BaseModel):
    cau_hoi:      str = ""
    history_id:   Optional[str] = None
    image_base64: Optional[str] = None
    image_type:   Optional[str] = "image/jpeg"

class TraLoiResponse(BaseModel):
    tra_loi:    str
    anh:        list[str] = []
    user_id:    str
    history_id: Optional[str] = None

# ── Static files ──────────────────────────────────────────────────────────────
_BASE   = os.path.dirname(os.path.abspath(__file__))
_static = os.path.join(_BASE, "static")
if os.path.exists(_static):
    app.mount("/static", StaticFiles(directory=_static), name="static")

def _page(name: str) -> FileResponse:
    path = os.path.join(_static, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{name} không tìm thấy.")
    return FileResponse(path, media_type="text/html")

# ── Page routes ───────────────────────────────────────────────────────────────
@app.api_route("/",               methods=["GET", "HEAD"])
def root():       return _page("landing.html")

@app.api_route("/demo",           methods=["GET", "HEAD"])
def demo():       return _page("demo.html")

@app.api_route("/login",          methods=["GET", "HEAD"])
def login():      return _page("login.html")

@app.api_route("/chat",           methods=["GET", "HEAD"])
def chat():       return _page("chat.html")

@app.api_route("/reset-password", methods=["GET", "HEAD"])
def reset_pw():   return _page("reset-password.html")

@app.api_route("/verify",         methods=["GET", "HEAD"])
def verify():     return _page("verify.html")

@app.api_route("/health",         methods=["GET", "HEAD"])
def health():     return {"status": "ok"}

# ── Demo chat (không auth) ────────────────────────────────────────────────────
@app.post("/demo/hoi")
def demo_hoi(body: CauHoiRequest):
    if not body.cau_hoi.strip() and not body.image_base64:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")
    try:
        bot     = lay_bot("demo")
        ket_qua = _goi_bot(bot, body)
        return {"tra_loi": ket_qua["tra_loi"], "anh": ket_qua.get("anh", [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Authenticated chat ────────────────────────────────────────────────────────
@app.post("/hoi", response_model=TraLoiResponse)
def hoi(body: CauHoiRequest, user_id: str = Depends(get_user_id)):
    if not body.cau_hoi.strip() and not body.image_base64:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

    history_id = body.history_id
    if not history_id and sb:
        res = sb.table("chat_histories").insert({
            "user_id": user_id,
            "title": (body.cau_hoi[:60] or "Cuộc trò chuyện mới"),
        }).execute()
        history_id = res.data[0]["id"]

    try:
        bot     = lay_bot(user_id)
        ket_qua = _goi_bot(bot, body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if sb and history_id:
        sb.table("chat_messages").insert([
            {"history_id": history_id, "role": "user",     "content": body.cau_hoi or "(ảnh)"},
            {"history_id": history_id, "role": "assistant", "content": ket_qua["tra_loi"]},
        ]).execute()
        sb.table("chat_histories").update({"updated_at": "now()"}).eq("id", history_id).execute()

    return TraLoiResponse(
        tra_loi=ket_qua["tra_loi"],
        anh=ket_qua.get("anh", []),
        user_id=user_id,
        history_id=history_id,
    )

@app.post("/reset")
def reset(user_id: str = Depends(get_user_id)):
    if user_id in sessions:
        sessions[user_id].reset_lich_su()
    return {"status": "ok"}

# ── History API ───────────────────────────────────────────────────────────────
@app.get("/histories")
def get_histories(user_id: str = Depends(get_user_id)):
    if not sb:
        return []
    res = sb.table("chat_histories")\
        .select("id,title,created_at,updated_at")\
        .eq("user_id", user_id)\
        .order("updated_at", desc=True)\
        .limit(50).execute()
    return res.data

@app.get("/histories/{history_id}")
def get_messages(history_id: str, user_id: str = Depends(get_user_id)):
    if not sb:
        return []
    check = sb.table("chat_histories")\
        .select("id").eq("id", history_id).eq("user_id", user_id).execute()
    if not check.data:
        raise HTTPException(status_code=403, detail="Không có quyền truy cập.")
    res = sb.table("chat_messages")\
        .select("role,content,created_at")\
        .eq("history_id", history_id)\
        .order("created_at").execute()
    return res.data

@app.delete("/histories/{history_id}")
def delete_history(history_id: str, user_id: str = Depends(get_user_id)):
    if not sb:
        return {"status": "ok"}
    check = sb.table("chat_histories")\
        .select("id").eq("id", history_id).eq("user_id", user_id).execute()
    if not check.data:
        raise HTTPException(status_code=403, detail="Không có quyền.")
    sb.table("chat_messages").delete().eq("history_id", history_id).execute()
    sb.table("chat_histories").delete().eq("id", history_id).execute()
    if user_id in sessions:
        sessions[user_id].reset_lich_su()
    return {"status": "ok"}

# ── Helper nội bộ ─────────────────────────────────────────────────────────────
def _goi_bot(bot: TuVanTuyenSinh, body: CauHoiRequest) -> dict:
    if body.image_base64:
        ket_qua = bot.hoi_voi_anh(
            cau_hoi=body.cau_hoi or "(Xem ảnh đính kèm)",
            image_base64=body.image_base64,
            image_type=body.image_type or "image/jpeg",
        )
    else:
        ket_qua = bot.hoi(body.cau_hoi)
    if isinstance(ket_qua, str):
        return {"tra_loi": ket_qua, "anh": []}
    return {"tra_loi": ket_qua.get("tra_loi", ""), "anh": ket_qua.get("anh", [])}

# ── Dev server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
