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
import json as _json
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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
class FileAttachment(BaseModel):
    base64:     str
    media_type: str   # image/jpeg, image/png, application/pdf, text/plain, ...
    name:       str   # tên file gốc

class CauHoiRequest(BaseModel):
    cau_hoi:      str = ""
    history_id:   Optional[str] = None
    # Đa file — ưu tiên dùng files[] thay cho image_base64 cũ
    files:        list[FileAttachment] = []
    # Backward compat với client cũ (1 file)
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

# ── Authenticated chat (SSE streaming progress) ───────────────────────────────
@app.post("/hoi")
def hoi(body: CauHoiRequest, user_id: str = Depends(get_user_id)):
    # Validation: cần ít nhất text hoặc file
    all_files_check = list(body.files) or ([body.image_base64] if body.image_base64 else [])
    if not body.cau_hoi.strip() and not all_files_check:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

    def event_stream():
        def emit(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {_json.dumps(data, ensure_ascii=False)}\n\n"

        def _file_label(f) -> str:
            m = f.media_type if hasattr(f, "media_type") else (f or "")
            if "pdf" in m: return "PDF"
            if m.startswith("image/"): return "ảnh"
            if "word" in m or "document" in m: return "file Word"
            if "csv" in m: return "file CSV"
            if "text" in m: return "file văn bản"
            return "file"

        try:
            # ── Chuẩn hoá danh sách file ──────────────────────────────────
            all_files = list(body.files)
            if not all_files and body.image_base64:
                all_files = [FileAttachment(
                    base64=body.image_base64,
                    media_type=body.image_type or "image/jpeg",
                    name="file đính kèm",
                )]

            imgs  = [f for f in all_files if f.media_type.startswith("image/")]
            pdfs  = [f for f in all_files if "pdf" in f.media_type]
            docs  = [f for f in all_files if "word" in f.media_type or "document" in f.media_type]
            txts  = [f for f in all_files if "text" in f.media_type or "csv" in f.media_type]
            has_file = bool(all_files)

            # Bước 1: Khởi tạo history
            yield emit("progress", {"step": 1, "text": "🔐 Xác thực và khởi tạo cuộc trò chuyện…"})
            history_id = body.history_id
            if not history_id and sb:
                title = body.cau_hoi[:60] if body.cau_hoi.strip() else (
                    "📎 " + ", ".join(f.name for f in all_files[:2]) if all_files else "Cuộc trò chuyện mới"
                )
                res = sb.table("chat_histories").insert({
                    "user_id": user_id, "title": title,
                }).execute()
                history_id = res.data[0]["id"]

            bot = lay_bot(user_id)

            # ── Nhánh có file đính kèm ────────────────────────────────────
            if has_file:
                # Tóm tắt các loại file
                parts = []
                if imgs:  parts.append(f"{len(imgs)} ảnh")
                if pdfs:  parts.append(f"{len(pdfs)} PDF")
                if docs:  parts.append(f"{len(docs)} file Word")
                if txts:  parts.append(f"{len(txts)} file văn bản")
                summary = " + ".join(parts) or f"{len(all_files)} file"

                yield emit("progress", {"step": 2, "text": f"📂 Đang đọc và giải mã {summary}…"})

                # Từng file nếu > 1
                if len(all_files) > 1:
                    for i, f in enumerate(all_files, 1):
                        yield emit("progress", {
                            "step": 3,
                            "text": f"🔍 Đang phân tích {_file_label(f)} ({i}/{len(all_files)}): {f.name}…"
                        })
                else:
                    yield emit("progress", {
                        "step": 3,
                        "text": f"🔍 AI đang phân tích nội dung {_file_label(all_files[0])}…"
                    })

                yield emit("progress", {"step": 4, "text": "✍️ AI đang soạn nhận xét và tư vấn…"})

                # Ghép context nếu nhiều file
                extra_ctx = ""
                if len(all_files) > 1:
                    extra_ctx = "\n\n[Danh sách file đính kèm: " + "; ".join(
                        f"{f.name} ({_file_label(f)})" for f in all_files
                    ) + "]"

                main_file = (imgs + pdfs + docs + txts)[0]
                ket_qua = bot.hoi_voi_anh(
                    cau_hoi=(body.cau_hoi or f"Hãy phân tích {summary} đính kèm") + extra_ctx,
                    image_base64=main_file.base64,
                    image_type=main_file.media_type,
                )
                if isinstance(ket_qua, str):
                    ket_qua = {"tra_loi": ket_qua, "anh": []}

                tra_loi_raw = ket_qua.get("tra_loi", "")
                anh_list    = ket_qua.get("anh", [])
                user_log    = body.cau_hoi or f"(Gửi {summary})"

                bot.lich_su += [
                    {"role": "user",      "content": user_log},
                    {"role": "assistant", "content": tra_loi_raw},
                ]
                bot.lich_su = bot.lich_su[-20:]

            # ── Nhánh text thuần ─────────────────────────────────────────
            else:
                yield emit("progress", {"step": 2, "text": "🧠 Đang phân tích câu hỏi và chọn chuyên gia…"})
                agents, can_hoi_them = bot._phan_loai(body.cau_hoi)

                agent_names = {
                    "diem_chuan": "điểm chuẩn", "truong": "thông tin trường",
                    "nganh": "ngành học", "to_hop": "tổ hợp môn",
                    "huong_nghiep": "hướng nghiệp", "hoc_tap": "lộ trình học tập",
                    "kien_thuc": "kiến thức"
                }
                agent_label = ", ".join(agent_names.get(a, a) for a in agents)
                yield emit("progress", {"step": 3, "text": f"🔍 Đang tìm kiếm dữ liệu về: {agent_label}…"})
                yield emit("progress", {"step": 4, "text": "✍️ AI đang soạn câu trả lời…"})

                if can_hoi_them and len(bot.lich_su) < 4:
                    tra_loi_raw = can_hoi_them
                    bot.lich_su += [
                        {"role": "user",      "content": body.cau_hoi},
                        {"role": "assistant", "content": can_hoi_them},
                    ]
                    bot.lich_su = bot.lich_su[-20:]
                    anh_list = []
                else:
                    ket_qua = {}
                    for i, agent in enumerate(agents):
                        if len(agents) > 1:
                            yield emit("progress", {"step": 4, "text": f"✍️ Chuyên gia {agent_names.get(agent, agent)} đang trả lời… ({i+1}/{len(agents)})"})
                        ket_qua[agent] = bot._chay_agent(agent, body.cau_hoi)

                    if len(ket_qua) > 1:
                        yield emit("progress", {"step": 4, "text": "🔗 Đang tổng hợp từ nhiều chuyên gia…"})
                        tra_loi_raw = bot._tong_hop(body.cau_hoi, ket_qua)
                    else:
                        tra_loi_raw = list(ket_qua.values())[0]

                    from tuyen_sinh_AI import xu_ly_anh_trong_tra_loi
                    tra_loi_raw, anh_list = xu_ly_anh_trong_tra_loi(tra_loi_raw)

                    bot.lich_su += [
                        {"role": "user",      "content": body.cau_hoi},
                        {"role": "assistant", "content": tra_loi_raw},
                    ]
                    bot.lich_su = bot.lich_su[-20:]

            # Bước 5: Lưu Supabase
            yield emit("progress", {"step": 5, "text": "💾 Đang lưu lịch sử trò chuyện…"})
            if sb and history_id:
                sb.table("chat_messages").insert([
                    {"history_id": history_id, "role": "user",      "content": body.cau_hoi or "(ảnh)"},
                    {"history_id": history_id, "role": "assistant",  "content": tra_loi_raw},
                ]).execute()
                from datetime import datetime, timezone
                sb.table("chat_histories").update({
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }).eq("id", history_id).execute()

            # Xong — gửi kết quả
            yield emit("done", {
                "tra_loi":    tra_loi_raw,
                "anh":        anh_list,
                "history_id": history_id,
                "user_id":    user_id,
            })

        except Exception as e:
            import traceback, logging
            logging.error(f"[/hoi SSE] {e}\n{traceback.format_exc()}")
            yield emit("error", {"detail": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

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
