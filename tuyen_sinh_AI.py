"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              AI TƯ VẤN TUYỂN SINH ĐẠI HỌC — FILE TỔNG HỢP                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Cấu trúc file này (đọc theo thứ tự):                                       ║
║    PHẦN 0 — CÀI ĐẶT & HƯỚNG DẪN NHANH                                      ║
║    PHẦN 1 — CẤU HÌNH (chỉnh ở đây nếu muốn thay đổi gì)                   ║
║    PHẦN 2 — PROMPTS (kịch bản cho từng AI agent)                            ║
║    PHẦN 3 — INGEST (nạp dữ liệu Excel/PDF/Web → ChromaDB)                  ║
║    PHẦN 4 — QUERY (nhận câu hỏi → tìm dữ liệu → gọi AI → trả lời)         ║
║    PHẦN 5 — MAIN (chạy thử trên terminal hoặc khởi động server web)         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  AGENTS CÓ SẴN:                                                              ║
║    diem_chuan   — tư vấn điểm chuẩn, cơ hội trúng tuyển                    ║
║    truong       — thông tin trường, học phí, cơ sở vật chất                 ║
║    nganh        — ngành học, môn học, nghề nghiệp sau ra trường              ║
║    to_hop       — tổ hợp xét tuyển A00/B00/D01...                           ║
║    huong_nghiep — định hướng chọn ngành theo đam mê & tố chất               ║
║    hoc_tap      — lộ trình học tập, kỹ năng cần có sau khi chọn ngành       ║
║    kien_thuc    — dạy kiến thức (lập trình, toán, khoa học...)              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CÁCH DÙNG NHANH:                                                            ║
║    1. pip install -r requirements.txt                                        ║
║    2. Đặt GROQ_API_KEY vào biến môi trường hoặc file .env                   ║
║    3. python tuyen_sinh_AI.py ingest   ← nạp dữ liệu (làm 1 lần)           ║
║    4. python tuyen_sinh_AI.py chat     ← test thử trên terminal             ║
║    5. python tuyen_sinh_AI.py server   ← khởi động server cho nhóm web      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1 — CẤU HÌNH TRUNG TÂM
# Chỉnh sửa ở đây khi cần thay đổi model, đường dẫn, hoặc thêm website crawl
# ══════════════════════════════════════════════════════════════════════════════

import os
import re
import sys
import uuid
import json
import time
import logging
from datetime import datetime

# Thư viện bên ngoài — cài bằng: pip install -r requirements.txt
import chromadb
from groq import Groq
import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from dotenv import load_dotenv

# Thư viện vẽ hình — cài thêm: pip install matplotlib networkx
try:
    import matplotlib
    matplotlib.use("Agg")  # không cần GUI, render sang file
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import networkx as nx
    _CAN_DRAW = True
except ImportError:
    _CAN_DRAW = False
    log_msg = "matplotlib/networkx chưa cài — tính năng vẽ hình bị tắt. Chạy: pip install matplotlib networkx numpy"

load_dotenv()  # Đọc API key từ file .env nếu có

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── API & Model ───────────────────────────────────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")   # lấy tại console.groq.com
LLM_MODEL       = "llama-3.3-70b-versatile"        # miễn phí, mạnh, tiếng Việt tốt
# EMBEDDING_MODEL không dùng — embedding chạy local qua ChromaDB DefaultEmbeddingFunction (ONNX)

# ── ChromaDB ─────────────────────────────────────────────────────────────────
# Đường dẫn tính từ vị trí file .py, không phụ thuộc thư mục đang chạy lệnh
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_DB_PATH  = os.path.join(_BASE_DIR, "chroma_db")
COLLECTION_NAME = "tuyen_sinh_2024"

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500   # số ký tự mỗi đoạn văn
CHUNK_OVERLAP = 50    # số ký tự chồng lấp giữa 2 đoạn liên tiếp

# ── RAG ──────────────────────────────────────────────────────────────────────
TOP_K_RESULTS = 5     # lấy 5 đoạn văn liên quan nhất khi tìm kiếm

# ── Thư mục dữ liệu ──────────────────────────────────────────────────────────
EXCEL_DIR = os.path.join(_BASE_DIR, "data", "excel")
PDF_DIR   = os.path.join(_BASE_DIR, "data", "pdf")

# ── Website cần crawl ────────────────────────────────────────────────────────
# THÊM TRƯỜNG MỚI: copy một block { } và điền vào
WEBSITES_TO_CRAWL = [
    {
        "url": "https://tuyensinh.hust.edu.vn",
        "truong": "DHBK_HN",
        "ten_truong": "ĐH Bách Khoa Hà Nội",
    },
    # {
    #     "url": "https://tuyensinh.vnu.edu.vn",
    #     "truong": "DHQGHN",
    #     "ten_truong": "ĐH Quốc Gia Hà Nội",
    # },
]


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1B — TỔ HỢP MÔN CHUẨN (bảng cứng, không để AI tự đoán)
#
# Nguồn: Quy chế tuyển sinh Bộ GD&ĐT. Cập nhật khi Bộ thay đổi.
# ══════════════════════════════════════════════════════════════════════════════

TO_HOP_TABLE: dict[str, list[str]] = {
    # Khối A
    "A00": ["Toán", "Vật lý", "Hóa học"],
    "A01": ["Toán", "Vật lý", "Tiếng Anh"],
    "A02": ["Toán", "Vật lý", "Sinh học"],
    "A05": ["Toán", "Hóa học", "Tiếng Anh"],
    "A06": ["Toán", "Vật lý", "Địa lý"],
    "A07": ["Toán", "Lịch sử", "Địa lý"],
    "A08": ["Toán", "Hóa học", "Sinh học"],
    "A09": ["Toán", "Địa lý", "Tiếng Anh"],
    "A10": ["Toán", "Vật lý", "Tin học"],
    "A14": ["Toán", "Tiếng Anh", "Tin học"],
    "A16": ["Toán", "Vật lý", "GDCD"],
    # Khối B
    "B00": ["Toán", "Hóa học", "Sinh học"],
    "B01": ["Toán", "Sinh học", "Tiếng Anh"],
    "B03": ["Toán", "Sinh học", "Lịch sử"],
    "B04": ["Toán", "Sinh học", "Địa lý"],
    "B08": ["Toán", "Sinh học", "GDCD"],
    # Khối C
    "C00": ["Ngữ văn", "Lịch sử", "Địa lý"],
    "C01": ["Ngữ văn", "Toán", "Vật lý"],
    "C02": ["Ngữ văn", "Toán", "Hóa học"],
    "C03": ["Ngữ văn", "Toán", "Lịch sử"],
    "C04": ["Ngữ văn", "Toán", "Địa lý"],
    "C05": ["Ngữ văn", "Vật lý", "Hóa học"],
    "C06": ["Ngữ văn", "Vật lý", "Sinh học"],
    "C07": ["Ngữ văn", "Hóa học", "Sinh học"],
    "C08": ["Ngữ văn", "Lịch sử", "GDCD"],
    "C14": ["Toán", "Ngữ văn", "GDCD"],
    "C19": ["Ngữ văn", "Lịch sử", "Tiếng Anh"],
    "C20": ["Ngữ văn", "Địa lý", "GDCD"],
    # Khối D
    "D01": ["Ngữ văn", "Toán", "Tiếng Anh"],
    "D02": ["Ngữ văn", "Toán", "Tiếng Nga"],
    "D03": ["Ngữ văn", "Toán", "Tiếng Pháp"],
    "D04": ["Ngữ văn", "Toán", "Tiếng Trung"],
    "D07": ["Toán", "Hóa học", "Tiếng Anh"],
    "D08": ["Toán", "Sinh học", "Tiếng Anh"],
    "D09": ["Toán", "Lịch sử", "Tiếng Anh"],
    "D10": ["Toán", "Địa lý", "Tiếng Anh"],
    "D14": ["Ngữ văn", "Lịch sử", "Tiếng Anh"],
    "D15": ["Ngữ văn", "Địa lý", "Tiếng Anh"],
    # Năng khiếu / Thể thao (tham khảo)
    "H00": ["Ngữ văn", "Năng khiếu 1", "Năng khiếu 2"],
    "T00": ["Toán", "Thể dục", "Năng khiếu"],
}

# Lookup ngược: từ danh sách môn → mã tổ hợp
_MON_ALIAS: dict[str, str] = {
    "toan": "Toán", "vat ly": "Vật lý", "ly": "Vật lý", "vật lý": "Vật lý",
    "hoa hoc": "Hóa học", "hoa": "Hóa học", "hóa": "Hóa học", "hóa học": "Hóa học",
    "sinh hoc": "Sinh học", "sinh": "Sinh học", "sinh học": "Sinh học",
    "ngu van": "Ngữ văn", "van": "Ngữ văn", "ngữ văn": "Ngữ văn",
    "lich su": "Lịch sử", "su": "Lịch sử", "lịch sử": "Lịch sử",
    "dia ly": "Địa lý", "dia": "Địa lý", "địa lý": "Địa lý",
    "tieng anh": "Tiếng Anh", "anh": "Tiếng Anh", "tiếng anh": "Tiếng Anh",
    "tin hoc": "Tin học", "tin": "Tin học", "tin học": "Tin học",
    "gdcd": "GDCD", "cong dan": "GDCD", "công dân": "GDCD",
    "tieng trung": "Tiếng Trung", "trung": "Tiếng Trung",
    "tieng phap": "Tiếng Pháp", "phap": "Tiếng Pháp",
    "tieng nga": "Tiếng Nga", "nga": "Tiếng Nga",
}

def tra_to_hop(ma_to_hop: str) -> str | None:
    """Tra cứu tổ hợp từ mã (vd: 'A01') → 'A01: Toán, Vật lý, Tiếng Anh'."""
    ma = ma_to_hop.strip().upper()
    if ma in TO_HOP_TABLE:
        mon = ", ".join(TO_HOP_TABLE[ma])
        return f"{ma}: {mon}"
    return None

def tim_ma_to_hop(cac_mon: list[str]) -> list[str]:
    """Từ danh sách môn học → tìm mã tổ hợp khớp hoàn toàn."""
    # Chuẩn hóa tên môn
    chuan = []
    for m in cac_mon:
        m_lower = m.strip().lower()
        chuan.append(_MON_ALIAS.get(m_lower, m.strip()))
    chuan_set = set(chuan)
    ket_qua = []
    for ma, mon_list in TO_HOP_TABLE.items():
        if set(mon_list) == chuan_set:
            ket_qua.append(f"{ma}: {', '.join(mon_list)}")
    return ket_qua

def to_hop_context() -> str:
    """Tạo chuỗi mô tả toàn bộ bảng tổ hợp để inject vào system prompt."""
    lines = ["BẢNG TỔ HỢP MÔN XÉT TUYỂN ĐẠI HỌC (CHUẨN BỘ GD&ĐT):"]
    for ma, mon in TO_HOP_TABLE.items():
        lines.append(f"  {ma}: {', '.join(mon)}")
    lines.append("")
    lines.append("LƯU Ý QUAN TRỌNG — CÁC NHẦM LẪN PHỔ BIẾN:")
    lines.append("  - A01 = Toán, Vật lý, Tiếng ANH (KHÔNG phải Tin học)")
    lines.append("  - A10 = Toán, Vật lý, Tin học (KHÔNG phải Tiếng Anh)")
    lines.append("  - Toán + Lý + Tin → A10, KHÔNG phải A01")
    lines.append("  - B00 = Toán, Hóa, Sinh (KHÔNG có Vật lý)")
    lines.append("  - D01 = Ngữ văn, Toán, Tiếng Anh (KHÔNG phải Toán, Lý, Anh)")
    lines.append("  - C00 = Ngữ văn, Lịch sử, Địa lý (KHÔNG có Tiếng Anh)")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1C — WEB SEARCH (tra mạng thực sự khi cần thông tin mới nhất)
# ══════════════════════════════════════════════════════════════════════════════

_WEB_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MagerokBot/1.0; +https://magerok.com)",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}

_TUYEN_SINH_SITES = [
    "tuyensinh247.com",
    "diemthi.24h.com.vn",
    "tuyensinh.vn",
]

def _google_search_urls(query: str, num: int = 5) -> list[str]:
    """
    Gọi DuckDuckGo HTML search (không cần API key) → trả về list URL.
    Ưu tiên các site tuyển sinh uy tín.
    """
    site_filter = " OR ".join(f"site:{s}" for s in _TUYEN_SINH_SITES)
    full_query  = f"{query} ({site_filter})"
    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": full_query, "kl": "vn-vi"},
            headers=_WEB_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        urls = []
        for a in soup.select("a.result__url"):
            href = a.get("href", "")
            if href.startswith("http") and len(urls) < num:
                urls.append(href)
        # fallback: lấy tất cả link kết quả
        if not urls:
            for a in soup.select(".result__title a"):
                href = a.get("href", "")
                if "duckduckgo.com" not in href and href.startswith("http"):
                    urls.append(href)
                if len(urls) >= num:
                    break
        return urls
    except Exception as e:
        log.warning(f"[WebSearch] DuckDuckGo lỗi: {e}")
        return []

def _fetch_page_text(url: str, max_chars: int = 3000) -> str:
    """Tải một trang web và trích xuất văn bản sạch."""
    try:
        resp = requests.get(url, headers=_WEB_HEADERS, timeout=10)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        # Xoá nav/footer/script/style
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "form", "iframe", "ads"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Gộp dòng trống
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines)[:max_chars]
    except Exception as e:
        log.warning(f"[WebFetch] {url} → {e}")
        return ""

def tim_kiem_web(query: str, n_trang: int = 3) -> str:
    """
    Tìm kiếm DuckDuckGo → tải nội dung trang → trả về chuỗi tổng hợp.
    Dùng khi ChromaDB không có dữ liệu hoặc câu hỏi cần thông tin mới nhất.
    """
    log.info(f"[WebSearch] Truy vấn: {query}")
    urls = _google_search_urls(query, num=n_trang + 2)
    if not urls:
        return ""
    doan_van = []
    for url in urls[:n_trang]:
        text = _fetch_page_text(url)
        if text and len(text) > 200:
            doan_van.append(f"[Nguồn: {url}]\n{text}")
        if len(doan_van) >= n_trang:
            break
    if not doan_van:
        return ""
    return "\n\n===\n\n".join(doan_van)


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1D — ENGINE VẼ HÌNH (Matplotlib / NetworkX)
#
# AI trả về tag [GENERATE_IMAGE: <mô tả>] → hàm ve_hinh() xử lý → base64 PNG
# ══════════════════════════════════════════════════════════════════════════════

import base64
import io
import re as _re

# Thư mục lưu ảnh sinh ra (dùng cho API trả file)
_IMG_DIR = os.path.join(_BASE_DIR, "static", "generated")
os.makedirs(_IMG_DIR, exist_ok=True)

_IMAGE_TAG_RE = _re.compile(r'\[GENERATE_IMAGE:\s*(.+?)\]', _re.IGNORECASE | _re.DOTALL)


def _fig_to_base64(fig) -> str:
    """Chuyển matplotlib Figure → base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode("utf-8")


def _parse_image_desc(desc: str) -> dict:
    """
    Phân tích mô tả tự nhiên → dict cấu hình:
    loai: 'plot' | 'geometry' | 'concept_map' | 'vector' | 'unit_circle'
    """
    d = desc.lower()
    if any(k in d for k in ["concept map", "concept_map", "mind map", "sơ đồ", "diagram"]):
        return {"loai": "concept_map"}
    if any(k in d for k in ["unit circle", "vòng tròn đơn vị", "circle sin cos"]):
        return {"loai": "unit_circle"}
    if any(k in d for k in ["triangle", "tam giác", "rectangle", "hình chữ nhật",
                             "polygon", "geometry", "hình học", "inclined plane"]):
        return {"loai": "geometry"}
    if any(k in d for k in ["vector", "force", "lực", "arrow"]):
        return {"loai": "vector"}
    # Mặc định: vẽ đồ thị hàm số
    return {"loai": "plot"}


def _ve_plot(desc: str):
    """Vẽ đồ thị hàm số từ mô tả tự nhiên."""
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#f8fbff")
    ax.set_facecolor("#ffffff")
    ax.grid(True, linestyle="--", alpha=0.5, color="#ccddee")

    x = np.linspace(-10, 10, 800)
    colors = ["#1F3A5F", "#2EC4B6", "#E05A2B", "#8B5CF6", "#059669"]
    plotted = 0

    # Trích hàm số từ mô tả (dạng y=..., f(x)=...)
    funcs = _re.findall(r'y\s*=\s*([^,\[\]]+?)(?:,|\band\b|$)', desc, _re.IGNORECASE)
    if not funcs:
        funcs = _re.findall(r'f\(x\)\s*=\s*([^,\[\]]+?)(?:,|\band\b|$)', desc, _re.IGNORECASE)
    if not funcs:
        # Fallback: vẽ y=x^2 nếu không parse được
        funcs = ["x**2"]

    for i, expr in enumerate(funcs[:5]):
        expr_py = (expr.strip()
                   .replace("^", "**")
                   .replace("sqrt", "np.sqrt")
                   .replace("sin", "np.sin")
                   .replace("cos", "np.cos")
                   .replace("tan", "np.tan")
                   .replace("log", "np.log")
                   .replace("abs", "np.abs")
                   .replace("pi", "np.pi")
                   .replace("exp", "np.exp"))
        try:
            y = eval(expr_py, {"x": x, "np": np, "__builtins__": {}})
            label = f"y = {expr.strip()}"
            ax.plot(x, y, color=colors[i % len(colors)], linewidth=2.2, label=label)
            plotted += 1
        except Exception:
            continue

    # Đánh dấu giao điểm nếu có 2 hàm
    if plotted == 2:
        try:
            f1 = eval(funcs[0].strip().replace("^", "**").replace("sqrt","np.sqrt")
                      .replace("sin","np.sin").replace("cos","np.cos")
                      .replace("pi","np.pi"), {"x": x, "np": np, "__builtins__": {}})
            f2 = eval(funcs[1].strip().replace("^", "**").replace("sqrt","np.sqrt")
                      .replace("sin","np.sin").replace("cos","np.cos")
                      .replace("pi","np.pi"), {"x": x, "np": np, "__builtins__": {}})
            diff = f1 - f2
            sign_change = np.where(np.diff(np.sign(diff)))[0]
            for idx in sign_change[:6]:
                ax.plot(x[idx], f1[idx], "o", color="#E05A2B", markersize=7, zorder=5)
        except Exception:
            pass

    ax.axhline(0, color="#333", linewidth=0.8)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(desc[:80], fontsize=11, color="#1F3A5F", pad=10)
    if plotted > 0:
        ax.legend(fontsize=10)
    ax.set_ylim(-20, 20)
    return fig


def _ve_hinh_hoc(desc: str):
    """Vẽ hình học cơ bản từ mô tả."""
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#f8fbff")
    ax.set_facecolor("#ffffff")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(desc[:80], fontsize=11, color="#1F3A5F", pad=10)

    d = desc.lower()

    # Tam giác vuông với cạnh a, b, c
    m_tri = _re.search(r'(?:sides?|cạnh)[^\d]*(\d+)[^\d]+(\d+)[^\d]+(\d+)', desc, _re.IGNORECASE)
    if "triangle" in d or "tam giác" in d:
        if m_tri:
            a, b, c = int(m_tri.group(1)), int(m_tri.group(2)), int(m_tri.group(3))
        else:
            a, b, c = 3, 4, 5
        pts = np.array([[0,0],[a,0],[0,b]])
        tri = plt.Polygon(pts, fill=True, facecolor="#EEF4FB", edgecolor="#1F3A5F", linewidth=2)
        ax.add_patch(tri)
        # Nhãn cạnh
        ax.text(a/2, -0.3, f"a={a}", ha="center", fontsize=11, color="#E05A2B", fontweight="bold")
        ax.text(-0.4, b/2, f"b={b}", ha="center", fontsize=11, color="#E05A2B", fontweight="bold")
        ax.text(a/2+0.2, b/2+0.2, f"c={c}", ha="center", fontsize=11, color="#2EC4B6", fontweight="bold")
        # Góc vuông
        sq = plt.Polygon([[0,0],[0.3,0],[0.3,0.3],[0,0.3]], fill=False,
                          edgecolor="#1F3A5F", linewidth=1.2)
        ax.add_patch(sq)
        # Nhãn góc
        ax.text(0.1, b+0.15, "A", fontsize=12, color="#1F3A5F", fontweight="bold")
        ax.text(a+0.1, 0, "B", fontsize=12, color="#1F3A5F", fontweight="bold")
        ax.text(-0.35, -0.35, "C", fontsize=12, color="#1F3A5F", fontweight="bold")
        ax.set_xlim(-1, a+1); ax.set_ylim(-1, b+1)

    elif "inclined plane" in d or "mặt phẳng nghiêng" in d:
        angle = 30
        m_ang = _re.search(r'(\d+)\s*degree', desc, _re.IGNORECASE)
        if m_ang: angle = int(m_ang.group(1))
        ang_r = np.radians(angle)
        L = 5
        # Mặt phẳng nghiêng
        pts = np.array([[0,0],[L*np.cos(ang_r), L*np.sin(ang_r)],[L*np.cos(ang_r),0]])
        tri = plt.Polygon(pts, fill=True, facecolor="#F3EBDD", edgecolor="#1F3A5F", linewidth=2)
        ax.add_patch(tri)
        # Khối
        bx, by = L*np.cos(ang_r)*0.5, L*np.sin(ang_r)*0.5
        block = plt.Rectangle((bx-0.3, by), 0.6, 0.5, angle=np.degrees(ang_r),
                                facecolor="#4E7FB6", edgecolor="#1F3A5F", linewidth=1.5)
        ax.add_patch(block)
        # Vectors lực
        ax.annotate("", xy=(bx, by-1.2), xytext=(bx, by),
                    arrowprops=dict(arrowstyle="->", color="#E05A2B", lw=2))
        ax.text(bx+0.1, by-0.7, "P (trọng lực)", fontsize=9, color="#E05A2B")
        ax.annotate("", xy=(bx-np.sin(ang_r)*1.2, by+np.cos(ang_r)*1.2), xytext=(bx, by),
                    arrowprops=dict(arrowstyle="->", color="#2EC4B6", lw=2))
        ax.text(bx-np.sin(ang_r)*1.3-0.5, by+np.cos(ang_r)*1.3, "N (pháp tuyến)",
                fontsize=9, color="#2EC4B6")
        ax.text(0.5, -0.3, f"{angle}°", fontsize=11, color="#1F3A5F", fontweight="bold")
        ax.set_xlim(-0.5, L+0.5); ax.set_ylim(-1.5, L*np.sin(ang_r)+1)
    else:
        # Hình chữ nhật mặc định
        rect = plt.Rectangle((0.5, 0.5), 4, 2.5, fill=True,
                               facecolor="#EEF4FB", edgecolor="#1F3A5F", linewidth=2)
        ax.add_patch(rect)
        ax.text(2.5, -0.1, "a", ha="center", fontsize=12, color="#E05A2B", fontweight="bold")
        ax.text(0.1, 1.75, "b", ha="center", fontsize=12, color="#E05A2B", fontweight="bold")
        ax.set_xlim(0, 5.5); ax.set_ylim(-0.5, 3.5)
    return fig


def _ve_vong_tron_don_vi():
    """Vẽ vòng tròn đơn vị với các góc đặc biệt."""
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#f8fbff")
    ax.set_facecolor("#ffffff")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)

    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="#1F3A5F", linewidth=2)
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.axvline(0, color="#555", linewidth=0.8)

    gocs = [(0,"0°","1","0"), (30,"30°","√3/2","1/2"), (45,"45°","√2/2","√2/2"),
            (60,"60°","1/2","√3/2"), (90,"90°","0","1"), (120,"120°","-1/2","√3/2"),
            (135,"135°","-√2/2","√2/2"), (150,"150°","-√3/2","1/2"),
            (180,"180°","-1","0"), (210,"210°","-√3/2","-1/2"),
            (240,"240°","-1/2","-√3/2"), (270,"270°","0","-1"),
            (300,"300°","1/2","-√3/2"), (315,"315°","√2/2","-√2/2"),
            (330,"330°","√3/2","-1/2")]

    for deg, label, cos_v, sin_v in gocs:
        rad = np.radians(deg)
        x, y = np.cos(rad), np.sin(rad)
        ax.plot(x, y, "o", color="#2EC4B6", markersize=6, zorder=5)
        offset = 0.18
        ax.text(x*(1+offset), y*(1+offset),
                f"{label}\n({cos_v}, {sin_v})",
                ha="center", va="center", fontsize=7.5,
                color="#1F3A5F", fontweight="bold")
        ax.plot([0, x], [0, y], color="#4E7FB6", linewidth=0.7, alpha=0.5)

    ax.set_title("Vòng tròn đơn vị — Các góc đặc biệt", fontsize=13,
                 color="#1F3A5F", fontweight="bold", pad=12)
    ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7)
    return fig


def _ve_concept_map(desc: str):
    """Vẽ sơ đồ khái niệm (concept map) bằng NetworkX."""
    if not _CAN_DRAW:
        return None

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#f8fbff")
    ax.set_facecolor("#ffffff")
    ax.set_title(desc[:80], fontsize=11, color="#1F3A5F", pad=10)

    G = nx.DiGraph()
    edges = []

    # Parse cặp A -> B từ mô tả
    arrows = _re.findall(r'([A-Za-zÀ-ỹ ]+?)\s*[-=>]+\s*([A-Za-zÀ-ỹ ]+?)(?:,|;|$)',
                         desc, _re.IGNORECASE)
    for src, dst in arrows:
        src, dst = src.strip(), dst.strip()
        if src and dst and len(src) < 40 and len(dst) < 40:
            G.add_edge(src, dst)
            edges.append((src, dst))

    if not G.nodes:
        # Fallback: sơ đồ mẫu
        G.add_edges_from([("Quang hợp","Phản ứng ánh sáng"),
                          ("Quang hợp","Phản ứng tối (Calvin)"),
                          ("Phản ứng ánh sáng","ATP + NADPH"),
                          ("Phản ứng tối (Calvin)","Glucose")])

    pos = nx.spring_layout(G, seed=42, k=2.5)
    node_colors = ["#1F3A5F" if G.in_degree(n)==0 else "#2EC4B6" for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=2200, alpha=0.92)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color="white",
                             font_size=9, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#4E7FB6",
                            arrows=True, arrowsize=22,
                            width=2, connectionstyle="arc3,rad=0.08")
    ax.axis("off")
    return fig


def ve_hinh(mo_ta: str) -> str | None:
    """
    Nhận mô tả (tiếng Anh hoặc Việt) → vẽ hình → trả về base64 PNG.
    Trả về None nếu không thể vẽ.
    """
    if not _CAN_DRAW:
        return None
    try:
        cfg = _parse_image_desc(mo_ta)
        loai = cfg["loai"]
        if loai == "unit_circle":
            fig = _ve_vong_tron_don_vi()
        elif loai == "geometry":
            fig = _ve_hinh_hoc(mo_ta)
        elif loai == "concept_map":
            fig = _ve_concept_map(mo_ta)
        elif loai == "vector":
            fig = _ve_hinh_hoc(mo_ta)  # dùng lại engine hình học
        else:
            fig = _ve_plot(mo_ta)
        return _fig_to_base64(fig)
    except Exception as e:
        log.warning(f"[VẼ HÌNH] Lỗi: {e}")
        return None


def xu_ly_anh_trong_tra_loi(tra_loi: str) -> tuple[str, list[str]]:
    """
    Quét câu trả lời AI tìm tag [GENERATE_IMAGE: ...].
    Trả về:
      - tra_loi đã xóa tag
      - list base64 PNG (có thể rỗng)
    """
    tags = _IMAGE_TAG_RE.findall(tra_loi)
    tra_loi_sach = _IMAGE_TAG_RE.sub("", tra_loi).strip()
    anh_b64_list = []
    for mo_ta in tags:
        b64 = ve_hinh(mo_ta.strip())
        if b64:
            anh_b64_list.append(b64)
            log.info(f"[VẼ HÌNH] Đã vẽ: {mo_ta[:60]}")
        else:
            log.warning(f"[VẼ HÌNH] Bỏ qua (lỗi): {mo_ta[:60]}")
    return tra_loi_sach, anh_b64_list
#
# Mỗi agent có:
#   - [TÊN]_SYSTEM : mô tả vai trò, viết 1 lần và cố định
#   - build_[tên]_prompt() : ghép dữ liệu + câu hỏi thành prompt hoàn chỉnh
# ══════════════════════════════════════════════════════════════════════════════


def _ngay_hom_nay() -> str:
    """Trả về chuỗi ngày giờ hiện tại để inject vào prompt."""
    now = datetime.now()
    thu = ["Thứ Hai","Thứ Ba","Thứ Tư","Thứ Năm","Thứ Sáu","Thứ Bảy","Chủ Nhật"][now.weekday()]
    return f"{thu}, ngày {now.day} tháng {now.month} năm {now.year}"

# ── Orchestrator — "Lễ tân" phân loại câu hỏi ────────────────────────────────

ORCHESTRATOR_SYSTEM = """
Bạn là hệ thống phân loại câu hỏi tuyển sinh đại học Việt Nam.

Nhiệm vụ duy nhất: đọc câu hỏi và trả về JSON với 2 trường: agents và can_hoi_them.

Các agent có sẵn:
- "diem_chuan"  : câu hỏi về điểm chuẩn, xét tuyển, cơ hội đỗ
- "truong"      : câu hỏi về thông tin trường, học phí, cơ sở vật chất
- "nganh"       : câu hỏi về ngành học, môn học, ra trường làm gì
- "to_hop"      : câu hỏi về tổ hợp xét tuyển (A00, B00, D01...)
- "huong_nghiep": câu hỏi về chọn ngành, định hướng nghề nghiệp, sở thích
- "hoc_tap"     : câu hỏi về cách học, lộ trình học tập, kỹ năng cần có sau khi chọn ngành
- "kien_thuc"   : câu hỏi muốn học/hiểu một kiến thức cụ thể (lập trình, toán, khoa học...)

Trường "can_hoi_them": câu hỏi ngắn để hỏi thêm thông tin nếu câu hỏi còn mơ hồ.
- Đặt "" nếu câu hỏi đã đủ thông tin để trả lời
- Đặt câu hỏi ngắn (1 câu) nếu thiếu thông tin quan trọng, ví dụ:
  + Chưa biết điểm số → hỏi "Bạn đang có khoảng bao nhiêu điểm và tổ hợp nào vậy?"
  + Chưa biết sở thích → hỏi "Bạn thích thiên về kỹ thuật, kinh doanh hay sáng tạo?"
  + Hỏi về ngành nhưng chưa biết trường → hỏi "Bạn muốn học ở khu vực nào, Hà Nội hay TP.HCM?"

Quy tắc:
- Trả về đúng định dạng JSON, không giải thích thêm gì
- Có thể chọn nhiều agent nếu câu hỏi liên quan nhiều chủ đề

Ví dụ:
  Câu hỏi: "Em 25 điểm A00, học CNTT trường nào được?"
  Trả về: {"agents": ["diem_chuan", "truong", "nganh"], "can_hoi_them": ""}

  Câu hỏi: "Em thích lập trình, nên chọn ngành gì?"
  Trả về: {"agents": ["huong_nghiep", "nganh"], "can_hoi_them": "Bạn đang có khoảng bao nhiêu điểm và học tổ hợp nào nhỉ?"}

  Câu hỏi: "Học CNTT thì cần học những gì?"
  Trả về: {"agents": ["hoc_tap", "nganh"], "can_hoi_them": ""}

  Câu hỏi: "Con trỏ trong C++ là gì vậy?"
  Trả về: {"agents": ["kien_thuc"], "can_hoi_them": ""}
"""

def build_orchestrator_prompt(cau_hoi: str) -> str:
    return f'Câu hỏi của học sinh: "{cau_hoi}"'


# ── Agent 1: Điểm chuẩn ───────────────────────────────────────────────────────

DIEM_CHUAN_SYSTEM = """
Bạn là người anh/chị đi trước đang giúp em học sinh hiểu về điểm chuẩn đại học.
Xưng "mình", gọi người hỏi là "bạn". Nói chuyện tự nhiên, thân thiện, dùng emoji phù hợp để tạo cảm giác gần gũi.

Ngày hôm nay: {ngay_hom_nay}. Dùng thông tin này khi được hỏi về thời gian hiện tại hoặc năm hiện tại.

Bạn sẽ được cung cấp dữ liệu điểm chuẩn thực tế. Dựa vào đó để tư vấn.

Nguyên tắc:
1. Luôn dùng số liệu cụ thể từ dữ liệu được cung cấp 📊
2. So sánh điểm học sinh với điểm chuẩn — cao hơn thì nói "an toàn" ✅,
   thấp hơn thì gợi ý phương án dự phòng cụ thể
3. Đề cập xu hướng điểm qua các năm nếu có dữ liệu 📈
4. Nếu không có dữ liệu cho trường/ngành được hỏi, nói thẳng và
   hướng dẫn kiểm tra trang chính thức
5. Tuyệt đối không bịa ra số liệu
6. Cuối câu trả lời, nếu cần thêm thông tin để tư vấn tốt hơn,
   hỏi thêm 1 câu ngắn gọn, tự nhiên

Trả lời bằng tiếng Việt. Ngắn gọn, đi thẳng vào vấn đề. Dùng emoji tự nhiên (không lạm dụng).
"""

def build_diem_chuan_prompt(du_lieu: str, cau_hoi: str, lich_su: str = "") -> str:
    prompt = f"Lịch sử trò chuyện:\n{lich_su}\n\n" if lich_su else ""
    prompt += f"""Dữ liệu điểm chuẩn liên quan:
---
{du_lieu}
---

Câu hỏi của học sinh: {cau_hoi}

Hãy tư vấn dựa trên dữ liệu trên. Nếu dữ liệu không đủ, nói thẳng
và hướng dẫn học sinh tìm thông tin ở đâu."""
    return prompt


# ── Agent 2: Thông tin trường ─────────────────────────────────────────────────

TRUONG_SYSTEM = """
Bạn là người anh/chị đang chia sẻ thật về các trường đại học Việt Nam.
Xưng "mình", gọi người hỏi là "bạn". Nói như đang nhắn tin cho bạn bè, không như viết báo cáo. Dùng emoji phù hợp để tạo cảm giác gần gũi.

Ngày hôm nay: {ngay_hom_nay}. Dùng thông tin này khi được hỏi về thời gian hiện tại hoặc năm hiện tại.

Bạn sẽ được cung cấp thông tin về trường từ cơ sở dữ liệu.

Nguyên tắc:
1. Trả lời đúng điều bạn hỏi — không liệt kê tất cả 🎯
2. Nêu cả ưu điểm lẫn điểm cần cân nhắc — tư vấn thật, không quảng cáo ✅
3. Nếu so sánh nhiều trường, phân tích rõ ràng theo từng tiêu chí 🏫
4. Chỉ nói những gì có trong dữ liệu — không bịa thông tin
5. Cuối câu trả lời hỏi thêm nếu cần để hiểu bạn muốn gì hơn

Trả lời bằng tiếng Việt. Dùng emoji tự nhiên (không lạm dụng).
"""

def build_truong_prompt(du_lieu: str, cau_hoi: str, lich_su: str = "") -> str:
    prompt = f"Lịch sử trò chuyện:\n{lich_su}\n\n" if lich_su else ""
    prompt += f"Thông tin các trường liên quan:\n---\n{du_lieu}\n---\n\nCâu hỏi: {cau_hoi}"
    return prompt


# ── Agent 3: Ngành học ────────────────────────────────────────────────────────

NGANH_SYSTEM = """
Bạn là người anh/chị đang làm trong ngành, chia sẻ thật về nghề nghiệp.
Xưng "mình", gọi người hỏi là "bạn". Tự nhiên như đang nói chuyện thật. Dùng emoji phù hợp để tạo cảm giác gần gũi.

Ngày hôm nay: {ngay_hom_nay}. Dùng thông tin này khi được hỏi về thời gian hiện tại hoặc năm hiện tại.

Nguyên tắc:
1. Giải thích theo ngôn ngữ dễ hiểu — không dùng từ kỹ thuật mà không giải thích 💡
2. Nêu cụ thể: học những môn gì, ra trường làm ở đâu, mức lương thực tế 💼
3. Trả lời thẳng vào câu hỏi
4. Nếu phù hợp, gợi ý thêm 1-2 ngành liên quan để bạn cân nhắc thêm 🔍
5. Cuối trả lời, hỏi thêm 1 câu để hiểu bạn hơn (ví dụ: bạn thiên về lý thuyết hay thực hành?)

Trả lời bằng tiếng Việt. Thân thiện, thực tế. Dùng emoji tự nhiên (không lạm dụng).
"""

def build_nganh_prompt(du_lieu: str, cau_hoi: str, lich_su: str = "") -> str:
    prompt = f"Lịch sử trò chuyện:\n{lich_su}\n\n" if lich_su else ""
    prompt += f"Thông tin ngành học liên quan:\n---\n{du_lieu}\n---\n\nCâu hỏi: {cau_hoi}"
    return prompt


# ── Agent 4: Tổ hợp môn ──────────────────────────────────────────────────────

_TO_HOP_SYSTEM_BASE = """
Bạn là người anh/chị giải thích rõ về tổ hợp môn xét tuyển đại học Việt Nam.
Xưng "mình", gọi người hỏi là "bạn". Giải thích đơn giản, dễ nhớ. Dùng emoji phù hợp để tạo cảm giác gần gũi.

Ngày hôm nay: {ngay_hom_nay}.

== BẢNG TỔ HỢP CHUẨN (BỘ GD&ĐT) — PHẢI DÙNG CHÍNH XÁC, KHÔNG TỰ SUY DIỄN ==

{bang_to_hop}

== NGUYÊN TẮC BẮT BUỘC ==
1. CHỈ dùng thông tin trong bảng trên — KHÔNG bịa, KHÔNG đoán, KHÔNG suy diễn
2. Nếu user hỏi mã tổ hợp → tra bảng → trả lời chính xác 3 môn
3. Nếu user kể 3 môn → tra bảng ngược → cho biết mã tổ hợp
4. Nếu không tìm thấy trong bảng → nói thẳng "tổ hợp này không có trong danh mục chuẩn"
5. TUYỆT ĐỐI KHÔNG nói "A01 gồm Toán, Lý, Tin" — A01 là Toán, Vật lý, TIẾNG ANH
6. TUYỆT ĐỐI KHÔNG nói "A10 gồm Toán, Lý, Anh" — A10 là Toán, Vật lý, TIN HỌC
7. Nhắc user kiểm tra đề án tuyển sinh từng trường vì tổ hợp xét tuyển có thể khác nhau 📋
8. Cuối câu trả lời hỏi thêm nếu cần

Trả lời bằng tiếng Việt. Chính xác tuyệt đối. Dùng emoji tự nhiên.
"""

def _build_to_hop_system() -> str:
    """Tạo system prompt với bảng tổ hợp cứng được inject vào."""
    return _TO_HOP_SYSTEM_BASE.replace("{bang_to_hop}", to_hop_context())

# TO_HOP_SYSTEM được resolve lúc runtime (sau khi to_hop_context() sẵn sàng)
TO_HOP_SYSTEM = _build_to_hop_system()

def build_to_hop_prompt(du_lieu: str, cau_hoi: str, lich_su: str = "") -> str:
    prompt = f"Lịch sử trò chuyện:\n{lich_su}\n\n" if lich_su else ""
    # du_lieu từ ChromaDB / web search — bổ sung thêm ngoài bảng cứng đã có trong system
    if du_lieu and du_lieu != "Không có dữ liệu liên quan trong hệ thống.":
        prompt += f"Thông tin bổ sung từ cơ sở dữ liệu:\n---\n{du_lieu}\n---\n\n"
    prompt += f"Câu hỏi: {cau_hoi}"
    return prompt


# ── Agent 5: Hướng nghiệp ────────────────────────────────────────────────────

HUONG_NGHIEP_SYSTEM = """
Bạn là người anh/chị thật sự quan tâm, đang ngồi nói chuyện 1-1 với em học sinh.
Xưng "mình", gọi người hỏi là "bạn". Không phải tư vấn viên, không phải chatbot — là người thật. Dùng emoji phù hợp để tạo cảm giác gần gũi.

Ngày hôm nay: {ngay_hom_nay}. Dùng thông tin này khi được hỏi về thời gian hiện tại hoặc năm hiện tại.

Bạn tiếp cận theo 4 chiều: ĐAM MÊ, TỐ CHẤT, TIỀM NĂNG, và ĐỊNH HƯỚNG THỬ NGHIỆM.

== CÁCH TIẾP CẬN ==

Nếu chưa đủ thông tin, hỏi thêm tự nhiên theo thứ tự:

1. ĐAM MÊ — "Bạn thích làm gì nhất khi rảnh? Chủ đề gì mà bạn tự tìm hiểu mà không cần ai nhắc?"

2. TỐ CHẤT — "Bạn bè hay nhờ bạn giúp việc gì? Bạn làm gì tốt hơn người khác mà không cần cố nhiều?"

3. TIỀM NĂNG — Phân tích tiềm năng trong 10-15 năm tới. Liên kết với xu hướng thị trường Việt Nam.

4. ĐỊNH HƯỚNG THỬ NGHIỆM — Gợi ý 1-2 hành động cụ thể có thể làm NGAY để kiểm chứng.
   Ví dụ: "Thử học 1 khóa Python miễn phí trên YouTube 2 tuần, thấy thích thì CNTT là hướng đúng."

== NGUYÊN TẮC ==
- Không phán xét bất kỳ sở thích nào
- Không chỉ nhìn vào điểm số — người giỏi toán chưa chắc hợp kỹ thuật
- Luôn nêu cả ưu điểm lẫn thách thức thật của từng ngành
- Nếu phân vân 2 ngành, phân tích: "Nếu chọn A, 5 năm sau sẽ làm gì? Nếu chọn B thì sao?"
- Kết thúc bằng 1 câu hỏi mở, chân thành

== XU HƯỚNG NGHỀ NGHIỆP ==
Ngành tăng trưởng mạnh ở Việt Nam 2025-2035:
- Công nghệ thông tin, AI, dữ liệu (thiếu nhân lực trầm trọng)
- Bán dẫn vi mạch (Việt Nam đang thu hút đầu tư lớn)
- Logistics và chuỗi cung ứng
- Năng lượng tái tạo (điện mặt trời, điện gió)
- Y tế và chăm sóc sức khỏe (dân số già hóa)
- Tài chính số, fintech

Trả lời bằng tiếng Việt. Thân thiện, thực tế, có chiều sâu.
"""


# ── Agent 6: Lộ trình học tập ────────────────────────────────────────────────

HOC_TAP_SYSTEM = """
Bạn là người anh/chị đang học hoặc đã ra trường ngành đó, chia sẻ lộ trình học tập thực tế.
Xưng "mình", gọi người hỏi là "bạn". Nói chuyện thật, không sách vở. Dùng emoji phù hợp để tạo cảm giác gần gũi.

Ngày hôm nay: {ngay_hom_nay}. Dùng thông tin này khi được hỏi về thời gian hiện tại hoặc năm hiện tại.

Khi được hỏi về cách học hoặc chuẩn bị cho một ngành, hãy tư vấn:

1. CHUẨN BỊ TRƯỚC KHI VÀO ĐẠI HỌC (nếu bạn còn đang học cấp 3):
   - Kiến thức nền cần có
   - Kỹ năng tốt để học đại học
   - Thứ có thể tự học thêm ngay bây giờ (khóa học, sách, dự án nhỏ)

2. LỘ TRÌNH 4 NĂM ĐẠI HỌC:
   - Năm 1-2: nền tảng cần nắm chắc là gì
   - Năm 3-4: chuyên sâu và thực tế ra sao
   - Điểm nào sinh viên hay "vấp" và cách vượt qua

3. KỸ NĂNG SONG SONG cần phát triển:
   - Kỹ năng mềm quan trọng với ngành đó
   - Tiếng Anh cần thiết đến đâu
   - Có nên học chứng chỉ thêm không, loại nào hữu ích

4. NGUỒN HỌC MIỄN PHÍ gợi ý:
   - Website, YouTube channel, khóa học online phù hợp
   - Cộng đồng, forum nên tham gia

Nguyên tắc:
- Thực tế, không lý thuyết suông
- Nếu chưa biết bạn học ngành gì, hỏi thêm trước
- Cuối trả lời hỏi bạn đang ở giai đoạn nào để tư vấn trúng hơn

Trả lời bằng tiếng Việt. Cụ thể, có thể áp dụng ngay.
"""

def build_hoc_tap_prompt(du_lieu: str, cau_hoi: str, lich_su: str = "") -> str:
    prompt = f"Lịch sử trò chuyện:\n{lich_su}\n\n" if lich_su else ""
    if du_lieu.strip() and "Không có dữ liệu" not in du_lieu:
        prompt += f"Thông tin ngành học tham khảo:\n---\n{du_lieu}\n---\n\n"
    prompt += f"""Bạn hỏi: {cau_hoi}

Hãy tư vấn lộ trình học tập cụ thể, thực tế. Nếu chưa rõ bạn đang học ngành gì
hoặc đang ở giai đoạn nào, hỏi thêm trước khi đưa ra lộ trình chi tiết."""
    return prompt


# ── Agent 7: Dạy kiến thức ───────────────────────────────────────────────────

KIEN_THUC_SYSTEM = """
Bạn là người anh/chị đang dạy kèm, giải thích kiến thức theo kiểu dễ hiểu nhất.
Xưng "mình", gọi người hỏi là "bạn". Kiên nhẫn, vui vẻ, không phán xét khi hỏi câu "ngây thơ". Dùng emoji phù hợp để tạo cảm giác gần gũi.

Ngày hôm nay: {ngay_hom_nay}. Dùng thông tin này khi được hỏi về thời gian hiện tại hoặc năm hiện tại.

== QUY TẮC KÝ HIỆU TOÁN HỌC — BẮT BUỘC TUÂN THỦ ==
Luôn dùng ký hiệu ASCII/text thuần, KHÔNG dùng Unicode toán học:
  - Nhân        : dùng  *        (KHÔNG dùng ×, ·)
  - Chia        : dùng  /        (KHÔNG dùng ÷)
  - Mũ          : dùng  ^        (KHÔNG dùng ², ³, chữ số nhỏ trên)
  - Căn bậc 2   : dùng  sqrt()   (KHÔNG dùng √)
  - Căn bậc n   : dùng  nrt()    (KHÔNG dùng ∛, ∜)
  - Phân số     : dùng  a/b      (KHÔNG dùng ½, ¾)
  - Tổng sigma  : dùng  sum()    (KHÔNG dùng Σ)
  - Tích phân   : dùng  int()    (KHÔNG dùng ∫)
  - Vô cực      : dùng  inf      (KHÔNG dùng ∞)
  - Góc         : dùng  goc()    hoặc angle() (KHÔNG dùng ∠)
  - Pi          : dùng  pi       (KHÔNG dùng π)
  - Delta       : dùng  delta    (KHÔNG dùng Δ, δ)
  - Thuộc       : dùng  in       (KHÔNG dùng ∈)
  - Suy ra      : dùng  =>       (KHÔNG dùng ⇒, →)
  - Tương đương : dùng  <=>      (KHÔNG dùng ⟺)

Ví dụ đúng : "3 * 2 = 6",  "x^2 + 2*x + 1 = 0",  "sqrt(16) = 4",  "S = pi * r^2"
Ví dụ SAI  : "3 × 2 = 6",  "x² + 2x + 1 = 0",    "√16 = 4",       "S = πr²"

== KHI NÀO TẠO HÌNH ẢNH ==
Nếu câu hỏi liên quan đến: đồ thị hàm số, hình học phẳng/không gian, sơ đồ khái niệm,
biểu đồ vật lý, chu kỳ tế bào, sơ đồ hóa học — hãy thêm dòng đặc biệt ở CUỐI câu trả lời:

[GENERATE_IMAGE: <mô tả bằng tiếng Anh, ngắn gọn, đủ để vẽ bằng matplotlib/networkx>]

Ví dụ:
  [GENERATE_IMAGE: plot y=x^2 and y=2*x+1 on same axes, mark intersection points, label axes x and y]
  [GENERATE_IMAGE: draw right triangle with sides a=3 b=4 c=5, label all sides and angles]
  [GENERATE_IMAGE: draw unit circle with angle 30 45 60 90 degrees marked, show sin cos values]
  [GENERATE_IMAGE: force diagram: block on inclined plane 30 degrees, show weight normal friction vectors]
  [GENERATE_IMAGE: concept map: Photosynthesis -> Light reaction, Dark reaction; show inputs CO2 H2O light outputs glucose O2]

Chỉ thêm [GENERATE_IMAGE] khi hình ảnh THỰC SỰ giúp hiểu bài — không spam.
Không thêm [GENERATE_IMAGE] cho câu hỏi thuần lý thuyết/định nghĩa.

== CÁCH GIẢI THÍCH ==
1. BẮT ĐẦU ĐƠN GIẢN: dùng ví dụ thực tế, so sánh với thứ quen thuộc
2. TĂNG DẦN ĐỘ SÂU: khái niệm → cách dùng → ví dụ/bài tập → lỗi thường gặp
3. CODE MẪU (lập trình): ngắn, có comment từng dòng
4. KIỂM TRA HIỂU BÀI: hỏi 1 câu nhỏ cuối bài để bạn tự suy nghĩ
5. GỢI Ý HỌC TIẾP: sau khái niệm này nên học gì

Nguyên tắc:
- Không bao giờ nói "câu hỏi này quá cơ bản"
- Dùng emoji ✅ ❌ 💡 làm nổi bật điểm quan trọng
- Khuyến khích bạn thử tự làm trước khi xem đáp án

Trả lời bằng tiếng Việt. Thân thiện như gia sư tốt nhất bạn từng gặp.
"""

def build_kien_thuc_prompt(du_lieu: str, cau_hoi: str, lich_su: str = "") -> str:
    prompt = f"Lịch sử trò chuyện:\n{lich_su}\n\n" if lich_su else ""
    prompt += f"""Bạn hỏi: {cau_hoi}

Hãy giải thích theo cách dễ hiểu nhất, dùng ví dụ thực tế.
Nhớ dùng ký hiệu ASCII thuần (*, ^, sqrt(), pi...) cho toán học.
Nếu bài cần hình minh hoạ thì thêm [GENERATE_IMAGE: ...] ở cuối.
Nếu câu hỏi chưa đủ rõ, hỏi thêm để giải thích trúng hơn."""
    return prompt

def build_huong_nghiep_prompt(du_lieu: str, cau_hoi: str, lich_su: str = "") -> str:
    prompt = f"Lịch sử trò chuyện:\n{lich_su}\n\n" if lich_su else ""
    if du_lieu.strip():
        prompt += f"Thông tin ngành học tham khảo:\n---\n{du_lieu}\n---\n\n"
    prompt += f"""Học sinh chia sẻ: {cau_hoi}

Hãy tư vấn theo 4 chiều: Đam mê, Tố chất, Tiềm năng, Định hướng thử nghiệm.
Nếu chưa đủ thông tin về học sinh, hãy hỏi thêm trước khi đưa ra gợi ý cụ thể."""
    return prompt



# ── Aggregator — tổng hợp kết quả từ nhiều agent ─────────────────────────────

AGGREGATOR_SYSTEM = """
Bạn tổng hợp thông tin từ nhiều nguồn và viết thành 1 câu trả lời hoàn chỉnh.
Xưng "mình", gọi người hỏi là "bạn". Viết tự nhiên như đang nhắn tin, không như bài luận. Dùng emoji phù hợp để tạo cảm giác gần gũi.

Ngày hôm nay: {ngay_hom_nay}. Dùng thông tin này khi được hỏi về thời gian hiện tại hoặc năm hiện tại.

Nguyên tắc:
1. Kết hợp thông tin tự nhiên — không copy nguyên xi, không lặp lại
2. Trả lời câu hỏi chính trước, thông tin bổ sung sau
3. Độ dài vừa phải — đủ để trả lời, không dài dòng
4. Kết thúc bằng 1 câu hỏi gợi mở ngắn, tự nhiên nếu bạn cần tư vấn thêm
5. Nếu nhiều agent đề xuất hỏi thêm, chỉ hỏi 1 câu thôi — câu quan trọng nhất

Trả lời bằng tiếng Việt. Thân thiện, tự nhiên, có emoji.
"""

def build_aggregator_prompt(cau_hoi_goc: str, cac_ket_qua: dict) -> str:
    ten_agent = {
        "diem_chuan":   "Chuyên gia điểm chuẩn",
        "truong":       "Chuyên gia thông tin trường",
        "nganh":        "Chuyên gia ngành học",
        "to_hop":       "Chuyên gia tổ hợp môn",
        "huong_nghiep": "Chuyên gia hướng nghiệp",
        "hoc_tap":      "Chuyên gia lộ trình học tập",
        "kien_thuc":    "Chuyên gia kiến thức",
    }
    ket_qua_text = ""
    for agent, ket_qua in cac_ket_qua.items():
        ten = ten_agent.get(agent, agent)
        ket_qua_text += f"--- {ten} ---\n{ket_qua}\n\n"

    return f"""Câu hỏi gốc của học sinh: "{cau_hoi_goc}"

Thông tin từ các chuyên gia:
{ket_qua_text}
Hãy tổng hợp thành một câu trả lời hoàn chỉnh, tự nhiên cho học sinh."""


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 3 — INGEST: nạp dữ liệu vào ChromaDB
# Chạy 1 lần lúc setup, chạy lại mỗi khi có dữ liệu tuyển sinh mới
# ══════════════════════════════════════════════════════════════════════════════

def _khoi_tao_chroma(reset: bool = False):
    """Tạo hoặc mở ChromaDB collection."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            log.info("Đã xóa collection cũ.")
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=DefaultEmbeddingFunction(),
    )
    log.info(f"Collection '{COLLECTION_NAME}' — {collection.count()} documents hiện có.")
    return collection


def _tao_groq_client():
    """Tạo Groq client dùng chung cho cả LLM và Embeddings."""
    return Groq(api_key=os.getenv("GROQ_API_KEY", GROQ_API_KEY))


def _embed(texts: list[str]) -> list[list[float]]:
    """
    Tạo vector embedding dùng ChromaDB DefaultEmbeddingFunction (onnxruntime).
    Không cần API key, chạy hoàn toàn local.
    """
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    ef = DefaultEmbeddingFunction()
    return ef(texts)


def _chunk_text(text: str) -> list[str]:
    """
    Cắt văn bản thành các đoạn nhỏ có chồng lấp.
    Ưu tiên cắt tại ranh giới câu để không cắt giữa chừng.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            boundary = max(
                text.rfind('. ', start, end),
                text.rfind('! ', start, end),
                text.rfind('? ', start, end),
                text.rfind('\n', start, end),
            )
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    return chunks


def _luu_vao_chroma(collection, groq_client, documents, metadatas, ids, ten_nguon):
    """Embed documents rồi lưu vào ChromaDB theo batch."""
    if not documents:
        return
    BATCH = 96
    log.info(f"  Đang embed {len(documents)} documents từ '{ten_nguon}'...")
    for i in tqdm(range(0, len(documents), BATCH), desc="  Embed", unit="batch"):
        batch_docs = documents[i:i+BATCH]
        batch_meta = metadatas[i:i+BATCH]
        batch_ids  = ids[i:i+BATCH]
        embeddings = _embed(batch_docs)
        collection.add(documents=batch_docs, embeddings=embeddings,
                       metadatas=batch_meta, ids=batch_ids)
    log.info(f"  Đã lưu {len(documents)} documents từ '{ten_nguon}'.")


def _suy_loai(filename: str) -> str:
    fn = filename.lower()
    if "diem_chuan" in fn or "diem-chuan" in fn: return "diem_chuan"
    if "de_an"      in fn or "de-an"      in fn: return "de_an"
    if "truong"     in fn:                        return "thong_tin_truong"
    if "nganh"      in fn:                        return "nganh"
    return "khac"


def _suy_nam(filename: str) -> int:
    m = re.search(r'(202\d)', filename)
    return int(m.group(1)) if m else 0


def _suy_truong(filename: str) -> str:
    fn = filename.replace('.pdf', '').upper()
    for code in ["DHBK_HN", "DHBK_HCM", "DHQGHN", "DHQG_HCM", "NEU", "HVKTQS"]:
        if code in fn:
            return code
    return ""


def nap_excel(collection, groq_client):
    """Đọc tất cả file Excel/CSV trong EXCEL_DIR và nạp vào ChromaDB."""
    if not os.path.exists(EXCEL_DIR):
        log.warning(f"Thư mục {EXCEL_DIR} không tồn tại, bỏ qua.")
        return
    files = [f for f in os.listdir(EXCEL_DIR) if f.endswith(('.xlsx', '.csv', '.xls'))]
    if not files:
        log.warning(f"Không có file Excel/CSV nào trong {EXCEL_DIR}.")
        return
    log.info(f"Tìm thấy {len(files)} file Excel/CSV.")

    for filename in files:
        log.info(f"  Đang đọc: {filename}")
        filepath = os.path.join(EXCEL_DIR, filename)
        df = pd.read_csv(filepath, encoding='utf-8-sig') if filename.endswith('.csv') \
             else pd.read_excel(filepath)

        loai = _suy_loai(filename)
        nam  = _suy_nam(filename)
        documents, metadatas, ids = [], [], []

        for _, row in df.iterrows():
            # Dùng cột "noi_dung" nếu có, không thì ghép tất cả cột lại
            if "noi_dung" in row and pd.notna(row["noi_dung"]):
                text = str(row["noi_dung"])
            else:
                parts = [f"{col}: {val}" for col, val in row.items()
                         if pd.notna(val) and str(val).strip()]
                text = " | ".join(parts)
            if not text.strip():
                continue

            documents.append(text)
            metadatas.append({
                "nguon":  "excel", "file": filename, "loai": loai,
                "nam":    int(row.get("nam", nam)) if "nam" in row else nam,
                "truong": str(row.get("truong", "")).strip(),
                "nganh":  str(row.get("nganh",  "")).strip(),
                "to_hop": str(row.get("to_hop", "")).strip(),
            })
            ids.append(str(uuid.uuid4()))

        _luu_vao_chroma(collection, groq_client, documents, metadatas, ids, filename)


def nap_pdf(collection, groq_client):
    """Đọc tất cả file PDF trong PDF_DIR, trích văn bản và nạp vào ChromaDB."""
    if not os.path.exists(PDF_DIR):
        log.warning(f"Thư mục {PDF_DIR} không tồn tại, bỏ qua.")
        return
    files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    if not files:
        log.warning(f"Không có file PDF nào trong {PDF_DIR}.")
        return
    log.info(f"Tìm thấy {len(files)} file PDF.")

    for filename in files:
        log.info(f"  Đang đọc: {filename}")
        filepath = os.path.join(PDF_DIR, filename)
        full_text = ""
        try:
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        full_text += t + "\n"
        except Exception as e:
            log.error(f"  Lỗi đọc PDF {filename}: {e}")
            continue

        if not full_text.strip():
            log.warning(f"  Không trích được văn bản từ {filename} (có thể là PDF scan).")
            continue

        chunks = _chunk_text(full_text)
        log.info(f"  → {len(chunks)} chunks từ {filename}")
        loai   = _suy_loai(filename)
        nam    = _suy_nam(filename)
        truong = _suy_truong(filename)

        documents = chunks
        metadatas = [{"nguon": "pdf", "file": filename, "loai": loai, "nam": nam,
                      "truong": truong, "chunk_idx": i, "nganh": "", "to_hop": ""}
                     for i in range(len(chunks))]
        ids = [str(uuid.uuid4()) for _ in chunks]
        _luu_vao_chroma(collection, groq_client, documents, metadatas, ids, filename)


def nap_web(collection, groq_client):
    """Crawl các website trong WEBSITES_TO_CRAWL và nạp vào ChromaDB."""
    if not WEBSITES_TO_CRAWL:
        log.warning("Chưa cấu hình WEBSITES_TO_CRAWL.")
        return

    headers = {"User-Agent": "Mozilla/5.0 (compatible; TuyenSinhBot/1.0)"}
    for site in WEBSITES_TO_CRAWL:
        url, truong = site["url"], site["truong"]
        log.info(f"  Đang crawl: {url}")
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            resp.encoding = 'utf-8'
        except Exception as e:
            log.error(f"  Lỗi crawl {url}: {e}")
            continue

        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.find("body")
        text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")
        text = re.sub(r'\n{3,}', '\n\n', text).strip()

        if len(text) < 100:
            log.warning(f"  Trang {url} gần như trống, bỏ qua.")
            continue

        chunks = _chunk_text(text)
        log.info(f"  → {len(chunks)} chunks từ {url}")
        documents = chunks
        metadatas = [{"nguon": "web", "url": url, "truong": truong,
                      "loai": "thong_tin_truong", "nam": 2024,
                      "chunk_idx": i, "nganh": "", "to_hop": ""}
                     for i in range(len(chunks))]
        ids = [str(uuid.uuid4()) for _ in chunks]
        _luu_vao_chroma(collection, groq_client, documents, metadatas, ids, url)
        time.sleep(1)


def chay_ingest():
    """Hàm chính của phần Ingest — gọi khi chạy: python tuyen_sinh_AI.py ingest"""
    print("=" * 60)
    print("  INGEST DỮ LIỆU TUYỂN SINH → CHROMADB")
    print("=" * 60)

    import sys; reset = sys.stdin.isatty() and input("\nReset toàn bộ DB cũ? (y/N): ").strip().lower() == 'y'
    collection  = _khoi_tao_chroma(reset=reset)
    groq_client = _tao_groq_client()
    os.makedirs(EXCEL_DIR, exist_ok=True)
    os.makedirs(PDF_DIR,   exist_ok=True)

    print("\n--- Nạp từ Excel/CSV ---")
    nap_excel(collection, groq_client)
    print("\n--- Nạp từ PDF ---")
    nap_pdf(collection, groq_client)
    print("\n--- Nạp từ Website ---")
    nap_web(collection, groq_client)

    print("\n" + "=" * 60)
    print(f"HOÀN TẤT! Tổng số documents trong DB: {collection.count()}")
    print(f"DB lưu tại: {os.path.abspath(CHROMA_DB_PATH)}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 4 — QUERY: nhận câu hỏi → tìm dữ liệu → gọi AI → trả lời
# ══════════════════════════════════════════════════════════════════════════════

class TuVanTuyenSinh:
    """
    Hệ thống tư vấn tuyển sinh — lớp chính để dùng từ bên ngoài.

    Cách dùng đơn giản nhất:
        bot = TuVanTuyenSinh()
        print(bot.hoi("Em thích lập trình, nên học ngành gì?"))
    """

    def __init__(self):
        log.info("Đang khoi dong he thong tu van...")
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = chroma_client.get_or_create_collection(
            COLLECTION_NAME,
            embedding_function=DefaultEmbeddingFunction(),
        )
        key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
        self.groq = Groq(api_key=key)  # dùng cho cả LLM lẫn Embeddings
        self.lich_su = []
        log.info("San sang!")

    def hoi(self, cau_hoi: str) -> dict:
        """
        Hàm duy nhất nhóm web cần gọi.
        Nhận câu hỏi → xử lý toàn bộ pipeline → trả về dict:
          {
            "tra_loi": str,          # Câu trả lời văn bản (đã xóa tag [GENERATE_IMAGE])
            "anh": list[str],        # List base64 PNG (rỗng nếu không vẽ hình)
          }
        """
        log.info(f"[Câu hỏi] {cau_hoi}")

        # Bước 1: Orchestrator xác định agent và có cần hỏi thêm không
        agents, can_hoi_them = self._phan_loai(cau_hoi)
        log.info(f"[Agents] Sẽ gọi: {agents}")

        # Nếu Orchestrator thấy cần hỏi thêm VÀ chưa có lịch sử đủ để trả lời
        # → trả về câu hỏi thêm, nhưng vẫn lưu câu hỏi gốc vào lịch sử
        if can_hoi_them:
            lich_su_dai = len(self.lich_su) >= 4
            if not lich_su_dai:
                log.info(f"[Hỏi thêm] {can_hoi_them}")
                self.lich_su.append({"role": "user", "content": cau_hoi})
                self.lich_su.append({"role": "assistant", "content": can_hoi_them})
                self.lich_su = self.lich_su[-20:]
                return {"tra_loi": can_hoi_them, "anh": []}

        # Bước 2: Mỗi agent tìm dữ liệu + trả lời độc lập
        ket_qua = {agent: self._chay_agent(agent, cau_hoi) for agent in agents}

        # Bước 3: Tổng hợp nếu nhiều agent, trả về luôn nếu chỉ 1
        tra_loi_raw = list(ket_qua.values())[0] if len(ket_qua) == 1 \
                      else self._tong_hop(cau_hoi, ket_qua)

        # Bước 4: Xử lý [GENERATE_IMAGE] — tách tag, vẽ hình, trả base64
        tra_loi, anh_list = xu_ly_anh_trong_tra_loi(tra_loi_raw)

        # Lưu lịch sử hội thoại (tối đa 20 lượt gần nhất)
        self.lich_su += [{"role": "user", "content": cau_hoi},
                         {"role": "assistant", "content": tra_loi}]
        self.lich_su = self.lich_su[-20:]

        return {"tra_loi": tra_loi, "anh": anh_list}

    def hoi_voi_anh(self, cau_hoi: str, image_base64: str, image_type: str = "image/jpeg") -> str:
        """
        Nhận câu hỏi kèm ảnh → phân tích ảnh bằng vision model → trả lời.
        Kết quả phân tích ảnh được bổ sung vào context rồi gọi pipeline thông thường.
        """
        log.info("[Vision] Đang phân tích ảnh...")

        # Bước 1: Gọi vision model phân tích ảnh
        try:
            vision_resp = self.groq.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                max_tokens=2000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{image_type};base64,{image_base64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Đây là ảnh liên quan đến tuyển sinh đại học Việt Nam. "
                                    "Hãy đọc và trích xuất TOÀN BỘ nội dung văn bản trong ảnh, "
                                    "đặc biệt chú ý:\n"
                                    "- Nếu là bảng dữ liệu/CSV: đọc từng dòng, từng cột, "
                                    "liệt kê đầy đủ tất cả tên ngành, mã ngành, điểm số, năm, tổ hợp xét tuyển. "
                                    "KHÔNG bỏ sót dòng nào, KHÔNG tóm tắt.\n"
                                    "- Nếu là bảng điểm/học bạ: đọc từng môn, từng điểm số chính xác.\n"
                                    "- Nếu là thông báo/văn bản: chép lại nguyên văn nội dung quan trọng.\n"
                                    "Trả lời bằng tiếng Việt. Liệt kê đầy đủ, không tóm tắt, không bỏ sót."
                                ),
                            },
                        ],
                    }
                ],
            )
            mo_ta_anh = vision_resp.choices[0].message.content.strip()
            log.info(f"[Vision] Mô tả ảnh: {mo_ta_anh[:100]}...")
        except Exception as e:
            log.warning(f"[Vision] Lỗi phân tích ảnh: {e}")
            mo_ta_anh = "(Không thể phân tích ảnh)"

        # Bước 2: Ghép mô tả ảnh vào câu hỏi rồi xử lý pipeline thông thường
        cau_hoi_day_du = f"{cau_hoi}\n\n[Nội dung ảnh đính kèm]: {mo_ta_anh}".strip()
        return self.hoi(cau_hoi_day_du)

    def reset_lich_su(self):
        """Xóa lịch sử — gọi khi người dùng bắt đầu hội thoại mới."""
        self.lich_su = []

    # ── Các bước nội bộ ─────────────────────────────────────────────────────

    def _phan_loai(self, cau_hoi: str) -> tuple[list[str], str]:
        """Gọi Orchestrator → lấy danh sách agent cần dùng và câu hỏi thêm nếu có."""
        resp = self.groq.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": ORCHESTRATOR_SYSTEM},
                {"role": "user",   "content": build_orchestrator_prompt(cau_hoi)},
            ],
            max_tokens=200,
        )
        try:
            text = resp.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text)
            agents = data.get("agents", ["nganh"])
            can_hoi_them = data.get("can_hoi_them", "").strip()
            hop_le = {"diem_chuan", "truong", "nganh", "to_hop", "huong_nghiep", "hoc_tap", "kien_thuc"}
            agents_sach = [a for a in agents if a in hop_le] or ["nganh"]
            return agents_sach, can_hoi_them
        except json.JSONDecodeError:
            return ["nganh"], ""

    def _tim_du_lieu(self, cau_hoi: str, loai_filter: str = None) -> str:
        """Tìm dữ liệu liên quan trong ChromaDB bằng vector search."""
        query_vec = _embed([cau_hoi])[0]
        try:
            where   = {"loai": loai_filter} if loai_filter else None
            results = self.collection.query(
                query_embeddings=[query_vec], n_results=TOP_K_RESULTS, where=where)
        except Exception:
            results = self.collection.query(
                query_embeddings=[query_vec], n_results=TOP_K_RESULTS)

        docs = results.get("documents", [[]])[0]
        return "\n---\n".join(docs) if docs else ""

    _KHONG_CO_DU_LIEU = "Không có dữ liệu liên quan trong hệ thống."

    # Agent nào nên tra web khi ChromaDB không đủ
    _CAN_WEB_SEARCH = {"diem_chuan", "truong", "nganh", "to_hop"}

    def _tim_du_lieu_voi_web(self, ten_agent: str, cau_hoi: str, loai_filter: str | None) -> str:
        """
        1. Tra ChromaDB trước.
        2. Nếu thiếu dữ liệu VÀ agent thuộc nhóm cần tra mạng → gọi tim_kiem_web().
        3. Ghép cả hai nguồn lại.
        """
        du_lieu_local = self._tim_du_lieu(cau_hoi, loai_filter)

        # Với agent to_hop: bảng cứng đã có trong system prompt, không cần web
        if ten_agent == "to_hop":
            return du_lieu_local or self._KHONG_CO_DU_LIEU

        # Các agent khác: tra web nếu ChromaDB trống hoặc quá ngắn
        can_web = (
            ten_agent in self._CAN_WEB_SEARCH
            and (not du_lieu_local or len(du_lieu_local) < 300)
        )
        if can_web:
            log.info(f"[WebSearch] ChromaDB thiếu dữ liệu cho '{ten_agent}' → tra mạng")
            du_lieu_web = tim_kiem_web(cau_hoi, n_trang=3)
            if du_lieu_web:
                if du_lieu_local:
                    return f"{du_lieu_local}\n\n--- Dữ liệu bổ sung từ web ---\n{du_lieu_web}"
                return f"[Dữ liệu từ web]\n{du_lieu_web}"

        return du_lieu_local or self._KHONG_CO_DU_LIEU

    def _chay_agent(self, ten_agent: str, cau_hoi: str) -> str:
        """Chạy một specialist agent: tìm dữ liệu → ghép prompt → gọi LLM."""
        cau_hinh = {
            "diem_chuan":   (DIEM_CHUAN_SYSTEM,   build_diem_chuan_prompt,   "diem_chuan"),
            "truong":       (TRUONG_SYSTEM,        build_truong_prompt,       "thong_tin_truong"),
            "nganh":        (NGANH_SYSTEM,         build_nganh_prompt,        "nganh"),
            "to_hop":       (TO_HOP_SYSTEM,        build_to_hop_prompt,       None),
            "huong_nghiep": (HUONG_NGHIEP_SYSTEM,  build_huong_nghiep_prompt, "nganh"),
            "hoc_tap":      (HOC_TAP_SYSTEM,       build_hoc_tap_prompt,      "nganh"),
            "kien_thuc":    (KIEN_THUC_SYSTEM,     build_kien_thuc_prompt,    None),
        }
        system_prompt, build_fn, loai_filter = cau_hinh[ten_agent]
        du_lieu = self._tim_du_lieu_voi_web(ten_agent, cau_hoi, loai_filter)

        lich_su_text = ""
        for msg in self.lich_su[-6:]:
            prefix = "Bạn" if msg["role"] == "user" else "Mình"
            lich_su_text += f"{prefix}: {msg['content']}\n"

        # Inject ngày giờ thực vào system prompt
        system_prompt_final = system_prompt.replace("{ngay_hom_nay}", _ngay_hom_nay())

        resp = self.groq.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt_final},
                {"role": "user",   "content": build_fn(du_lieu, cau_hoi, lich_su_text)},
            ],
            max_tokens=1200,
        )
        return resp.choices[0].message.content.strip()

    def _tong_hop(self, cau_hoi_goc: str, cac_ket_qua: dict) -> str:
        """Gọi Aggregator tổng hợp kết quả từ nhiều agent thành 1 câu trả lời."""
        aggregator_final = AGGREGATOR_SYSTEM.replace("{ngay_hom_nay}", _ngay_hom_nay())
        resp = self.groq.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": aggregator_final},
                {"role": "user",   "content": build_aggregator_prompt(cau_hoi_goc, cac_ket_qua)},
            ],
            max_tokens=1200,
        )
        return resp.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 5 — MAIN: chạy trên terminal hoặc khởi động server cho nhóm web
# ══════════════════════════════════════════════════════════════════════════════

def chay_chat():
    """Chat thử ngay trên terminal — dùng để test."""
    print("=" * 55)
    print("   AI TƯ VẤN TUYỂN SINH ĐẠI HỌC")
    print("   Gõ 'thoat' để thoát | 'moi' để bắt đầu lại")
    print("=" * 55)

    bot = TuVanTuyenSinh()
    while True:
        try:
            cau_hoi = input("\nBạn: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nTạm biệt!")
            break
        if not cau_hoi:
            continue
        if cau_hoi.lower() == "thoat":
            print("Tạm biệt!")
            break
        if cau_hoi.lower() == "moi":
            bot.reset_lich_su()
            print("--- Bắt đầu hội thoại mới ---")
            continue
        print(f"\nAI: {bot.hoi(cau_hoi)}")


def chay_server(port: int = 8000):
    """
    Khởi động HTTP server để nhóm web gọi vào.

    Nhóm web gọi: POST http://localhost:8000/hoi
    Body JSON: { "session_id": "user_abc", "cau_hoi": "Em hỏi gì đó..." }
    Response:  { "tra_loi": "...", "session_id": "user_abc" }
    """
    from http.server import HTTPServer, BaseHTTPRequestHandler

    sessions: dict[str, TuVanTuyenSinh] = {}

    def lay_bot(sid: str) -> TuVanTuyenSinh:
        if sid not in sessions:
            sessions[sid] = TuVanTuyenSinh()
        return sessions[sid]

    class Handler(BaseHTTPRequestHandler):
        def do_OPTIONS(self):
            self._headers(200)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                data    = json.loads(body)
                sid     = data.get("session_id", "default")
                cau_hoi = data.get("cau_hoi", "").strip()
                if not cau_hoi:
                    self._json({"loi": "Thiếu câu hỏi"}, 400)
                    return
                tra_loi = lay_bot(sid).hoi(cau_hoi)
                self._json({"tra_loi": tra_loi, "session_id": sid})
            except Exception as e:
                self._json({"loi": str(e)}, 500)

        def _headers(self, status):
            self.send_response(status)
            self.send_header("Content-Type",                  "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin",  "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def _json(self, data, status=200):
            self._headers(status)
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

        def log_message(self, *args):
            pass

    print(f"Server đang chạy tại http://localhost:{port}")
    print("Nhóm web gọi: POST /hoi  với body JSON: {session_id, cau_hoi}")
    print("Ctrl+C để tắt\n")
    HTTPServer(("", port), Handler).serve_forever()


# ── Điểm vào chương trình ────────────────────────────────────────────────────

def _huong_dan():
    print("""
╔══════════════════════════════════════════════════════╗
║         AI TƯ VẤN TUYỂN SINH — CÁCH DÙNG            ║
╠══════════════════════════════════════════════════════╣
║  python tuyen_sinh_AI.py ingest    Nạp dữ liệu       ║
║  python tuyen_sinh_AI.py chat      Chat thử terminal  ║
║  python tuyen_sinh_AI.py server    Server cho web     ║
╚══════════════════════════════════════════════════════╝

Lần đầu dùng:
  1. pip install chromadb google-generativeai pandas pdfplumber
              requests beautifulsoup4 sentence-transformers
              tqdm python-dotenv openpyxl
  2. Tạo file .env và thêm: GROQ_API_KEY=gsk_...
  3. Để file dữ liệu vào data/excel/ hoặc data/pdf/
  4. python tuyen_sinh_AI.py ingest
  5. python tuyen_sinh_AI.py chat
""")


if __name__ == "__main__":
    lenh = sys.argv[1] if len(sys.argv) > 1 else ""

    if lenh == "ingest":
        chay_ingest()
    elif lenh == "chat":
        chay_chat()
    elif lenh == "server":
        chay_server()
    else:
        _huong_dan()
