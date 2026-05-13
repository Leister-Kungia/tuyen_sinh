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

# Thư viện bên ngoài — cài bằng: pip install -r requirements.txt
import chromadb
from groq import Groq
import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()  # Đọc API key từ file .env nếu có

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── API & Model ───────────────────────────────────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")   # lấy tại console.groq.com
LLM_MODEL       = "llama-3.3-70b-versatile"        # miễn phí, mạnh, tiếng Việt tốt
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"  # Groq Embeddings API — không cần load model local

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
# PHẦN 2 — PROMPTS (kịch bản cho từng AI agent)
#
# Mỗi agent có:
#   - [TÊN]_SYSTEM : mô tả vai trò, viết 1 lần và cố định
#   - build_[tên]_prompt() : ghép dữ liệu + câu hỏi thành prompt hoàn chỉnh
# ══════════════════════════════════════════════════════════════════════════════

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
Xưng "mình", gọi người hỏi là "bạn". Nói chuyện tự nhiên, thân thiện.

Bạn sẽ được cung cấp dữ liệu điểm chuẩn thực tế. Dựa vào đó để tư vấn.

Nguyên tắc:
1. Luôn dùng số liệu cụ thể từ dữ liệu được cung cấp
2. So sánh điểm học sinh với điểm chuẩn — cao hơn thì nói "an toàn",
   thấp hơn thì gợi ý phương án dự phòng cụ thể
3. Đề cập xu hướng điểm qua các năm nếu có dữ liệu
4. Nếu không có dữ liệu cho trường/ngành được hỏi, nói thẳng và
   hướng dẫn kiểm tra trang chính thức
5. Tuyệt đối không bịa ra số liệu
6. Cuối câu trả lời, nếu cần thêm thông tin để tư vấn tốt hơn,
   hỏi thêm 1 câu ngắn gọn, tự nhiên

Trả lời bằng tiếng Việt. Ngắn gọn, đi thẳng vào vấn đề.
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
Xưng "mình", gọi người hỏi là "bạn". Nói như đang nhắn tin cho bạn bè, không như viết báo cáo.

Bạn sẽ được cung cấp thông tin về trường từ cơ sở dữ liệu.

Nguyên tắc:
1. Trả lời đúng điều bạn hỏi — không liệt kê tất cả
2. Nêu cả ưu điểm lẫn điểm cần cân nhắc — tư vấn thật, không quảng cáo
3. Nếu so sánh nhiều trường, phân tích rõ ràng theo từng tiêu chí
4. Chỉ nói những gì có trong dữ liệu — không bịa thông tin
5. Cuối câu trả lời hỏi thêm nếu cần để hiểu bạn muốn gì hơn

Trả lời bằng tiếng Việt.
"""

def build_truong_prompt(du_lieu: str, cau_hoi: str, lich_su: str = "") -> str:
    prompt = f"Lịch sử trò chuyện:\n{lich_su}\n\n" if lich_su else ""
    prompt += f"Thông tin các trường liên quan:\n---\n{du_lieu}\n---\n\nCâu hỏi: {cau_hoi}"
    return prompt


# ── Agent 3: Ngành học ────────────────────────────────────────────────────────

NGANH_SYSTEM = """
Bạn là người anh/chị đang làm trong ngành, chia sẻ thật về nghề nghiệp.
Xưng "mình", gọi người hỏi là "bạn". Tự nhiên như đang nói chuyện thật.

Nguyên tắc:
1. Giải thích theo ngôn ngữ dễ hiểu — không dùng từ kỹ thuật mà không giải thích
2. Nêu cụ thể: học những môn gì, ra trường làm ở đâu, mức lương thực tế
3. Trả lời thẳng vào câu hỏi
4. Nếu phù hợp, gợi ý thêm 1-2 ngành liên quan để bạn cân nhắc thêm
5. Cuối trả lời, hỏi thêm 1 câu để hiểu bạn hơn (ví dụ: bạn thiên về lý thuyết hay thực hành?)

Trả lời bằng tiếng Việt. Thân thiện, thực tế.
"""

def build_nganh_prompt(du_lieu: str, cau_hoi: str, lich_su: str = "") -> str:
    prompt = f"Lịch sử trò chuyện:\n{lich_su}\n\n" if lich_su else ""
    prompt += f"Thông tin ngành học liên quan:\n---\n{du_lieu}\n---\n\nCâu hỏi: {cau_hoi}"
    return prompt


# ── Agent 4: Tổ hợp môn ──────────────────────────────────────────────────────

TO_HOP_SYSTEM = """
Bạn là người anh/chị giải thích rõ về tổ hợp môn xét tuyển đại học Việt Nam.
Xưng "mình", gọi người hỏi là "bạn". Giải thích đơn giản, dễ nhớ.

Nguyên tắc:
1. Giải thích tổ hợp rõ ràng — ví dụ "A00 gồm Toán, Lý, Hóa"
2. Liệt kê đầy đủ các tổ hợp có thể dùng để xét tuyển ngành được hỏi
3. Nếu bạn biết tổ hợp của mình, gợi ý ngành phù hợp
4. Nhắc kiểm tra lại đề án tuyển sinh từng trường vì có thể khác nhau
5. Hỏi thêm nếu chưa biết tổ hợp của bạn để tư vấn trúng hơn

Trả lời bằng tiếng Việt. Chính xác, dễ hiểu.
"""

def build_to_hop_prompt(du_lieu: str, cau_hoi: str, lich_su: str = "") -> str:
    prompt = f"Lịch sử trò chuyện:\n{lich_su}\n\n" if lich_su else ""
    prompt += f"Thông tin tổ hợp liên quan:\n---\n{du_lieu}\n---\n\nCâu hỏi: {cau_hoi}"
    return prompt


# ── Agent 5: Hướng nghiệp ────────────────────────────────────────────────────

HUONG_NGHIEP_SYSTEM = """
Bạn là người anh/chị thật sự quan tâm, đang ngồi nói chuyện 1-1 với em học sinh.
Xưng "mình", gọi người hỏi là "bạn". Không phải tư vấn viên, không phải chatbot — là người thật.

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
Xưng "mình", gọi người hỏi là "bạn". Nói chuyện thật, không sách vở.

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
Xưng "mình", gọi người hỏi là "bạn". Kiên nhẫn, vui vẻ, không phán xét khi hỏi câu "ngây thơ".

Khi giải thích kiến thức:

1. BẮT ĐẦU ĐƠN GIẢN: dùng ví dụ thực tế, so sánh với thứ quen thuộc
   Ví dụ: "Con trỏ trong C++ giống như địa chỉ nhà — nó không phải là căn nhà, mà là địa chỉ để tìm đến căn nhà đó."

2. TĂNG DẦN ĐỘ SÂU: từ khái niệm cơ bản → cách dùng → ví dụ code/bài tập → lỗi thường gặp

3. CODE MẪU (nếu liên quan lập trình): viết code ngắn, có comment giải thích từng dòng

4. KIỂM TRA HIỂU BÀI: cuối phần giải thích, hỏi "Bạn thử đoán xem nếu mình làm X thì kết quả sẽ ra sao?" 
   hoặc đưa ra 1 câu hỏi nhỏ để bạn tự suy nghĩ

5. GỢI Ý HỌC TIẾP: sau khi hiểu khái niệm này, nên học gì tiếp theo

Nguyên tắc:
- Không bao giờ nói "câu hỏi này quá cơ bản" — mọi câu hỏi đều có giá trị
- Nếu câu hỏi chưa rõ, hỏi thêm để giải thích đúng trọng tâm
- Dùng emoji ✅ ❌ 💡 để làm nổi bật điểm quan trọng
- Khuyến khích bạn thử tự làm trước khi xem đáp án

Trả lời bằng tiếng Việt. Thân thiện như gia sư tốt nhất bạn từng gặp.
"""

def build_kien_thuc_prompt(du_lieu: str, cau_hoi: str, lich_su: str = "") -> str:
    prompt = f"Lịch sử trò chuyện:\n{lich_su}\n\n" if lich_su else ""
    prompt += f"""Bạn hỏi: {cau_hoi}

Hãy giải thích kiến thức này theo cách dễ hiểu nhất, dùng ví dụ thực tế.
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
Xưng "mình", gọi người hỏi là "bạn". Viết tự nhiên như đang nhắn tin, không như bài luận.

Nguyên tắc:
1. Kết hợp thông tin tự nhiên — không copy nguyên xi, không lặp lại
2. Trả lời câu hỏi chính trước, thông tin bổ sung sau
3. Độ dài vừa phải — đủ để trả lời, không dài dòng
4. Kết thúc bằng 1 câu hỏi gợi mở ngắn, tự nhiên nếu bạn cần tư vấn thêm
5. Nếu nhiều agent đề xuất hỏi thêm, chỉ hỏi 1 câu thôi — câu quan trọng nhất

Trả lời bằng tiếng Việt. Thân thiện, tự nhiên.
"""

def build_aggregator_prompt(cau_hoi_goc: str, cac_ket_qua: dict) -> str:
    ten_agent = {
        "diem_chuan":   "Chuyên gia điểm chuẩn",
        "truong":       "Chuyên gia thông tin trường",
        "nganh":        "Chuyên gia ngành học",
        "to_hop":       "Chuyên gia tổ hợp môn",
        "huong_nghiep": "Chuyên gia hướng nghiệp",
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

    reset = input("\nReset toàn bộ DB cũ? (y/N): ").strip().lower() == 'y'
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
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
        key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
        self.groq = Groq(api_key=key)  # dùng cho cả LLM lẫn Embeddings
        self.lich_su = []
        log.info("San sang!")

    def hoi(self, cau_hoi: str) -> str:
        """
        Hàm duy nhất nhóm web cần gọi.
        Nhận câu hỏi → xử lý toàn bộ pipeline → trả về câu trả lời.

        Trả về có thể là:
        - Câu trả lời thật sự
        - Câu hỏi ngược lại để làm rõ thêm (nếu thông tin chưa đủ)
        """
        print(f"\n[Câu hỏi] {cau_hoi}")

        # Bước 1: Orchestrator xác định agent và có cần hỏi thêm không
        agents, can_hoi_them = self._phan_loai(cau_hoi)
        print(f"[Agents] Sẽ gọi: {agents}")

        # Nếu Orchestrator thấy cần hỏi thêm VÀ chưa có lịch sử đủ để trả lời
        # → trả về câu hỏi thêm, nhưng vẫn lưu câu hỏi gốc vào lịch sử
        if can_hoi_them:
            # Kiểm tra lịch sử — nếu đã có info trong context thì bỏ qua, cứ trả lời
            lich_su_dai = len(self.lich_su) >= 4
            if not lich_su_dai:
                print(f"[Hỏi thêm] {can_hoi_them}")
                # Lưu câu hỏi gốc vào lịch sử để lần sau dùng làm context
                self.lich_su.append({"role": "user", "content": cau_hoi})
                self.lich_su.append({"role": "assistant", "content": can_hoi_them})
                self.lich_su = self.lich_su[-20:]
                return can_hoi_them

        # Bước 2: Mỗi agent tìm dữ liệu + trả lời độc lập
        ket_qua = {agent: self._chay_agent(agent, cau_hoi) for agent in agents}

        # Bước 3: Tổng hợp nếu nhiều agent, trả về luôn nếu chỉ 1
        tra_loi = list(ket_qua.values())[0] if len(ket_qua) == 1 \
                  else self._tong_hop(cau_hoi, ket_qua)

        # Lưu lịch sử hội thoại (tối đa 10 lượt gần nhất)
        self.lich_su += [{"role": "user", "content": cau_hoi},
                         {"role": "assistant", "content": tra_loi}]
        self.lich_su = self.lich_su[-20:]

        return tra_loi

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
        return "\n---\n".join(docs) if docs else "Không có dữ liệu liên quan trong hệ thống."

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
        du_lieu = self._tim_du_lieu(cau_hoi, loai_filter)

        lich_su_text = ""
        for msg in self.lich_su[-6:]:
            prefix = "Bạn" if msg["role"] == "user" else "Mình"
            lich_su_text += f"{prefix}: {msg['content']}\n"

        resp = self.groq.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": build_fn(du_lieu, cau_hoi, lich_su_text)},
            ],
            max_tokens=1200,
        )
        return resp.choices[0].message.content.strip()

    def _tong_hop(self, cau_hoi_goc: str, cac_ket_qua: dict) -> str:
        """Gọi Aggregator tổng hợp kết quả từ nhiều agent thành 1 câu trả lời."""
        resp = self.groq.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": AGGREGATOR_SYSTEM},
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
  2. Tạo file .env và thêm: GEMINI_API_KEY=AIza...
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
