import requests
from datetime import datetime
import re
import unicodedata
import numpy as np
from sentence_transformers import util


def _strip_accents(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s

def _normalize_vi(s: str) -> str:
    """
    - lower
    - bỏ dấu
    - giữ lại chữ + số + khoảng trắng
    """
    s = (s or "").strip().lower()
    s = _strip_accents(s)
    s = re.sub(r"[^0-9a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _token_set(s: str):
    return set(_normalize_vi(s).split())


# =========================
# 1) Ưu tiên tìm theo tên bệnh (title)
# =========================
def _extract_disease_keyword(text: str) -> str:
    """
    Lấy phần 'tên bệnh' từ câu hỏi:
    - 'triệu chứng của viêm da cơ địa' -> 'viem da co dia'
    - 'bệnh lang ben nguyên nhân do đâu' -> 'lang ben'
    - 'nguyên nhân của ghé nước là gì' -> 'ghe nuoc'
    """
    if not text:
        return ""
    s = text.lower()

    # bỏ các cụm thường gặp trong câu hỏi
    s = re.sub(r"triệu chứng của|trieu chung cua", " ", s)
    s = re.sub(r"triệu chứng|trieu chung", " ", s)
    s = re.sub(r"nguyên nhân của|nguyen nhan cua", " ", s)
    s = re.sub(r"nguyên nhân|nguyen nhan", " ", s)
    s = re.sub(r"cách điều trị|cach dieu tri|điều trị|dieu tri", " ", s)
    s = re.sub(r"phòng ngừa|phong ngua|phòng tránh|phong tranh", " ", s)
    s = re.sub(r"là gì|la gi|do đâu|do dau|gây ra|gay ra", " ", s)
    s = re.sub(r"bệnh|benh", " ", s)

    # chuẩn hoá còn lại
    return _normalize_vi(s)

def keyword_match_articles_by_symptoms(
    query: str,
    articles,
    min_hits: int = 1,
    top_k: int = 5
):
    """
    Tìm bệnh giống LIKE nhiều từ khóa trên field `symptoms`.

    - Chuẩn hoá query (bỏ dấu, lower, giữ chữ + số + space)
    - Tách thành các keyword
    - Với mỗi article:
        + chuẩn hoá symptoms
        + đếm số keyword xuất hiện (substring)
    - Chỉ lấy article có hits >= min_hits
    - Sắp xếp theo hits giảm dần, trả về tối đa top_k

    Trả về: list[article], mỗi article có thêm field `_hits`.
    """
    if not query or not articles:
        return []

    norm_q = _normalize_vi(query)
    # bỏ keyword quá ngắn cho đỡ nhiễu
    keywords = [w for w in norm_q.split() if len(w) > 1]
    if not keywords:
        return []

    scored = []

    for art in articles:
        sym = art.get("symptoms") or ""
        norm_sym = _normalize_vi(sym)
        if not norm_sym:
            continue

        hits = 0
        for kw in keywords:
            if kw in norm_sym:
                hits += 1

        if hits >= min_hits:
            a = dict(art)   # copy để không sửa object gốc
            a["_hits"] = hits
            scored.append(a)

    scored.sort(key=lambda a: a["_hits"], reverse=True)
    return scored[:top_k]

def _title_match_score(query_kw: str, title: str) -> float:
    """
    Score trùng token giữa keyword và title:
    - dùng: |Q ∩ T| / |T|
    - nghĩa là: các từ trong title có xuất hiện trong câu hỏi không?
    """
    if not query_kw:
        return 0.0
    q_tokens = _token_set(query_kw)
    t_tokens = _token_set(title)
    if not t_tokens:
        return 0.0
    inter = q_tokens & t_tokens
    return len(inter) / float(len(t_tokens))


def _find_article_by_title_keyword(query: str, articles, min_score: float = 0.6):
    """
    Thử match theo tên bệnh (title) trước.
    Trả về article nếu score >= min_score, ngược lại trả None.
    """
    query_kw = _extract_disease_keyword(query)
    if not query_kw:
        return None

    best_article = None
    best_score = 0.0

    for art in articles or []:
        title = art.get("title", "")
        score = _title_match_score(query_kw, title)
        if score > best_score:
            best_score = score
            best_article = art

    if best_article and best_score >= min_score:
        print(f"[find_article] title keyword match: '{best_article.get('title')}' score={best_score:.3f}")
        return best_article

    return None


# =========================
# 2) Fallback: FAISS + cosine sim
# =========================
def find_articles_by_symptoms(query, model, index, articles, top_k=3):
    """
    Tìm các bệnh có triệu chứng khớp với mô tả của user.
    
    Args:
        query: Mô tả triệu chứng từ user (VD: "ngứa, nổi mẩn đỏ")
        model: Sentence transformer model để embed text
        index: FAISS index
        articles: Danh sách tất cả bài viết
        top_k: Số lượng bệnh trả về (mặc định 3)
    
    Returns:
        List các bài viết (bệnh) khớp nhất, sắp xếp theo độ khớp
    """
    import numpy as np
    
    # Embed câu hỏi
    query_emb = model.encode([query], convert_to_tensor=True)
    
    # Tìm kiếm với FAISS (lấy nhiều hơn top_k để có khoảng lọc)
    D, I = index.search(np.array(query_emb.cpu()), k=min(top_k * 3, len(articles)))
    
    results = []
    for idx, distance in zip(I[0], D[0]):
        if idx >= len(articles):
            continue
            
        article = articles[idx]
        
        # Lọc: chỉ lấy những bệnh có field "symptoms" 
        # và có độ khớp tốt (distance < 1.5)
        if article.get("symptoms") and distance < 1.5:
            results.append(article)
            if len(results) >= top_k:
                break
    
    return results

def find_article(query, model, index, articles, top_k: int = 5, min_sim: float = 0.45):
    """
    Tìm article an toàn hơn:
    1) Ưu tiên match theo title (tên bệnh) bằng keyword.
    2) Nếu không thấy:
       - dùng FAISS lấy top_k candidates,
       - tính cosine similarity bằng SentenceTransformer,
       - nếu sim < min_sim -> trả None (không đoán bừa).
    """
    # 1) thử match theo title
    art = _find_article_by_title_keyword(query, articles)
    if art is not None:
        return art

    # 2) fallback FAISS nếu có index + model
    if index is None or model is None or not articles:
        return None

    try:
        query_vec = model.encode([query], convert_to_numpy=True)  # (1, D)
    except Exception as e:
        print("⚠️ find_article encode error:", e)
        return None

    try:
        distances, indices = index.search(query_vec, top_k)
    except Exception as e:
        print("⚠️ find_article FAISS search error:", e)
        return None

    cand_ids = [int(i) for i in indices[0] if i >= 0]

    if not cand_ids:
        return None

    # Lấy text để tính sim: title + symptoms + cause
    cand_texts = []
    valid_ids = []
    for i in cand_ids:
        if 0 <= i < len(articles):
            art = articles[i]
            text = f"{art.get('title','')} {art.get('symptoms','')} {art.get('cause','')}"
            cand_texts.append(text)
            valid_ids.append(i)

    if not cand_texts:
        return None

    try:
        cand_embs = model.encode(cand_texts, convert_to_numpy=True)  # (k, D)
        sims = util.cos_sim(query_vec, cand_embs)[0]  # (k,)
        best_pos = int(np.argmax(sims))
        best_sim = float(sims[best_pos])
        best_idx = valid_ids[best_pos]
    except Exception as e:
        print("⚠️ find_article similarity error:", e)
        return None

    print(f"[find_article] best_sim={best_sim:.3f} article='{articles[best_idx].get('title')}'")

    if best_sim < min_sim:
        return None

    return articles[best_idx]


def find_doctor_by_name(name, doctors, model):
    names = [d["name"] for d in doctors]
    embeddings = model.encode(names, convert_to_numpy=True)
    query_vec = model.encode([name], convert_to_numpy=True)
    sims = util.cos_sim(query_vec, embeddings)[0]
    best_idx = int(np.argmax(sims))
    if sims[best_idx] > 0.6:
        return doctors[best_idx]
    return None


def get_available_slots(doctor_id, slots_api):
    try:
        url = slots_api.replace("{id}", str(doctor_id))
        resp = requests.get(url, timeout=3000)
        resp.raise_for_status()
        slots_data = resp.json()
        if not isinstance(slots_data, dict) or not slots_data:
            return []
        result = []
        for date_str, times in slots_data.items():
            for t in times:
                result.append(f"{date_str}T{t}")
        return result
    except Exception as e:
        print("⚠️ Lỗi lấy lịch trống:", e)
        return []


def create_appointment(data, appointments_api):
    try:
        resp = requests.post(appointments_api, json=data, timeout=15)
        if resp.status_code in [200, 201]:
            return True
        print("❌ Lỗi tạo lịch:", resp.status_code, resp.text)
    except Exception as e:
        print("⚠️ Không gửi được yêu cầu:", e)
    return False

def get_user_appointments_for_creator(api_template: str, creator_id: int, auth_token: str | None = None):
    """
    Gọi API lấy tất cả lịch của 1 user, lọc:
    - status == 'active'
    - time > hiện tại
    """
    url = api_template.replace("{creatorId}", str(creator_id))

    headers = {}
    if auth_token:
        headers["Authorization"] = auth_token

    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    items = []
    if isinstance(data, dict):
        items = data.get("content") or []
    elif isinstance(data, list):
        items = data
    else:
        return []

    now = datetime.now()
    result = []
    for a in items:
        if a.get("status") != "active":
            continue
        t_str = a.get("time")
        if not t_str:
            continue
        try:
            dt = datetime.fromisoformat(t_str)
        except Exception:
            # parse lỗi thì bỏ qua
            continue
        if dt >= now:
            result.append(a)

    # sắp xếp theo thời gian tăng dần
    result.sort(key=lambda x: x.get("time", ""))
    return result


def cancel_appointment_by_id(api_template: str, appointment_id: int, auth_token: str | None = None) -> bool:
    url = api_template.replace("{id}", str(appointment_id))

    headers = {}
    if auth_token:
        headers["Authorization"] = auth_token

    try:
        resp = requests.delete(url, headers=headers, timeout=10)
        if resp.status_code in (200, 204):
            return True
        print("❌ Hủy lịch thất bại:", resp.status_code, resp.text)
    except Exception as e:
        print("⚠️ Lỗi gọi API hủy lịch:", e)
    return False

