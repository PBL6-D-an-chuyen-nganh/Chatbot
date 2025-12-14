from typing import Dict, Any, Tuple, List, Optional
import re
import unicodedata

def _strip_accents(s: str) -> str:
    """
    Bỏ dấu tiếng Việt: 'Mỹ Nhi' -> 'my nhi'
    """
    if not s:
        return ""
    s = s.strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


def _norm(s: str) -> str:
    s = _strip_accents(s or "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _split_tokens(s: str) -> List[str]:
    s = _norm(s)
    s = re.sub(r"[^0-9a-z\s]", " ", s)
    raw_tokens = [t for t in s.split() if t]

    # Bỏ bớt stop-word phổ biến trong câu hỏi
    STOP_WORDS = {
        "cho", "toi", "toi",
        "xin", "hoi", "thong", "tin",
        "bac", "si", "bacsi", "bs",
        "ve", "thay", "muon",
        "can", "lam", "nhan", "vien",
    }

    return [t for t in raw_tokens if t not in STOP_WORDS]

def _score_name(q: str, name: str) -> float:
    qset, nset = set(_split_tokens(q)), set(_split_tokens(name))
    if not qset or not nset:
        return 0.0
    inter = qset & nset
    return 2.0 * len(inter) / float(len(qset) + len(nset))


def _find_best_doctor(doctors: List[Dict[str, Any]], query: str) -> Optional[Dict[str, Any]]:
    if not doctors:
        return None
    best, best_score = None, 0.0
    for d in doctors:
        sc = _score_name(query, d.get("name", ""))
        if sc > best_score:
            best, best_score = d, sc
    print(f"[doctor_intent] best match='{best.get('name') if best else None}', score={best_score:.3f}")
    return best if best_score >= 0.4 else None


# =========================
# Handler chính
# =========================
def handle_doctor_intent(
    entity: str,
    doctors: List[Dict[str, Any]],
    state: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:

    if not doctors:
        return "Hiện tại hệ thống chưa có danh sách bác sĩ. Bạn vui lòng thử lại sau nhé.", state

    text = (entity or "").strip()
    if not text:
        return "Bạn muốn tìm thông tin về bác sĩ nào ạ?", state

    # Tìm bác sĩ phù hợp nhất (fuzzy, không phân biệt dấu)
    best = _find_best_doctor(doctors, text)

    if not best:
        return (
            "Em chưa tìm được bác sĩ nào phù hợp với tên anh/chị cung cấp. "
            "Anh/chị có thể cho em tên đầy đủ hơn được không ạ?",
            state,
        )

    doc_id = (
        best.get("doctorID")
        or best.get("doctorId")
        or best.get("id")
        or best.get("userId")
    )

    state["selected_doctor"] = best
    if doc_id is not None:
        state["doctor_id"] = doc_id

    name = best.get("name") or "—"
    email = best.get("email") or "—"
    phone = best.get("phoneNumber") or best.get("phone") or "—"
    ach = best.get("achievements") or "—"
    pos = best.get("position") or ""
    deg = best.get("degree") or ""
    intro = best.get("introduction") or best.get("bio") or ""

    reply_lines = [
        f"👨‍⚕️ Thông tin bác sĩ: **{name}**",
    ]
    if pos or deg:
        reply_lines.append(f"• Chức vụ / Học vị: {pos} {deg}".strip())
    if phone or email:
        reply_lines.append(f"• Liên hệ: {phone} | {email}")
    if intro:
        reply_lines.append("")
        reply_lines.append(f"📝 Giới thiệu: {intro}")
    if ach:
        reply_lines.append("")
        reply_lines.append(f"🏆 Thành tựu: {ach}")

    reply_lines.append("")
    reply_lines.append(
        "Nếu bạn muốn đặt lịch với bác sĩ này, hãy nói: "
        "*\"đặt lịch khám với bác sĩ này\"* hoặc nêu rõ ngày giờ mong muốn nhé."
    )

    return "\n".join(reply_lines), state
