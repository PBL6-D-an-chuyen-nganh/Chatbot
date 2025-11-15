from typing import Dict, Any, List, Tuple, Optional
import re
import requests
from datetime import datetime

# =======================
# Helpers chuẩn hoá chuỗi
# =======================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _split_tokens(s: str) -> List[str]:
    s = _norm(s)
    s = re.sub(
        r"[^0-9a-záàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữự\s]",
        " ",
        s,
    )
    return [t for t in s.split() if t]

def _score_name(q: str, name: str) -> float:
    qset, nset = set(_split_tokens(q)), set(_split_tokens(name))
    if not qset or not nset:
        return 0.0
    inter = qset & nset
    return 2.0 * len(inter) / float(len(qset) + len(nset))

def _guess_name_after_bac_si(text: str) -> Optional[str]:
    m = re.search(r"bác\s*sĩ\s+(.+)$", (text or ""), flags=re.IGNORECASE)
    if not m:
        return None
    cand = m.group(1)
    cand = re.split(
        r"(?:vào|ngày|tuần|tháng|năm|ở|tại|khoa|phòng|lúc|khoảng|trong)\b",
        cand,
        flags=re.IGNORECASE,
    )[0]
    cand = re.sub(r"[\.,:;!?()\[\]{}<>\"']", " ", cand)
    return re.sub(r"\s+", " ", cand).strip() or None

def _find_best_doctor(doctors: List[Dict[str, Any]], user_text: str) -> Optional[Dict[str, Any]]:
    if not doctors:
        return None

    # 1) Ưu tiên phần sau 'bác sĩ ...'
    name_after = _guess_name_after_bac_si(user_text)
    best, best_score = None, 0.0
    if name_after:
        for d in doctors:
            sc = _score_name(name_after, d.get("name", "")) + 0.1
            if sc > best_score:
                best, best_score = d, sc
        print(f"[appointment] best_by_after_bac_si: name='{best.get('name') if best else None}', score={best_score:.3f}")
        if best and best_score >= 0.5:
            return best

    # 2) fallback: match cả câu
    best, best_score = None, 0.0
    for d in doctors:
        sc = _score_name(user_text, d.get("name", ""))
        if sc > best_score:
            best, best_score = d, sc
    print(f"[appointment] best_by_full_sentence: name='{best.get('name') if best else None}', score={best_score:.3f}")
    return best if best_score >= 0.4 else None

# =======================
# Lấy/ghi doctor_id trong state
# =======================
def _extract_doctor_id_from_state(state: Dict[str, Any]) -> Optional[Any]:
    keys = ["doctor_id", "doctorID", "doctorId"]
    for k in keys:
        if state.get(k) is not None:
            return state[k]
    sel = state.get("selected_doctor") or {}
    for k in keys + ["id"]:
        if sel.get(k) is not None:
            return sel[k]
    return None

def _doctor_label_from_state(state: Dict[str, Any]) -> str:
    doc = state.get("selected_doctor") or {}
    return doc.get("name") or "bác sĩ đã chọn"

# =======================
# Chuẩn hoá slots
# =======================
def _normalize_slots(slots_raw: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(slots_raw, list):
        for s in slots_raw:
            if not isinstance(s, dict): 
                continue
            time_str = s.get("time") or s.get("startTime") or s.get("start")
            date_str = s.get("date")
            if time_str and date_str:
                dt = f"{date_str} {time_str}"
                out.append({
                    "date": date_str,
                    "time": time_str,
                    "datetime": dt,
                    "display": dt,
                    "location": s.get("location") or s.get("room") or "",
                })
    elif isinstance(slots_raw, dict):
        for date_str, times in slots_raw.items():
            if not isinstance(times, list): 
                continue
            for t in times:
                time_str = str(t)
                dt = f"{date_str} {time_str}"
                out.append({
                    "date": date_str,
                    "time": time_str,
                    "datetime": dt,
                    "display": dt,
                    "location": "",
                })
    # sort
    def _to_key(s):
        try:
            return datetime.strptime(s.get("datetime",""), "%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.max
    out.sort(key=_to_key)
    return out

def _format_slots(slots: List[Dict[str, Any]]) -> str:
    if not slots:
        return "❌ Hiện chưa có lịch trống nào."
    lines = [
        "🗓️ Các lịch trống (chọn số):",
        "(Sau khi chọn giờ, bạn sẽ được hướng dẫn nhập Họ tên, SĐT, Giới tính, Ngày sinh, Email, Triệu chứng)"
    ]
    for i, s in enumerate(slots, 1):
        lines.append(f"{i}. {s['display']}{(' • ' + s['location']) if s.get('location') else ''}")
    return "\n".join(lines)

# =======================
# Chuẩn hoá form bệnh nhân
# =======================
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def _normalize_gender(g: str) -> Optional[str]:
    if not g:
        return None
    gl = g.strip().lower()
    if gl in ("nam", "male", "m"):
        return "Male"
    if gl in ("nữ", "nu", "female", "f"):
        return "Female"
    if gl in ("khác", "khac", "other", "o"):
        return "Other"
    return None  

def _normalize_dob(s: str) -> Optional[str]:
    """
    Nhận 'YYYY-MM-DD' hoặc 'DD/MM/YYYY' → trả 'YYYY-MM-DD'.
    """
    if not s:
        return None
    s = s.strip()
    # thử YYYY-MM-DD
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    # thử DD/MM/YYYY
    try:
        dt = datetime.strptime(s, "%d/%m/%Y")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None

def _init_or_update_patient_draft(state: Dict[str, Any], **updates) -> Dict[str, Any]:
    draft = state.get("patient_draft") or {
        "name": "",
        "phone": "",
        "gender": "",
        "dateOfBirth": "",
        "email": "",
        "symptom": "",
    }
    for k, v in updates.items():
        if v is not None:
            draft[k] = v
    state["patient_draft"] = draft
    return draft

def _all_patient_fields_ready(d: Dict[str, Any]) -> bool:
    return (
        bool(d.get("name")) and
        bool(d.get("phone")) and
        bool(d.get("gender")) and
        bool(d.get("dateOfBirth")) and
        bool(d.get("email"))
    )

def _format_missing_fields_prompt(d: Dict[str, Any]) -> str:
    if not d.get("name") or not d.get("phone"):
        return "Vui lòng nhập theo mẫu: **Họ tên - SĐT - (tuỳ chọn) Triệu chứng**"
    if not d.get("gender"):
        return "Vui lòng nhập **Giới tính** (Nam/Nữ/Khác)."
    if not d.get("dateOfBirth"):
        return "Vui lòng nhập **Ngày sinh** theo định dạng YYYY-MM-DD hoặc DD/MM/YYYY."
    if not d.get("email"):
        return "Vui lòng nhập **Email**."
    return ""

def _try_parse_full_line_once(user_text: str) -> Dict[str, Any]:
    """
    Cố gắng parse 1 shot:
    'Họ tên - SĐT - Giới tính - Ngày sinh - Email - (tuỳ chọn) Triệu chứng'
    """
    parts = [p.strip() for p in user_text.split("-")]
    out = {
        "name": "",
        "phone": "",
        "gender": "",
        "dateOfBirth": "",
        "email": "",
        "symptom": "",
    }
    if len(parts) >= 1:
        out["name"] = parts[0]
    if len(parts) >= 2:
        out["phone"] = parts[1]
    if len(parts) >= 3:
        g = _normalize_gender(parts[2])
        out["gender"] = g or ""
    if len(parts) >= 4:
        dob = _normalize_dob(parts[3])
        out["dateOfBirth"] = dob or ""
    if len(parts) >= 5:
        out["email"] = parts[4] if _EMAIL_RE.match(parts[4]) else ""
    if len(parts) >= 6:
        out["symptom"] = parts[5]
    return out

# =======================
# Entry points
# =======================
def start_appointment_booking(
    state: Dict[str, Any],
    SLOTS_API: str,
    doctors: Optional[List[Dict[str, Any]]] = None,
    user_text: str = "",
) -> Tuple[str, Dict[str, Any]]:
    # 1) xác định doctor_id
    doc_id = _extract_doctor_id_from_state(state)
    if not doc_id and doctors:
        cand = _find_best_doctor(doctors, user_text)
        if cand:
            state["selected_doctor"] = cand
            doc_id = cand.get("doctorID") or cand.get("id") or cand.get("doctorId")
            state["doctor_id"] = doc_id

    if not doc_id:
        return "Bạn muốn đặt lịch với bác sĩ nào ạ?", state

    print(f"[appointment] start_appointment_booking: doctor_id={doc_id}")

    # 2) Fetch slots
    try:
        url = SLOTS_API.format(id=doc_id)  
        print(f"[appointment] fetching slots URL: {url}")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        if resp.headers.get("content-type", "").startswith("application/json"):
            raw = resp.json()
        else:
            raw = []
        slots = _normalize_slots(raw)
    except Exception as e:
        print(f"[appointment] fetch slots error: {e}")
        return "Xin lỗi, không lấy được lịch trống. Vui lòng thử lại sau.", state

    print(f"[appointment] slots received: {len(slots)}")

    state["step"] = "cho_chon_gio"
    state["pending_slots"] = slots
    # show bác sĩ kèm lịch
    doc_name = _doctor_label_from_state(state)
    prefix = f"{doc_name}\n\n" if doc_name else ""
    return prefix + _format_slots(slots), state

def handle_time_selection(user_text: str, state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    slots = state.get("pending_slots") or []
    m = re.search(r"\b(\d+)\b", user_text)
    if not m:
        return "Vui lòng nhập số thứ tự của lịch bạn muốn đặt (ví dụ: 1, 2, 3).", state

    idx = int(m.group(1)) - 1
    if idx < 0 or idx >= len(slots):
        return "Số bạn chọn không hợp lệ. Hãy chọn lại.", state

    chosen = slots[idx]
    state["chosen_slot"] = chosen
    state["step"] = "nhap_thong_tin"

    # reset/khởi tạo draft
    _init_or_update_patient_draft(state, symptom="")

    return (
        f"Bạn chọn khung giờ: {chosen['display']}.\n"
        f"Vui lòng nhập thông tin bệnh nhân. Bạn có thể:\n"
        f"• Nhập nhanh theo mẫu **Họ tên - SĐT - Giới tính - Ngày sinh - Email - (tuỳ chọn) Triệu chứng**\n"
        f"  Ví dụ: Nguyễn Văn A - 0912345678 - Nam - 01/02/1990 - a@example.com - Ngứa toàn thân\n"
        f"• Hoặc nhập tối thiểu **Họ tên - SĐT - (tuỳ chọn) Triệu chứng**, sau đó tôi sẽ hỏi tiếp.",
        state,
    )

def handle_patient_info(user_text: str, state: Dict[str, Any], APPOINTMENTS_API: str) -> Tuple[str, Dict[str, Any]]:
    # Lấy/khởi tạo draft
    draft = state.get("patient_draft") or {
        "name": "",
        "phone": "",
        "gender": "",
        "dateOfBirth": "",
        "email": "",
        "symptom": "",
    }

    # Thử parse 1 lần đầy đủ
    parsed = _try_parse_full_line_once(user_text)

    # Nếu người dùng nhập dạng cũ: 'Họ tên - SĐT - Triệu chứng'
    if parsed["name"] and parsed["phone"] and not parsed["gender"] and not parsed["dateOfBirth"] and not parsed["email"]:
        # cập nhật name/phone/symptom
        draft = _init_or_update_patient_draft(
            state,
            name=parsed["name"],
            phone=parsed["phone"],
            symptom=parsed["symptom"] or draft.get("symptom",""),
        )
    else:
        # Người dùng có thể đã nhập đầy đủ hoặc một phần các field mới
        if parsed["name"]:
            draft["name"] = parsed["name"]
        if parsed["phone"]:
            draft["phone"] = parsed["phone"]
        if parsed["symptom"]:
            draft["symptom"] = parsed["symptom"]
        if parsed["gender"]:
            draft["gender"] = parsed["gender"]
        if parsed["dateOfBirth"]:
            draft["dateOfBirth"] = parsed["dateOfBirth"]
        if parsed["email"]:
            draft["email"] = parsed["email"]
        state["patient_draft"] = draft

    # Nếu vẫn thiếu -> hỏi lần lượt
    missing_prompt = _format_missing_fields_prompt(draft)
    if missing_prompt:
        return missing_prompt, state

    # Validate cuối: email hợp lệ?
    if not _EMAIL_RE.match(draft["email"]):
        return "Email chưa hợp lệ. Vui lòng nhập lại email.", state

    # Xác thực slot & doctor
    slot = state.get("chosen_slot") or {}
    doc_id = _extract_doctor_id_from_state(state)
    creator_id = state.get("creator_id")  

    if not doc_id or not slot:
        return "Thiếu thông tin đặt lịch. Vui lòng bắt đầu lại với 'đặt lịch'.", state

    payload = {
        "patientInfo": {
            "name": draft["name"],
            "email": draft["email"],
            "phoneNumber": draft["phone"],
            "gender": draft["gender"],
            "dateOfBirth": draft["dateOfBirth"],
        },
        "time": f"{slot['date']}T{slot['time']}",
        "note": draft.get("symptom",""),
        "doctorId": doc_id,
        "creatorId": str(creator_id) if creator_id else None,
    }

    print(f"[appointment] booking payload: {payload}")

    try:
        resp = requests.post(APPOINTMENTS_API, json=payload, timeout=100)
        resp.raise_for_status()
        data = resp.json() if resp.headers.get("content-type","").startswith("application/json") else {}
    except Exception as e:
        print(f"[appointment] create appointment error: {e}")
        return "Xin lỗi, tạo lịch khám thất bại. Bạn vui lòng thử lại sau.", state

    # reset flow
    state.pop("pending_slots", None)
    state.pop("chosen_slot", None)
    state.pop("patient_draft", None)
    state["step"] = None

    code = (data or {}).get("code") or (data or {}).get("id") or "—"
    return f"Đã đặt lịch thành công. Hẹn gặp bạn tại phòng khám!", state
