# intents/appointment/flow.py
from typing import Dict, Any, List, Tuple, Optional
import re
import requests
from datetime import datetime 

from intents.intent_utils import get_user_appointments_for_creator, cancel_appointment_by_id
from config import USER_APPOINTMENTS_API, CANCEL_APPOINTMENT_API

from .time_parses import extract_date_and_time
from .slots_service import normalize_slots, format_slots, find_slots_for_all_doctors


# =======================
# Helpers chuẩn hoá chuỗi / bác sĩ
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
    """
    Lấy phần sau 'bác sĩ ...' để đoán tên.
    Ví dụ: 'đặt lịch bác sĩ Nguyễn Văn A chiều mai' -> 'Nguyễn Văn A'
    """
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
    """
    Thử đoán bác sĩ từ câu nói của user.
    - Ưu tiên phần sau 'bác sĩ ...'
    - Nếu không có, match cả câu với tên bác sĩ.
    """
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
        print(
            f"[appointment] best_by_after_bac_si: name='{best.get('name') if best else None}', score={best_score:.3f}"
        )
        if best and best_score >= 0.5:
            return best

    # 2) Fallback: match cả câu
    best, best_score = None, 0.0
    for d in doctors:
        sc = _score_name(user_text, d.get("name", ""))
        if sc > best_score:
            best, best_score = d, sc
    print(
        f"[appointment] best_by_full_sentence: name='{best.get('name') if best else None}', score={best_score:.3f}"
    )
    return best if best_score >= 0.4 else None


# =======================
# Lấy/ghi doctor_id trong state
# =======================
def _extract_doctor_id_from_state(state: Dict[str, Any]) -> Optional[Any]:
    keys = ["doctor_id", "doctorID", "doctorId", "userId"]
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
# Entry points – Flow đặt lịch
# =======================
def start_appointment_booking(
    state: Dict[str, Any],
    SLOTS_API: str,
    doctors: Optional[List[Dict[str, Any]]] = None,
    user_text: str = "",
) -> Tuple[str, Dict[str, Any]]:
    """
    Flow đặt lịch:
    - Nếu user KHÔNG chỉ định bác sĩ, nhưng có ngày + thời gian (giờ hoặc sáng/chiều/tối):
        -> quét tất cả bác sĩ, tìm những người rảnh vào khoảng đó, cho user chọn.
    - Nếu user CHỈ ĐỊNH bác sĩ:
        -> lấy toàn bộ lịch trống của bác sĩ đó.
    - Nếu user chỉ nói mỗi ngày, không nói giờ/buổi:
        -> hỏi lại: "Bạn muốn đặt lịch với bác sĩ nào, và vào lúc nào ạ?"
    """
    text = user_text or ""

    day, hour, minute, half_day = extract_date_and_time(text)
    has_time_info = (hour is not None) or (half_day is not None)
    print(f"[appointment] parsed_date={day}, hour={hour}, minute={minute}, half_day={half_day}")

    # 1) xác định doctor_id từ state hoặc tên trong câu
    doc_id = _extract_doctor_id_from_state(state)
    if not doc_id and doctors:
        cand = _find_best_doctor(doctors, user_text)
        if cand:
            state["selected_doctor"] = cand
            doc_id = (
                cand.get("doctorID")
                or cand.get("id")
                or cand.get("doctorId")
                or cand.get("userId")
            )
            state["doctor_id"] = doc_id

    # ===== CASE A: user không chỉ định bác sĩ, nhưng có ngày + thông tin thời gian (giờ hoặc sáng/chiều/tối) =====
    if day and has_time_info and not doc_id and doctors:
        desired_minutes: Optional[int] = None
        if hour is not None:
            desired_minutes = hour * 60 + minute

        candidates = find_slots_for_all_doctors(
            doctors=doctors,
            SLOTS_API=SLOTS_API,
            day=day,
            desired_minutes=desired_minutes,
            half_day=half_day,
        )

        if not candidates:
            return (
                "Hiện tại em chưa tìm được bác sĩ nào rảnh đúng thời gian đó. "
                "Anh/chị có thể cho em một khoảng thời gian linh hoạt hơn "
                "(ví dụ: sáng/chiều hoặc giờ khác) hoặc nói rõ muốn khám với bác sĩ nào ạ?",
                state,
            )

        state["step"] = "cho_chon_gio"
        state["pending_slots"] = candidates

        lines = [
            f"Vào khoảng **{day.strftime('%d/%m/%Y')}**",
            "Em tìm được các lịch sau, anh/chị chọn giúp em **số thứ tự** ạ:",
        ]
        for i, s in enumerate(candidates, 1):
            doc_name = (
                s.get("doctorName")
                or s.get("doctor_name")
                or s.get("location")
                or ""
            )
            disp = s.get("display") or s.get("datetime")
            lines.append(f"{i}. {disp}{(' • ' + doc_name) if doc_name else ''}")

        return "\n".join(lines), state

    # ===== CASE B: đã có doctor_id (user đã chỉ định bác sĩ hoặc chọn trước đó) =====
    if doc_id:
        print(f"[appointment] start_appointment_booking: doctor_id={doc_id}")
        try:
            url = SLOTS_API.format(id=doc_id)
            print(f"[appointment] fetching slots URL: {url}")
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            if resp.headers.get("content-type", "").startswith("application/json"):
                raw = resp.json()
            else:
                raw = []
            slots = normalize_slots(raw)
        except Exception as e:
            print(f"[appointment] fetch slots error: {e}")
            return "Xin lỗi, không lấy được lịch trống. Vui lòng thử lại sau.", state

        print(f"[appointment] slots received: {len(slots)}")

        state["step"] = "cho_chon_gio"
        state["pending_slots"] = slots
        doc_name = _doctor_label_from_state(state)
        prefix = f"{doc_name}\n\n" if doc_name else ""
        return prefix + format_slots(slots), state

    # ===== CASE C: không có doctor, cũng không đủ thông tin giờ/buổi =====
    return (
        "Bạn muốn đặt lịch với bác sĩ nào, và vào lúc nào ạ? "
        "(ví dụ: đặt lịch bác sĩ A lúc 15h chiều mai, hoặc sáng 15/12 khám bác sĩ nào cũng được)",
        state,
    )


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

    # Nếu slot có thông tin bác sĩ (case nhiều bác sĩ) -> lưu vào state
    doc_id = (
        chosen.get("doctorId")
        or chosen.get("doctor_id")
        or chosen.get("userId")
        or chosen.get("id")
    )
    doc_name = (
        chosen.get("doctorName")
        or chosen.get("doctor_name")
        or chosen.get("location")
    )
    if doc_id:
        state["doctor_id"] = doc_id
        state["selected_doctor"] = {
            "id": doc_id,
            "name": doc_name or "bác sĩ",
        }

    # chuyển sang nhập thông tin từng trường
    state["step"] = "nhap_thong_tin"
    state["patient_info"] = {}
    state["patient_step"] = "name"

    return (
        f"Bạn chọn khung giờ: {chosen['display']}.\n"
        f"Trước tiên, anh/chị cho em xin **Họ tên** của bệnh nhân ạ.",
        state,
    )

def handle_view_appointments(state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    creator_id = state.get("creator_id")
    if not creator_id:
        return (
            "Em chưa xác định được tài khoản của anh/chị. "
            "Anh/chị vui lòng đăng nhập lại rồi thử lại giúp em nhé.",
            state,
        )

    auth_token = state.get("auth_token")

    try:
        appts = get_user_appointments_for_creator(
            USER_APPOINTMENTS_API,
            int(creator_id),
            auth_token,
        )
    except Exception as e:
        print("[appointments] get_user_appointments_for_creator error:", e)
        return (
            "Em không lấy được danh sách lịch khám. Anh/chị vui lòng thử lại sau giúp em nhé.",
            state,
        )

    if not appts:
        return (
            "Hiện tại anh/chị **không có lịch khám nào đang active trong tương lai**.",
            state,
        )

    state["last_appointments"] = appts

    lines = ["Các lịch khám sắp tới của anh/chị:\n"]
    for a in appts:
        appt_id = a.get("appointmentID")
        doc = a.get("doctor") or {}
        doc_name = doc.get("name") or "—"
        time_str = a.get("time") or ""
        note = a.get("note") or ""

        try:
            dt = datetime.fromisoformat(time_str)
            time_fmt = dt.strftime("%d/%m/%Y %H:%M")
        except Exception:
            time_fmt = time_str

        lines.append(
            f"- ID **{appt_id}**: {time_fmt} với bác sĩ **{doc_name}**"
            + (f" (ghi chú: {note})" if note else "")
        )

    lines.append("")
    lines.append("Nếu anh/chị muốn hủy một lịch, hãy nói: *\"hủy lịch 80\"* (dùng ID).")

    return "\n".join(lines), state


def handle_cancel_appointment(
    user_input: str,
    state: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    auth_token = state.get("auth_token")
    if not auth_token:
        return (
            "Em chưa nhận được thông tin đăng nhập, anh/chị vui lòng đăng nhập lại rồi hủy lịch giúp em nhé.",
            state,
        )

    m = re.search(r"(\d+)", user_input)
    if not m:
        return (
            "Anh/chị vui lòng cho em **ID lịch cần hủy**, ví dụ: 'hủy lịch 80'.",
            state,
        )

    appt_id = int(m.group(1))

    ok = cancel_appointment_by_id(CANCEL_APPOINTMENT_API, appt_id, auth_token)
    if not ok:
        return (
            f"Em hủy lịch có ID **{appt_id}** không thành công. "
            "Anh/chị kiểm tra lại ID hoặc thử lại sau giúp em nhé.",
            state,
        )

    if "last_appointments" in state:
        state["last_appointments"] = [
            a for a in state["last_appointments"]
            if a.get("appointmentID") != appt_id
        ]

    return f"Em đã hủy lịch khám có ID **{appt_id}** thành công.", state