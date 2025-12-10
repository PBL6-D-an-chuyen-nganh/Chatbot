from typing import Dict, Any, List, Tuple, Optional
import re
import requests
from datetime import datetime, date, timedelta

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
        print(f"[appointment] best_by_after_bac_si: name='{best.get('name') if best else None}', score={best_score:.3f}")
        if best and best_score >= 0.5:
            return best

    # 2) Fallback: match cả câu
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
# Chuẩn hoá slots
# =======================
def _normalize_slots(slots_raw: Any) -> List[Dict[str, Any]]:
    """
    Chuẩn hoá dữ liệu lịch trống về dạng:
    {
        "date": "YYYY-MM-DD",
        "time": "HH:MM[:SS]",
        "datetime": "YYYY-MM-DD HH:MM:SS",
        "display": "...",
        "location": "..."
    }
    """
    out: List[Dict[str, Any]] = []

    if isinstance(slots_raw, list):
        for s in slots_raw:
            if not isinstance(s, dict):
                continue
            time_str = s.get("time") or s.get("startTime") or s.get("start")
            date_str = s.get("date")
            if not time_str or not date_str:
                continue

            # Chuẩn hoá datetime "YYYY-MM-DD HH:MM:SS"
            if len(time_str) == 5:  # HH:MM
                dt_iso = f"{date_str} {time_str}:00"
            else:
                dt_iso = f"{date_str} {time_str}"

            out.append({
                "date": date_str,
                "time": time_str,
                "datetime": dt_iso,
                "display": dt_iso,
                "location": s.get("location") or s.get("room") or "",
            })

    elif isinstance(slots_raw, dict):
        for date_str, times in slots_raw.items():
            if not isinstance(times, list):
                continue
            for t in times:
                time_str = str(t)
                if len(time_str) == 5:
                    dt_iso = f"{date_str} {time_str}:00"
                else:
                    dt_iso = f"{date_str} {time_str}"

                out.append({
                    "date": date_str,
                    "time": time_str,
                    "datetime": dt_iso,
                    "display": dt_iso,
                    "location": "",
                })

    # sort
    def _to_key(s):
        try:
            return datetime.strptime(s.get("datetime", ""), "%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.max

    out.sort(key=_to_key)
    return out


def _format_slots(slots: List[Dict[str, Any]]) -> str:
    if not slots:
        return "❌ Hiện chưa có lịch trống nào."
    lines = [
        "🗓️ Các lịch trống (chọn số):",
        "(Sau khi chọn giờ, bạn sẽ được hướng dẫn nhập Họ tên và SĐT bệnh nhân.)"
    ]
    for i, s in enumerate(slots, 1):
        lines.append(f"{i}. {s['display']}{(' • ' + s['location']) if s.get('location') else ''}")
    return "\n".join(lines)

def _minutes_from_time_str(time_str: str) -> Optional[int]:
    """
    '07:35:00' -> 7*60+35
    '07:35'    -> 7*60+35
    """
    if not time_str:
        return None
    parts = time_str.split(":")
    if len(parts) < 2:
        return None
    try:
        h = int(parts[0])
        m = int(parts[1])
        return h * 60 + m
    except ValueError:
        return None

def _extract_date_and_time(user_text: str) -> Tuple[Optional[date], Optional[int], int, Optional[str]]:
    """
    Trả về:
      - day: datetime.date hoặc None
      - hour: int hoặc None
      - minute: int (mặc định 0)
      - half_day: 'morning' | 'afternoon' | 'evening' | None
    Hỗ trợ:
      - ngày: 15/12, 15-12, 15/12/2025, ...
      - giờ: 16:30, 16h30, 16h, 16 giờ, 7 giờ 35 phút, 7 gio 35 phut
    """
    text = (user_text or "").lower()

    # --- 1) Parse ngày: dd/mm(/yyyy) hoặc dd-mm(-yyyy) ---
    day: Optional[date] = None
    m_date = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{4}))?", text)
    if m_date:
        d = int(m_date.group(1))
        m = int(m_date.group(2))
        if m_date.group(3):
            y = int(m_date.group(3))
        else:
            y = datetime.now().year
        try:
            day = date(y, m, d)
        except ValueError:
            day = None

    # --- 2) Parse giờ ---
    hour: Optional[int] = None
    minute: int = 0

    # 2.1: dạng HH:MM
    m_time = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if m_time:
        hour = int(m_time.group(1))
        minute = int(m_time.group(2))
    else:
        # 2.2: dạng HHhMM
        m_time = re.search(r"\b(\d{1,2})h(\d{2})\b", text)
        if m_time:
            hour = int(m_time.group(1))
            minute = int(m_time.group(2))
        else:
            # 2.3: dạng "7 giờ 35 phút" / "7 gio 35 phut"
            m_time = re.search(r"\b(\d{1,2})\s*(?:giờ|gio|h)\s*(\d{1,2})\s*(?:phút|phut|p)?\b", text)
            if m_time:
                hour = int(m_time.group(1))
                minute = int(m_time.group(2))
            else:
                # 2.4: dạng "16h" / "16 giờ"
                m_time = re.search(r"\b(\d{1,2})\s*(?:h|giờ|gio)\b", text)
                if m_time:
                    hour = int(m_time.group(1))
                    minute = 0

    # --- 3) Sáng / chiều / tối ---
    half_day: Optional[str] = None
    if "sáng" in text:
        half_day = "morning"
    elif "chiều" in text:
        half_day = "afternoon"
    elif "tối" in text or "đêm" in text:
        half_day = "evening"

    return day, hour, minute, half_day


# =======================
# Entry points
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
        -> lấy toàn bộ lịch trống của bác sĩ đó (giống code cũ).
    - Nếu user chỉ nói mỗi ngày, không nói giờ/buổi:
        -> hỏi lại: "Bạn muốn đặt lịch với bác sĩ nào, và vào lúc nào ạ?"
    """
    text = user_text or ""
    text_lower = text.lower()

    day, hour, minute, half_day = _extract_date_and_time(text)
    has_time_info = (hour is not None) or (half_day is not None)
    print(f"[appointment] parsed_date={day}, hour={hour}, minute={minute}, half_day={half_day}")

    # 1) xác định doctor_id từ state hoặc tên trong câu
    doc_id = _extract_doctor_id_from_state(state)
    if not doc_id and doctors:
        cand = _find_best_doctor(doctors, user_text)
        if cand:
            state["selected_doctor"] = cand
            doc_id = cand.get("doctorId")
            state["doctor_id"] = doc_id

    # ===== CASE A: user không chỉ định bác sĩ, nhưng có ngày + thông tin thời gian (giờ hoặc sáng/chiều/tối) =====
    if day and has_time_info and not doc_id and doctors:
        date_str = day.strftime("%Y-%m-%d")
        desired_minutes: Optional[int] = None
        if hour is not None:
            desired_minutes = hour * 60 + minute

        NEAR_THRESHOLD = 35 

        all_candidates: List[Dict[str, Any]] = []

        for d in doctors:
            d_id = d.get("doctorID") or d.get("id") or d.get("doctorId") or d.get("userId")
            if not d_id:
                continue

            try:
                url = SLOTS_API.format(id=d_id)
                print(f"[appointment] fetching slots for doctor {d_id} URL: {url}")
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                raw = resp.json() if resp.headers.get("content-type","").startswith("application/json") else {}
                slots = _normalize_slots(raw)
            except Exception as e:
                print(f"[appointment] fetch slots error for doctor {d_id}: {e}")
                continue

            for s in slots:
                if s.get("date") != date_str:
                    continue

                slot_minutes = _minutes_from_time_str(s.get("time",""))
                if slot_minutes is None:
                    continue

                # Lọc theo buổi nếu có
                h_slot = slot_minutes // 60
                if half_day == "morning" and not (0 <= h_slot < 12):
                    continue
                if half_day == "afternoon" and not (12 <= h_slot < 18):
                    continue
                if half_day == "evening" and not (18 <= h_slot <= 23):
                    continue

                is_exact = False
                diff_min = None
                if desired_minutes is not None:
                    diff_min = abs(slot_minutes - desired_minutes)
                    is_exact = (diff_min == 0)
                else:
                    diff_min = 0

                all_candidates.append({
                    "date": s["date"],
                    "time": s["time"],
                    "datetime": s["datetime"],
                    "display": s["datetime"],
                    "location": d.get("name") or "",  # show tên bác sĩ sau dấu •
                    "doctorId": d_id,
                    "is_exact": is_exact,
                    "diff_min": diff_min,
                })

        if not all_candidates:
            return (
                "Hiện tại em chưa tìm được bác sĩ nào rảnh đúng thời gian đó. "
                "Anh/chị có thể cho em một khoảng thời gian linh hoạt hơn "
                "(ví dụ: sáng/chiều hoặc giờ khác) hoặc nói rõ muốn khám với bác sĩ nào ạ?",
                state,
            )

        # Ưu tiên slot khớp chính xác hh:mm
        exact_slots = [c for c in all_candidates if c["is_exact"]]
        if exact_slots:
            chosen_list = sorted(exact_slots, key=lambda c: c["datetime"])
        else:
            # Không có giờ chính xác -> lấy giờ gần xung quanh (± NEAR_THRESHOLD phút)
            near_slots = [
                c for c in all_candidates
                if c["diff_min"] is not None and c["diff_min"] <= NEAR_THRESHOLD
            ]
            if not near_slots:
                # quá xa, fallback như cũ
                return (
                    "Hiện tại em chưa tìm được bác sĩ nào rảnh đúng thời gian đó. "
                    "Anh/chị có thể cho em một khoảng thời gian linh hoạt hơn "
                    "(ví dụ: sáng/chiều hoặc giờ khác) hoặc nói rõ muốn khám với bác sĩ nào ạ?",
                    state,
                )
            # sắp xếp theo gần nhất trước
            chosen_list = sorted(near_slots, key=lambda c: (c["diff_min"], c["datetime"]))

        # Giới hạn số lượng gợi ý (ví dụ 10)
        chosen_list = chosen_list[:10]

        state["step"] = "cho_chon_gio"
        state["pending_slots"] = chosen_list

        # Format message cho user
        lines = [
            f"Vào khoảng **{day.strftime('%d/%m/%Y')}**",
            "Em tìm được các lịch sau, anh/chị chọn giúp em **số thứ tự** ạ:",
        ]
        for i, s in enumerate(chosen_list, 1):
            doc_name = s.get("location") or ""
            lines.append(f"{i}. {s['datetime']}{(' • ' + doc_name) if doc_name else ''}")

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
            slots = _normalize_slots(raw)
        except Exception as e:
            print(f"[appointment] fetch slots error: {e}")
            return "Xin lỗi, không lấy được lịch trống. Vui lòng thử lại sau.", state

        print(f"[appointment] slots received: {len(slots)}")

        state["step"] = "cho_chon_gio"
        state["pending_slots"] = slots
        doc_name = _doctor_label_from_state(state)
        prefix = f"{doc_name}\n\n" if doc_name else ""
        return prefix + _format_slots(slots), state

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
    doc_id = chosen.get("userId")
    doc_name = chosen.get("doctorName") or chosen.get("doctor_name")
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


def handle_patient_info(user_text: str, state: Dict[str, Any], APPOINTMENTS_API: str) -> Tuple[str, Dict[str, Any]]:
    """
    Flow nhập thông tin đơn giản:
    - Bước 1: hỏi Họ tên (patient_step = 'name')
    - Bước 2: hỏi SĐT   (patient_step = 'phone')
    Chỉ bắt buộc 2 field này. Các field khác để trống.
    """
    step = state.get("patient_step") or "name"
    info = state.get("patient_info") or {}

    text = user_text.strip()

    # Bước 1: Họ tên
    if step == "name":
        if not text:
            return "Anh/chị vui lòng cho em xin **Họ tên** của bệnh nhân ạ.", state

        info["name"] = text
        state["patient_info"] = info
        state["patient_step"] = "phone"

        return "Dạ em cảm ơn ạ. Anh/chị cho em xin thêm **Số điện thoại** liên hệ nhé.", state

    # Bước 2: Số điện thoại
    if step == "phone":
        digits = re.sub(r"\D", "", text)
        if len(digits) < 9:
            return "Số điện thoại chưa đúng lắm, anh/chị nhập lại giúp em (ít nhất 9 chữ số) nhé.", state

        info["phone"] = digits
        state["patient_info"] = info

        # Lấy slot & doctor để tạo lịch
        slot = state.get("chosen_slot") or {}
        doc_id = _extract_doctor_id_from_state(state)
        creator_id = state.get("creator_id")

        if not doc_id or not slot:
            return "Thiếu thông tin đặt lịch. Vui lòng bắt đầu lại với 'đặt lịch'.", state

        payload = {
            "patientInfo": {
                "name": info["name"],
                # Các field khác để rỗng (tuỳ backend của bạn)
                "email": "",
                "phoneNumber": info["phone"],
                "gender": "",
                "dateOfBirth": "",
            },
            "time": f"{slot['date']}T{slot['time']}",
            "note": "",
            "doctorId": doc_id,
            "creatorId": str(creator_id) if creator_id else None,
        }

        print(f"[appointment] booking payload: {payload}")

        try:
            resp = requests.post(APPOINTMENTS_API, json=payload, timeout=100)
            resp.raise_for_status()
            _ = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        except Exception as e:
            print(f"[appointment] create appointment error: {e}")
            return "Xin lỗi, tạo lịch khám thất bại. Bạn vui lòng thử lại sau.", state

        # reset flow
        state.pop("pending_slots", None)
        state.pop("chosen_slot", None)
        state.pop("patient_info", None)
        state["step"] = None
        state["patient_step"] = None

        doc_name = (state.get("selected_doctor") or {}).get("name") or "bác sĩ"
        display_time = slot.get("display") or f"{slot.get('date','')} {slot.get('time','')}"
        return (
            f"Đã đặt lịch thành công cho **{info['name']}**, SĐT **{info['phone']}** "
            f"với {doc_name} vào lúc **{display_time}**.\n"
            "Cảm ơn anh/chị đã đặt lịch khám!",
            state,
        )

    # fallback
    state["patient_step"] = "name"
    return "Anh/chị cho em xin **Họ tên** của bệnh nhân ạ.", state
