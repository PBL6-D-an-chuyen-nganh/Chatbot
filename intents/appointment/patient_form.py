# intents/appointment/patient_form.py
from email import header
from typing import Dict, Any, Tuple
import re
import requests


def _extract_doctor_id_from_state(state: Dict[str, Any]):
    keys = ["doctor_id", "doctorID", "doctorId", "userId"]
    for k in keys:
        if state.get(k) is not None:
            return state[k]
    sel = state.get("selected_doctor") or {}
    for k in keys + ["id"]:
        if sel.get(k) is not None:
            return sel[k]
    return None


def handle_patient_info(
    user_text: str, state: Dict[str, Any], APPOINTMENTS_API: str
) -> Tuple[str, Dict[str, Any]]:
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
        if len(digits) != 10:
            return "Số điện thoại chưa đúng lắm, anh/chị nhập lại giúp em (10 chữ số) nhé.", state

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

        raw_token = state.get("auth_token") 
        if raw_token:
            token = raw_token.strip()
            if not token.lower().startswith("bearer "):
                token = "Bearer " + token
            headers = {"Authorization": token}
        try:
            resp = requests.post(APPOINTMENTS_API, json=payload, headers=headers, timeout=100)
            resp.raise_for_status()
            _ = (
                resp.json()
                if resp.headers.get("content-type", "").startswith("application/json")
                else {}
            )
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
