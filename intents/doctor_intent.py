# intents/doctor_intent.py
from typing import Dict, Any, Tuple, List
from intents.appointment_intent import _score_name

def handle_doctor_intent(entity, doctors: List[Dict[str, Any]], st_model, state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    text = (entity or "").strip()
    if not doctors:
        return "Hiện chưa có danh sách bác sĩ.", state

    # tìm bác sĩ khớp nhất
    best = None
    best_score = 0.0
    for d in doctors:
        sc = _score_name(text, d.get("name", ""))
        if sc > best_score:
            best, best_score = d, sc

    if not best or best_score < 0.4:
        return "Không tìm thấy bác sĩ phù hợp. Bạn có thể cung cấp tên đầy đủ không?", state

    # lưu vào state để flow đặt lịch dùng
    doc_id = best.get("userId") or best.get("id") or best.get("doctorId")
    state["selected_doctor"] = best
    state["doctor_id"] = doc_id

    name = best.get("name") or "—"
    email = best.get("email") or "—"
    phone = best.get("phoneNumber") or "—"
    arch = best.get("architecture") or "—"

    reply = (
        f"👨‍⚕️ {name} \n"
        f"📧 {email}\n"
        f"📞 {phone}\n"
        f"🏆 {arch}\n\n"
        f"Bạn muốn xem lịch trống của bác sĩ này không? Gõ: 'đặt lịch'."
    )
    return reply, state
