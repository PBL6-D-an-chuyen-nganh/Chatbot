from intents.intent_utils import find_doctor_by_name

def handle_doctor_intent(entity, doctors, model, state):
    doctor = find_doctor_by_name(entity, doctors, model)
    if not doctor:
        return "Không tìm thấy bác sĩ đó.", state
    state["doctor"] = doctor
    return (
        f"👨‍⚕️ {doctor['name']} ({doctor.get('degree','')}, {doctor.get('position','')})\n"
        f"📧 {doctor.get('email','Không có')}\n"
        f"📞 {doctor.get('phoneNumber','Không có')}\n"
        f"🏆 {doctor.get('achievements','Chưa cập nhật')}", state
    )
