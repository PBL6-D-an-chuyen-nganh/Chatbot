from intents.intent_utils import get_available_slots, create_appointment
from datetime import datetime, timedelta
import re
import locale

# Thiết lập locale tiếng Việt (nếu có)
try:
    locale.setlocale(locale.LC_TIME, "vi_VN")
except:
    pass


def extract_requested_date(user_input):
    """
    Phân tích ngày mà người dùng yêu cầu (trả về yyyy-mm-dd hoặc None)
    """
    today = datetime.now()
    text = user_input.lower().strip()

    # 1️⃣ Các từ khóa đơn giản
    if "hôm nay" in text:
        return today.strftime("%Y-%m-%d")
    elif "ngày mai" in text:
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "ngày kia" in text:
        return (today + timedelta(days=2)).strftime("%Y-%m-%d")

    # 2️⃣ Dạng dd/mm/yyyy
    match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if match:
        d, m, y = map(int, match.groups())
        return datetime(y, m, d).strftime("%Y-%m-%d")

    # 3️⃣ Dạng dd/mm (tự hiểu là năm hiện tại)
    match = re.search(r"(\d{1,2})/(\d{1,2})", text)
    if match:
        d, m = map(int, match.groups())
        y = today.year
        return datetime(y, m, d).strftime("%Y-%m-%d")

    # 4️⃣ Nếu nói “thứ 2”, “thứ 3”, ...
    weekdays = {
        "thứ 2": 0, "thứ hai": 0,
        "thứ 3": 1, "thứ ba": 1,
        "thứ 4": 2, "thứ tư": 2,
        "thứ 5": 3, "thứ năm": 3,
        "thứ 6": 4, "thứ sáu": 4,
        "thứ 7": 5, "thứ bảy": 5,
        "chủ nhật": 6
    }
    for k, v in weekdays.items():
        if k in text:
            diff = (v - today.weekday() + 7) % 7
            diff = diff if diff != 0 else 7
            return (today + timedelta(days=diff)).strftime("%Y-%m-%d")

    return None


def start_appointment_booking(state, slots_api):
    """
    Bắt đầu quy trình đặt lịch, hiển thị các khung giờ trống
    """
    doctor = state.get("doctor")
    if not doctor:
        return "Bạn muốn đặt lịch với bác sĩ nào ạ?", state

    user_input = state.get("user_input", "")
    requested_date = extract_requested_date(user_input)

    doctor_id = doctor.get("id") or doctor.get("userId")
    slots_data = get_available_slots(doctor_id, slots_api)
    if not slots_data:
        return f"Hiện tại bác sĩ {doctor['name']} chưa có khung giờ trống nào.", state

    # 🔹 Lọc lịch nếu có ngày cụ thể
    if requested_date:
        slots_data = [s for s in slots_data if s.startswith(requested_date)]
        if not slots_data:
            return f"❌ Bác sĩ {doctor['name']} không có lịch trống vào ngày {requested_date}.", state

    # Gom nhóm theo ngày
    grouped = {}
    for s in slots_data:
        date_str, time_str = s.split("T")
        grouped.setdefault(date_str, []).append(time_str)

    text_lines = []
    for d, times in grouped.items():
        text_lines.append(f"\n📅 {d}:")
        text_lines += [f"  🕒 {t}" for t in times]

    state["step"] = "cho_chon_gio"
    state["slots"] = [s.strip() for s in slots_data]
    state["doctor_id"] = doctor_id

    return "\n".join(text_lines) + "\n\nVui lòng chọn một khung giờ (yyyy-mm-ddTHH:MM:SS):", state


def handle_time_selection(user_input, state):
    chosen = user_input.strip()
    slots = [s.strip() for s in state.get("slots", [])]
    if chosen not in slots:
        return "Giờ này không hợp lệ, vui lòng chọn đúng giờ có trong danh sách.", state

    state["chosen_time"] = chosen
    state["step"] = "nhap_thong_tin"
    return (
        "Vui lòng cung cấp thông tin bệnh nhân (cách nhau bằng dấu phẩy):\n"
        "- Họ tên,\n- Email,\n- Số điện thoại,\n- Giới tính (Nam/Nữ),\n- Ngày sinh (yyyy-mm-dd)."
    ), state


def handle_patient_info(user_input, state, appointments_api):
    parts = [p.strip() for p in user_input.split(",")]
    if len(parts) < 5:
        return "Vui lòng nhập đủ 5 thông tin, cách nhau bằng dấu phẩy.", state

    name, email, phone, gender, dob = parts
    data = {
        "patientInfo": {
            "name": name,
            "email": email,
            "phoneNumber": phone,
            "gender": gender,
            "dateOfBirth": dob
        },
        "time": state["chosen_time"],
        "note": f"Khám {state.get('disease','tổng quát')}",
        "doctorId": state["doctor_id"],
        "creatorId": "30"
    }

    if create_appointment(data, appointments_api):
        reply = f"✅ Đặt lịch thành công cho {name} vào {data['time']}!"
        state.clear()
        return reply, state
    else:
        return "❌ Đặt lịch thất bại, vui lòng thử lại.", state
