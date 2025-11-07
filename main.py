from config import *
from utils import fetch_and_cache, build_or_load_faiss
from intents.disease_intent import handle_disease_intent
from intents.doctor_intent import handle_doctor_intent
from intents.appointment_intent import start_appointment_booking, handle_time_selection, handle_patient_info
from sentence_transformers import SentenceTransformer
import re
import google.generativeai as genai
import json

# ======================= Cấu hình GenAI
genai.configure(api_key="AIzaSyCM6ydxee5VmjO0XvtK3YuVOC7J6DTP1Lg")

# ======================= Biến toàn cục
state = {}
print("🤖 Chatbot y tế sẵn sàng!")

# ======================= Load dữ liệu
model = SentenceTransformer("all-MiniLM-L6-v2")
articles = fetch_and_cache(ARTICLES_API, ARTICLES_PATH)
doctors = fetch_and_cache(DOCTORS_API, DOCTORS_PATH)
index, titles = build_or_load_faiss(articles, model, FAISS_PATH, TITLES_PATH)

# ======================= Hàm phân tích intent
def analyze_intent(user_input):
    prompt = f"""
Bạn là chatbot y tế. Phân tích câu nói sau và trả về JSON:
{{"intent": "intent_name", "entity": "nội dung"}}
Intent hợp lệ: hoi_trieu_chung, hoi_cach_dieu_tri, hoi_nguyen_nhan, hoi_phong_ngua, hoi_thong_tin_bac_si, hoi_lich_trong, dat_lich
Câu nói: "{user_input}"
"""
    try:
        model_gen = genai.GenerativeModel('gemini-2.5-pro')
        resp = model_gen.generate_content(prompt)
        match = re.search(r'\{.*\}', resp.text, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
            return result.get("intent"), result.get("entity")
    except Exception as e:
        print("⚠️ Lỗi intent:", e)
    return None, None


# ======================= Vòng lặp chính
while True:
    user_input = input("👤 Bạn: ")
    if user_input.lower() == "exit":
        print("💬 Chatbot: Tạm biệt!")
        break

    # FSM - Quản lý trạng thái
    if state.get("step") == "cho_chon_gio":
        reply, state = handle_time_selection(user_input, state)
    elif state.get("step") == "nhap_thong_tin":
        reply, state = handle_patient_info(user_input, state, APPOINTMENTS_API)
    else:
        intent, entity = analyze_intent(user_input)
        if not intent:
            reply = "Xin lỗi, tôi chưa hiểu ý bạn nói."
        elif intent in ["hoi_trieu_chung","hoi_cach_dieu_tri","hoi_nguyen_nhan","hoi_phong_ngua"]:
            reply, state = handle_disease_intent(intent, entity, user_input, model, index, articles, state)
        elif intent == "hoi_thong_tin_bac_si":
            reply, state = handle_doctor_intent(entity, doctors, model, state)
        elif intent == "dat_lich":
            state["user_input"] = user_input  # Lưu để extract ngày
            reply, state = start_appointment_booking(state, SLOTS_API)
        else:
            reply = "Tôi chưa hiểu yêu cầu của bạn."

    print("💬 Chatbot:", reply)
