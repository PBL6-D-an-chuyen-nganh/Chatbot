# CHATBOT3/main.py
import os, re, json
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ====== import các module sẵn có của bạn ======
from config import *  # ARTICLES_API, ARTICLES_PATH, DOCTORS_API, DOCTORS_PATH, FAISS_PATH, TITLES_PATH, SLOTS_API, APPOINTMENTS_API
from utils import fetch_and_cache, build_or_load_faiss
from intents.disease_intent import handle_disease_intent
from intents.doctor_intent import handle_doctor_intent
from intents.appointment_intent import start_appointment_booking, handle_time_selection, handle_patient_info

# ====== Env & GenAI ======
load_dotenv()  # đọc file .env nếu có
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

if not GEMINI_API_KEY:
    raise RuntimeError("Thiếu GEMINI_API_KEY. Tạo CHATBOT3/.env từ .env.example và điền key.")

genai.configure(api_key=GEMINI_API_KEY)
gen_model = genai.GenerativeModel(GEMINI_MODEL)

# ====== Khởi tạo tài nguyên dùng chung (load một lần) ======
print("🔧 Đang tải SentenceTransformer & FAISS index...")
st_model = SentenceTransformer("all-MiniLM-L6-v2")
articles = fetch_and_cache(ARTICLES_API, ARTICLES_PATH)
doctors = fetch_and_cache(DOCTORS_API, DOCTORS_PATH)
index, titles = build_or_load_faiss(articles, st_model, FAISS_PATH, TITLES_PATH)
print("✅ Tải xong. Chatbot API sẵn sàng.")

app = FastAPI(title="Healthcare Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

session_states: Dict[str, Dict[str, Any]] = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    state: Optional[Dict[str, Any]] = None

class ResetRequest(BaseModel):
    session_id: str

def analyze_intent(user_input: str):
    prompt = f"""
Bạn là chatbot y tế. Phân tích câu nói sau và trả về JSON:
{{"intent": "intent_name", "entity": "nội dung"}}
Intent hợp lệ: hoi_trieu_chung, hoi_cach_dieu_tri, hoi_nguyen_nhan, hoi_phong_ngua, hoi_thong_tin_bac_si, hoi_lich_trong, dat_lich
Câu nói: "{user_input}"
"""
    try:
        resp = gen_model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data.get("intent"), data.get("entity")
    except Exception as e:
        print("⚠️ Lỗi intent:", e)
    return None, None

@app.get("/api/health")
def health():
    return {"ok": True}

@app.post("/api/reset")
def reset(req: ResetRequest):
    session_states.pop(req.session_id, None)
    return {"ok": True}

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sid = (req.session_id or "default").strip()
    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message rỗng")

    state = session_states.get(sid, {})

    if state.get("step") == "cho_chon_gio":
        reply, state = handle_time_selection(msg, state)
    elif state.get("step") == "nhap_thong_tin":
        reply, state = handle_patient_info(msg, state, APPOINTMENTS_API)
    else:
        intent, entity = analyze_intent(msg)
        if not intent:
            reply = "Xin lỗi, tôi chưa hiểu ý bạn nói."
        elif intent in ["hoi_trieu_chung","hoi_cach_dieu_tri","hoi_nguyen_nhan","hoi_phong_ngua"]:
            reply, state = handle_disease_intent(intent, entity, msg, st_model, index, articles, state)
        elif intent == "hoi_thong_tin_bac_si":
            reply, state = handle_doctor_intent(entity, doctors, st_model, state)
        elif intent == "dat_lich":
            state["user_input"] = msg
            reply, state = start_appointment_booking(state, SLOTS_API)
        else:
            reply = "Tôi chưa hiểu yêu cầu của bạn."

    session_states[sid] = state
    return ChatResponse(reply=reply, state=state)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8081, reload=True)
