# CHATBOT3/main.py
import os, re, json, time, threading, asyncio
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
here = os.path.dirname(__file__)
# Load .env next to this file to avoid cwd issues
load_dotenv(dotenv_path=os.path.join(here, ".env"))

# Dev option to disable calling Gemini (for local dev / when quota exhausted)
DISABLE_GEMINI = str(os.getenv("DISABLE_GEMINI", "")).lower() in ("1", "true", "yes")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

if not GEMINI_API_KEY and not DISABLE_GEMINI:
    raise RuntimeError("Thiếu GEMINI_API_KEY. Tạo CHATBOT3/.env từ .env.example và điền key, or set DISABLE_GEMINI=true for local dev.")

if not DISABLE_GEMINI:
    genai.configure(api_key=GEMINI_API_KEY)
    gen_model = genai.GenerativeModel(GEMINI_MODEL)
else:
    gen_model = None

# Globals for heavy resources (lazy-loaded)
st_model = None
articles = None
doctors = None
index = None
titles = None

# Lock protecting swaps
resources_lock = threading.Lock()

# Gemini backoff/cooldown timestamp
gemini_cooldown_until = 0.0

def ensure_resources(force: bool = False):
    """Ensure sentence-transformer, articles/doctors and FAISS index are available.
    This is safe to call from multiple threads; the first caller will initialize resources.
    If force is True, it will re-fetch and rebuild index.
    """
    global st_model, articles, doctors, index, titles
    # quick check
    if st_model is not None and index is not None and not force:
        return

    with resources_lock:
        # double-check after acquiring lock
        if st_model is None or index is None or force:
            print("🔧 Initializing/refreshing heavy resources (SentenceTransformer, FAISS)...")
            # load model
            try:
                if st_model is None:
                    st_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print("⚠️ Không tải được SentenceTransformer:", e)
                raise

            # fetch data and normalize if function available
            try:
                raw_articles = fetch_and_cache(ARTICLES_API, ARTICLES_PATH)
                articles = raw_articles
            except Exception as e:
                print("⚠️ Lỗi tải articles:", e)
                articles = []

            try:
                doctors = fetch_and_cache(DOCTORS_API, DOCTORS_PATH)
            except Exception as e:
                print("⚠️ Lỗi tải doctors:", e)
                doctors = []

            try:
                index, titles = build_or_load_faiss(articles, st_model, FAISS_PATH, TITLES_PATH)
            except Exception as e:
                print("⚠️ Lỗi build/load FAISS:", e)
                index, titles = None, []

            print("✅ Heavy resources ready.")

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

def _fallback_intent_guess(user_input: str):
    """
    Simple rule-based fallback intent guess used when the LLM is unavailable or returns
    something unparsable. Returns (intent, entity) or (None, None).
    """
    text = (user_input or "").lower()
    patterns = [
        (r"(triệu chứng|biểu hiện)", "hoi_trieu_chung"),
        (r"(điều trị|chữa|thuốc)", "hoi_cach_dieu_tri"),
        (r"(nguyên nhân|tại sao)", "hoi_nguyen_nhan"),
        (r"(phòng ngừa|phòng tránh|tiêm|vaccine)", "hoi_phong_ngua"),
        (r"(bác sĩ|doctor|liên hệ bác sĩ)", "hoi_thong_tin_bac_si"),
        (r"(đặt lịch|đăng ký khám|book lịch|lịch trống|slot)", "dat_lich"),
    ]
    for pat, intent in patterns:
        if re.search(pat, text):
            return intent, user_input
    return None, None

def analyze_intent(user_input: str):
    prompt = f"""
Bạn là chatbot y tế. Phân tích câu nói sau và trả về JSON:
{{"intent": "intent_name", "entity": "nội dung"}}
Intent hợp lệ: hoi_trieu_chung, hoi_cach_dieu_tri, hoi_nguyen_nhan, hoi_phong_ngua, hoi_thong_tin_bac_si, hoi_lich_trong, dat_lich
Câu nói: "{user_input}"
"""
    # If Gemini disabled or in cooldown, use fallback
    global gemini_cooldown_until
    now = time.time()
    if DISABLE_GEMINI or (gemini_cooldown_until and now < gemini_cooldown_until):
        if gemini_cooldown_until and now < gemini_cooldown_until:
            print(f"⚠️ Gemini cooldown active, skipping call until {gemini_cooldown_until} (now {now})")
        return _fallback_intent_guess(user_input)

    try:
        if gen_model is None:
            return _fallback_intent_guess(user_input)

        resp = gen_model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data.get("intent"), data.get("entity")
    except Exception as e:
        err_text = str(e)
        print("⚠️ Lỗi intent (Gemini):", err_text)
        # try to extract retry seconds from message (best-effort)
        m = re.search(r"retry_delay.*seconds\s*:\s*(\d+)", err_text)
        if not m:
            m = re.search(r"seconds\s*:\s*(\d+)", err_text)
        try:
            retry_secs = int(m.group(1)) if m else 60
        except Exception:
            retry_secs = 60
        gemini_cooldown_until = time.time() + retry_secs
        print(f"⚠️ Setting Gemini cooldown for {retry_secs}s until {gemini_cooldown_until}")

    return _fallback_intent_guess(user_input)

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

    # ensure heavy resources are available (non-blocking for fast startup; will initialize on first request)
    try:
        # initialize lazily (may block on first request)
        ensure_resources()
    except Exception as e:
        print("⚠️ ensure_resources failed:", e)

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
