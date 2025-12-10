# CHATBOT3/main.py
import os, re, json, time, threading
from typing import Dict, Any, Optional, Tuple

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ====== import các module của bạn ======
# ARTICLES_API, ARTICLES_PATH, DOCTORS_API, DOCTORS_PATH, FAISS_PATH, TITLES_PATH, SLOTS_API, APPOINTMENTS_API
from config import *
from utils import fetch_and_cache, build_or_load_faiss
from intents.disease_intent import handle_disease_intent
from intents.doctor_intent import handle_doctor_intent
from intents.appointment_intent import (
    start_appointment_booking,
    handle_time_selection,
    handle_patient_info,
)

# ====== Env & GenAI ======
here = os.path.dirname(__file__)
load_dotenv(dotenv_path=os.path.join(here, ".env"))

# làm mới dữ liệu định kỳ
DATA_TTL_SECONDS = int(os.getenv("DATA_TTL_SECONDS", str(60 * 60)))  # 1 giờ
FORCE_RELOAD_ON_START = str(os.getenv("FORCE_RELOAD_ON_START", "true")).lower() in ("1", "true", "yes")

DISABLE_GEMINI = str(os.getenv("DISABLE_GEMINI", "")).lower() in ("1", "true", "yes")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

if not GEMINI_API_KEY and not DISABLE_GEMINI:
    raise RuntimeError(
        "Thiếu GEMINI_API_KEY. Tạo CHATBOT3/.env từ .env.example và điền key, hoặc set DISABLE_GEMINI=true cho local dev."
    )

if not DISABLE_GEMINI:
    genai.configure(api_key=GEMINI_API_KEY)
    gen_model = genai.GenerativeModel(GEMINI_MODEL)
else:
    gen_model = None

last_loaded_at = 0.0

# Globals (lazy)
st_model = None
articles = None
doctors = None
index = None
titles = None

resources_lock = threading.Lock()
gemini_cooldown_until = 0.0

def _unlink_if_exists(path):
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
            print(f"🧹 Removed cache file: {path}")
    except Exception as e:
        print(f"⚠️ Could not remove cache file {path}: {e}")

def ensure_resources(force: bool = False):
    """
    Đảm bảo SentenceTransformer, dữ liệu articles/doctors, FAISS index sẵn sàng.
    - force=True: luôn reload dữ liệu + rebuild chỉ mục.
    - TTL: quá DATA_TTL_SECONDS thì tự reload.
    """
    global st_model, articles, doctors, index, titles, last_loaded_at

    now = time.time()
    need_reload = force or (now - last_loaded_at >= DATA_TTL_SECONDS)

    if st_model is not None and index is not None and not need_reload:
        return

    with resources_lock:
        now = time.time()
        need_reload = force or (now - last_loaded_at >= DATA_TTL_SECONDS)
        if st_model is not None and index is not None and not need_reload:
            return

        print("🔧 [ensure_resources] Initializing/refreshing resources ...")
        print(f"    force={force}, TTL={DATA_TTL_SECONDS}s, last_loaded_at={last_loaded_at}, now={now}")

        # 1) ST model
        if st_model is None:
            try:
                st_model = SentenceTransformer("all-MiniLM-L6-v2")
                print("    ✅ SentenceTransformer loaded.")
            except Exception as e:
                print("⚠️ Không tải được SentenceTransformer:", e)
                raise

        # 2) Xoá cache local → fetch mới để luôn thấy dữ liệu mới nhất
        try:
            _unlink_if_exists(ARTICLES_PATH)
            _unlink_if_exists(DOCTORS_PATH)
            raw_articles = fetch_and_cache(ARTICLES_API, ARTICLES_PATH)
            articles = raw_articles or []
            print(f"    ✅ Articles loaded: {len(articles)} items.")
        except Exception as e:
            print("⚠️ Lỗi tải articles:", e)
            articles = []

        try:
            doctors = fetch_and_cache(DOCTORS_API, DOCTORS_PATH)
            print(f"    ✅ Doctors loaded: {len(doctors)} items.")
        except Exception as e:
            print("⚠️ Lỗi tải doctors:", e)
            doctors = []

        # 3) FAISS
        try:
            index, titles = build_or_load_faiss(articles, st_model, FAISS_PATH, TITLES_PATH)
            if titles is None: titles = []
            print("    ✅ FAISS index ready. Titles:", len(titles))
        except Exception as e:
            print("⚠️ Lỗi build/load FAISS:", e)
            index, titles = None, []

        last_loaded_at = time.time()
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_loaded_at))
        print(f"✅ [ensure_resources] Done at {ts}. Next auto-refresh in {DATA_TTL_SECONDS}s.")

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
    creator_id: Optional[str] = None  # đưa từ FE

class ChatResponse(BaseModel):
    reply: str
    state: Optional[Dict[str, Any]] = None

class ResetRequest(BaseModel):
    session_id: str

# -------- Intent helpers --------
DISEASE_INTENTS = {"hoi_trieu_chung", "hoi_cach_dieu_tri", "hoi_nguyen_nhan", "hoi_phong_ngua"}

def _fallback_intent_guess(user_input: str) -> Tuple[Optional[str], Optional[str]]:
    text = (user_input or "").lower()
    patterns = [
        (r"(triệu chứng|biểu hiện)", "hoi_trieu_chung"),
        (r"(điều trị|chữa|thuốc)", "hoi_cach_dieu_tri"),
        (r"(nguyên nhân|tại sao)", "hoi_nguyen_nhan"),
        (r"(phòng ngừa|phòng tránh|tiêm|vaccine)", "hoi_phong_ngua"),
        (r"(bác sĩ|doctor|liên hệ bác sĩ|thông tin bác sĩ)", "hoi_thong_tin_bac_si"),
        (r"(đặt lịch|đăng ký khám|book lịch|lịch trống|xem lịch|đặt khám|hẹn khám)", "dat_lich"),
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
    global gemini_cooldown_until
    now = time.time()
    if DISABLE_GEMINI or (gemini_cooldown_until and now < gemini_cooldown_until):
        if gemini_cooldown_until and now < gemini_cooldown_until:
            print(f"⚠️ Gemini cooldown active, skipping call until {gemini_cooldown_until} (now {now})")
        return _fallback_intent_guess(user_input)

    try:
        if gen_model is None:
            return _fallback_intent_guess(user_input)
        print("🧠 [analyze_intent] Prompt sent to Gemini.")
        resp = gen_model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data.get("intent"), data.get("entity")
    except Exception as e:
        err_text = str(e)
        print("⚠️ Lỗi intent (Gemini):", err_text)
        m = re.search(r"retry_delay.*seconds\s*:\s*(\d+)", err_text) or re.search(r"seconds\s*:\s*(\d+)", err_text)
        try:
            retry_secs = int(m.group(1)) if m else 60
        except Exception:
            retry_secs = 60
        gemini_cooldown_until = time.time() + retry_secs
        print(f"⚠️ Setting Gemini cooldown for {retry_secs}s until {gemini_cooldown_until}")

    return _fallback_intent_guess(user_input)

def _top_predicted_label_from_state(state: Dict[str, Any]) -> Optional[str]:
    try:
        diag = state.get("last_image_diagnosis") or {}
        preds = diag.get("predictions") or []
        return preds[0].get("label") if preds else None
    except Exception:
        return None

_GENERIC_REF_PHRASES = {
    "bệnh này", "bệnh đó", "bệnh kia", "loại này", "nó", "cái này",
    "triệu chứng", "điều trị", "nguyên nhân", "phòng ngừa", "phòng tránh",
}
def _is_generic_entity(entity: Optional[str]) -> bool:
    if not entity:
        return True
    norm = (entity or "").strip().lower()
    return norm in _GENERIC_REF_PHRASES or len(norm) <= 2

def _resolve_entity_with_image_context(
    user_input: str,
    intent: Optional[str],
    entity: Optional[str],
    state: Dict[str, Any]
) -> Tuple[Optional[str], Optional[str], bool]:
    if intent in DISEASE_INTENTS and _is_generic_entity(entity):
        top_label = _top_predicted_label_from_state(state)
        if top_label:
            return intent, top_label, True

    text = (user_input or "").lower()
    if not intent and state.get("last_image_diagnosis"):
        if re.search(r"(triệu chứng|điều trị|nguyên nhân|phòng ngừa|phòng tránh|thuốc)", text):
            top_label = _top_predicted_label_from_state(state)
            if top_label:
                if re.search(r"triệu chứng|biểu hiện", text):
                    return "hoi_trieu_chung", top_label, True
                if re.search(r"điều trị|thuốc|chữa", text):
                    return "hoi_cach_dieu_tri", top_label, True
                if re.search(r"nguyên nhân|tại sao", text):
                    return "hoi_nguyen_nhan", top_label, True
                if re.search(r"phòng ngừa|phòng tránh|tiêm|vaccine", text):
                    return "hoi_phong_ngua", top_label, True
    return intent, entity, False

# -------- API --------
@app.post("/api/admin/refresh")
def admin_refresh():
    try:
        ensure_resources(force=True)
        return {"ok": True, "refreshed_at": time.time()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/diagnose-image")
async def diagnose_image(
    image: UploadFile = File(...),
    session_id: str = Form(None),
    top_k: int = Form(3),
    segment: bool = Form(False)
):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là ảnh.")
    data = await image.read()
    MAX_MB = int(os.getenv("IMAGE_MAX_MB", "5"))
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File quá lớn.")
    try:
        ensure_resources()
        from image_diag import predict_image
        preds = predict_image(
            data,
            top_k=top_k,
            class_path=os.path.join(here, 'models', 'resnet101.pth'),
            seg_path=os.path.join(here, 'models', 'unet_segmenter.pth') if segment else None,
            labels_path=os.path.join(here, 'models', 'labels.json'),
        )
        if session_id:
            session_states.setdefault(session_id, {})['last_image_diagnosis'] = {'predictions': preds}
        return {"predictions": preds}
    except Exception as e:
        print("⚠️ diagnose-image error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health():
    return {"ok": True}

@app.post("/api/reset")
def reset(req: ResetRequest):
    session_states.pop(req.session_id, None)
    return {"ok": True}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    sid = (req.session_id or "default").strip()
    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message rỗng")

    # Lấy creator_id từ header hoặc body
    creator_id = request.headers.get("X-User-Id") or (req.creator_id or "")
    if creator_id:
        print(f"[chat] sid={sid} | creator_id={creator_id}")

    state = session_states.get(sid, {})
    if creator_id:
        state["creator_id"] = str(creator_id)

    # init heavy resources (lazy)
    try:
        ensure_resources()
    except Exception as e:
        print("⚠️ ensure_resources failed:", e)

    # luồng đặt lịch (đang chờ người dùng chọn giờ / nhập thông tin)
    if state.get("step") == "cho_chon_gio":
        reply, state = handle_time_selection(msg, state)
        session_states[sid] = state
        return ChatResponse(reply=reply, state=state)

    if state.get("step") == "nhap_thong_tin":
        reply, state = handle_patient_info(msg, state, APPOINTMENTS_API)
        session_states[sid] = state
        return ChatResponse(reply=reply, state=state)

    # 1) intent ban đầu
    intent, entity = analyze_intent(msg)

    # 2) HARD ROUTING: nếu câu có ý đặt lịch thì chuyển thẳng sang flow đặt lịch
    lower = msg.lower()
    booking_triggers = ("đặt lịch", "book lịch", "lịch trống", "xem lịch", "đặt khám", "hẹn khám")
    if any(k in lower for k in booking_triggers):
        reply, state = start_appointment_booking(
            state,
            SLOTS_API=SLOTS_API,
            doctors=doctors,
            user_text=msg
        )
        session_states[sid] = state
        return ChatResponse(reply=reply, state=state)

    # 3) dùng ngữ cảnh ảnh cho nhóm bệnh
    intent, entity, used_image_ctx = _resolve_entity_with_image_context(msg, intent, entity, state)

    # 4) điều phối
    if not intent:
        reply = "Xin lỗi, tôi chưa hiểu ý bạn nói."
    elif intent in DISEASE_INTENTS:
        reply, state = handle_disease_intent(intent, entity, msg, st_model, index, articles, state)
    elif intent == "hoi_thong_tin_bac_si":
        reply, state = handle_doctor_intent(entity, doctors, st_model, state)
        # nếu người dùng có chữ "đặt lịch" trong câu hỏi bác sĩ → nhảy vào flow đặt lịch luôn
        if any(k in lower for k in booking_triggers):
            reply, state = start_appointment_booking(state, SLOTS_API=SLOTS_API, doctors=doctors, user_text=msg)
    elif intent == "dat_lich":
        state["user_input"] = msg
        reply, state = start_appointment_booking(state, SLOTS_API=SLOTS_API, doctors=doctors, user_text=msg)
    else:
        reply = "Tôi chưa hiểu yêu cầu của bạn."

    if 'used_image_ctx' in locals() and used_image_ctx:
        reply = f"(Theo chẩn đoán hình ảnh gần nhất: {entity})\n" + reply

    session_states[sid] = state
    return ChatResponse(reply=reply, state=state)

def _bg_auto_refresh_loop():
    while True:
        try:
            time.sleep(DATA_TTL_SECONDS)
            print("⏰ Auto-refresh: reloading resources...")
            ensure_resources(force=True)
        except Exception as e:
            print("⚠️ Auto-refresh error:", e)

@app.on_event("startup")
def _on_startup():
    print("[startup] ensure_resources(force=True)")
    ensure_resources(force=FORCE_RELOAD_ON_START)
    t = threading.Thread(target=_bg_auto_refresh_loop, daemon=True)
    t.start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8081, reload=True)
