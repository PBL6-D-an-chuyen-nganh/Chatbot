import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
import threading
import logging
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import google.generativeai as genai
from sentence_transformers import SentenceTransformer

from config import *
from errors import ChatbotError, ResourceNotReadyError, ImageProcessingError, AppointmentError
from utils import fetch_and_cache, build_or_load_faiss
from intents.disease_intent import handle_disease_intent
from intents.doctor_intent import handle_doctor_intent
from intents.appointment import (
    start_appointment_booking,
    handle_time_selection,
    handle_patient_info,
    handle_view_appointments,
    handle_cancel_appointment,
)
from intents.intent_core import (
    analyze_intent,
    DISEASE_INTENTS,
    _resolve_entity_with_image_context,
)

from intents.symptom_checker import handle_symptom_inquiry

# ====== Env & GenAI ======
here = os.path.dirname(__file__)
load_dotenv(dotenv_path=os.path.join(here, ".env"))

# làm mới dữ liệu định kỳ
DATA_TTL_SECONDS = int(os.getenv("DATA_TTL_SECONDS", str(60 * 60)))  # 1 giờ
FORCE_RELOAD_ON_START = str(os.getenv("FORCE_RELOAD_ON_START", "true")).lower() in (
    "1",
    "true",
    "yes",
)

DISABLE_GEMINI = str(os.getenv("DISABLE_GEMINI", "")).lower() in ("1", "true", "yes")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

if not GEMINI_API_KEY and not DISABLE_GEMINI:
    raise RuntimeError(
        "Thiếu GEMINI_API_KEY. Tạo CHATBOT3/.env từ .env.example và điền key, "
        "hoặc set DISABLE_GEMINI=true cho local dev."
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

message_history = defaultdict(list)
MAX_MESSAGES_PER_MINUTE = 20
MAX_MESSAGE_LENGTH = 500

def check_rate_limit(session_id: str) -> bool:
    """Kiểm tra user không spam message"""
    now = datetime.now()
    cutoff = now - timedelta(minutes=1)
    # Xoá message cũ hơn 1 phút
    message_history[session_id] = [
        ts for ts in message_history[session_id] if ts > cutoff
    ]
    if len(message_history[session_id]) >= MAX_MESSAGES_PER_MINUTE:
        return False
    message_history[session_id].append(now)
    return True


def ensure_resources(force: bool = False):
    """
    Đảm bảo SentenceTransformer, dữ liệu articles/doctors, FAISS index sẵn sàng.
    - force=True: luôn reload dữ liệu + rebuild chỉ mục.
    - TTL: quá DATA_TTL_SECONDS thì tự reload.
    """
    global st_model, articles, doctors, index, titles, last_loaded_at

    now = time.time()
    need_reload = force or (now - last_loaded_at >= DATA_TTL_SECONDS)

    # nếu đã có model + index và chưa hết TTL thì thôi
    if st_model is not None and index is not None and not need_reload:
        return

    with resources_lock:
        now = time.time()
        need_reload = force or (now - last_loaded_at >= DATA_TTL_SECONDS)
        if st_model is not None and index is not None and not need_reload:
            return

        print("🔧 [ensure_resources] Initializing/refreshing resources ...")
        print(
            f"    force={force}, TTL={DATA_TTL_SECONDS}s, "
            f"last_loaded_at={last_loaded_at}, now={now}"
        )

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
            index, titles = build_or_load_faiss(
                articles, st_model, FAISS_PATH, TITLES_PATH
            )
            if titles is None:
                titles = []
            print("    ✅ FAISS index ready. Titles:", len(titles))
        except Exception as e:
            print("⚠️ Lỗi build/load FAISS:", e)
            index, titles = None, []

        last_loaded_at = time.time()
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_loaded_at))
        print(
            f"✅ [ensure_resources] Done at {ts}. "
            f"Next auto-refresh in {DATA_TTL_SECONDS}s."
        )


app = FastAPI(title="Healthcare Chatbot API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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
    segment: bool = Form(False),
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
            class_path=os.path.join(here, "models", "resnet101.pth"),
            seg_path=os.path.join(here, "models", "unet_segmenter.pth")
            if segment
            else None,
            labels_path=os.path.join(here, "models", "labels.json"),
        )
        if session_id:
            st = session_states.setdefault(session_id, {})
            st["last_image_diagnosis"] = {"predictions": preds}

            # 👉 map label ảnh (BCC, MEL, NV, ...) sang tên bệnh trong articles
            try:
                if preds:
                    top_label = preds[0].get("label")
                    if top_label:
                        label_str = str(top_label).strip()
                        # Nếu label đã là tên dài (Ví dụ model trả luôn 'Ung thư biểu mô tế bào đáy')
                        if len(label_str) > 3:
                            st["disease"] = label_str
                        else:
                            # Label dạng code ngắn: BCC, AKIEC, MEL, NV, BKL, DF, VASC...
                            code = label_str.upper()
                            best_article = None
                            for a in articles or []:
                                title = a.get("title", "") or ""
                                title_up = title.upper()
                                # match code trong ngoặc: (BCC - ..., (MEL - ...
                                if f"({code} " in title_up or f"({code}-" in title_up or f"({code})" in title_up:
                                    best_article = a
                                    break
                            if best_article:
                                st["disease"] = best_article.get("title")
                                print(f"[diagnose-image] mapped label '{code}' -> disease '{st['disease']}'")
                            else:
                                print(f"[diagnose-image] could not map label '{code}' to article title")
            except Exception as e2:
                print("⚠️ diagnose-image: error while mapping label to disease:", e2)
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
    global gemini_cooldown_until  # vì ta sẽ cập nhật biến global này

    sid = (req.session_id or "default").strip()
    msg = (req.message or "").strip()
    used_image_ctx = False
    if not msg:
        raise HTTPException(status_code=400, detail="message rỗng")
    
    if len(msg) > MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=400, detail=f"message quá dài (tối đa {MAX_MESSAGE_LENGTH} ký tự)")
    
    if not check_rate_limit(sid):
        raise HTTPException(status_code=429, detail="Quá nhiều tin nhắn trong một phút. Vui lòng thử lại sau.")

    # Lấy creator_id từ header hoặc body
    creator_id = request.headers.get("X-User-Id") or (req.creator_id or "")
    if creator_id:
        print(f"[chat] sid={sid} | creator_id={creator_id}")

    state = session_states.get(sid, {})
    if creator_id:
        state["creator_id"] = str(creator_id)
        
    auth_header = request.headers.get("Authorization")
    if auth_header:
        state["auth_token"] = auth_header
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

    # 1) intent ban đầu (qua Gemini + fallback)
    intent, entity, gemini_cooldown_until = analyze_intent(
        msg,
        gen_model=gen_model,
        disable_gemini=DISABLE_GEMINI,
        gemini_cooldown_until=gemini_cooldown_until,
    )
    
    lower = msg.lower()
    booking_triggers = (
        "đặt lịch",
        "book lịch",
        "lịch trống",
        "xem lịch",
        "đặt khám",
        "hẹn khám",
    )

    # 3) dùng ngữ cảnh ảnh cho nhóm bệnh
    intent, entity, used_image_ctx = _resolve_entity_with_image_context(
        msg, intent, entity, state
    )

    # 4) điều phối
    if not intent:
        reply = "Xin lỗi, tôi chưa hiểu ý bạn nói."
    elif intent in DISEASE_INTENTS:
        if st_model is None or index is None or articles is None:
            print(
                "⚠️ Disease intent requested but resources not ready "
                f"(st_model={st_model is not None}, "
                f"index={index is not None}, "
                f"articles={articles is not None})"
            )
            reply = (
                "Xin lỗi, hệ thống tra cứu bệnh hiện chưa sẵn sàng. "
                "Bạn vui lòng thử lại sau hoặc hỏi tôi về đặt lịch khám với bác sĩ nhé."
            )
        else:
            if intent == "mo_ta_trieu_chung":
                reply, state = handle_symptom_inquiry(
                    msg, st_model, index, articles, state
                )
            else:
                reply, state = handle_disease_intent(
                    intent, entity, msg, st_model, index, articles, state
            )
    elif intent == "hoi_thong_tin_bac_si":
        reply, state = handle_doctor_intent(entity, doctors, state)
        # nếu người dùng có chữ "đặt lịch" trong câu hỏi bác sĩ → nhảy vào flow đặt lịch luôn
        if any(k in lower for k in booking_triggers):
            reply, state = start_appointment_booking(
                state,
                SLOTS_API=SLOTS_API,
                doctors=doctors,
                user_text=msg,
            )
    elif intent == "dat_lich":
        state["user_input"] = msg
        reply, state = start_appointment_booking(
            state,
            SLOTS_API=SLOTS_API,
            doctors=doctors,
            user_text=msg,
        )
    elif intent == "xem_lich_da_dat":
        reply, state = handle_view_appointments(state)
    elif intent == "huy_lich":
        reply, state = handle_cancel_appointment(msg, state)
    else:
        reply = "Tôi chưa hiểu yêu cầu của bạn."

    if used_image_ctx:
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
