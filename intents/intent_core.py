import re
import json
import time
from typing import Dict, Any, Optional, Tuple

DISEASE_INTENTS = {
    "hoi_trieu_chung",
    "hoi_cach_dieu_tri",
    "hoi_nguyen_nhan",
    "hoi_phong_ngua",
    "mo_ta_trieu_chung",
}

_GENERIC_REF_PHRASES = {
    "bệnh này", "bệnh đó", "bệnh kia", "loại này", "nó", "cái này",
    "triệu chứng", "điều trị", "nguyên nhân", "phòng ngừa", "phòng tránh",
}


def _fallback_intent_guess(user_input: str) -> Tuple[Optional[str], Optional[str]]:
    text = (user_input or "").lower()

    # 1) Các câu hỏi rõ ràng (triệu chứng / điều trị / nguyên nhân / phòng ngừa / bác sĩ / đặt lịch)
    patterns = [
        (r"(bị|tôi bị|em bị|mình bị|anh bị|chị bị).*(ngứa|nóng rát|sưng|chảy máu|nổi mẩn|khô|đỏ|vàng|trắng|mụn|mụn mủ|chàm|vảy|u|khối|cục)", "mo_ta_trieu_chung"),

        # 👉 lịch đã đặt / lịch sử
        (r"(lịch sử|lich su).*(đặt lịch|dat lich|lịch khám|lich kham)", "xem_lich_da_dat"),
        (r"(các|tat ca|toan bo)?\s*lịch khám tôi đã đặt", "xem_lich_da_dat"),
        (r"(triệu chứng|biểu hiện)", "hoi_trieu_chung"),
        (r"(điều trị|chữa|thuốc)", "hoi_cach_dieu_tri"),
        (r"(nguyên nhân|tại sao)", "hoi_nguyen_nhan"),
        (r"(phòng ngừa|phòng tránh|tiêm|vaccine)", "hoi_phong_ngua"),
        (r"(bác sĩ|doctor|liên hệ bác sĩ|thông tin bác sĩ)", "hoi_thong_tin_bac_si"),
        (r"(đặt lịch|đăng ký khám|book lịch|lịch trống|xem lịch|đặt khám|hẹn khám)", "dat_lich"),
        (r"(hủy|huỷ).*(lịch|cuộc hẹn)", "huy_lich"),
    ]
    for pat, intent in patterns:
        if re.search(pat, text):
            return intent, user_input

    # 2) Mô tả triệu chứng (fallback)
    #    - Có đại từ xưng hô (tôi/em/anh/chị/mình/con/bé/cháu)
    #    - Và có ít nhất 1 từ về triệu chứng da liễu
    pronoun_re = r"\b(tôi|em|anh|chị|mình|con|bé|cháu)\b"
    symptom_re = (
        r"(khô|ngua|ngứa|mẩn đỏ|mẩn|nổi mẩn|"
        r"mụn|mụn mủ|vảy|bong vảy|tróc|tróc vảy|"
        r"chảy dịch|rỉ dịch|sưng|rát|nóng rát|phồng rộp)"
    )

    if re.search(pronoun_re, text) and re.search(symptom_re, text):
        return "mo_ta_trieu_chung", user_input

    # 3) Trường hợp có chữ "bị ... triệu chứng" (giữ rule cũ cho chắc)
    if re.search(
        r"(bị|toi bi|em bi|minh bi|anh bi|chi bi).*(ngứa|ngua|nóng rát|nong rat|"
        r"sưng|sung|chảy máu|chay mau|nổi mẩn|noi man|khô|kho|đỏ|do|"
        r"vàng|vang|trắng|trang|mụn|mun|mụn mủ|mun mu|chàm|cham|vảy|vay|u|khối|khoi|cục|cuc)",
        text,
    ):
        return "mo_ta_trieu_chung", user_input

    return None, None


def analyze_intent(
    user_input: str,
    gen_model,
    disable_gemini: bool,
    gemini_cooldown_until: float,
) -> Tuple[Optional[str], Optional[str], float]:
    """
    Trả về (intent, entity, new_cooldown_until)
    """
    prompt = f"""
Bạn là chatbot y tế. Phân tích câu nói sau và trả về JSON:
{{"intent": "intent_name", "entity": "nội dung"}}
Intent hợp lệ: hoi_trieu_chung, hoi_cach_dieu_tri, hoi_nguyen_nhan, hoi_phong_ngua, hoi_thong_tin_bac_si, hoi_lich_trong, dat_lich, xem_lich_da_dat, huy_lich, mo_ta_trieu_chung,
Câu nói: "{user_input}"
"""
    now = time.time()

    if disable_gemini or (gemini_cooldown_until and now < gemini_cooldown_until):
        if gemini_cooldown_until and now < gemini_cooldown_until:
            print(
                f"⚠️ Gemini cooldown active, skipping call until "
                f"{gemini_cooldown_until} (now {now})"
            )
        intent, entity = _fallback_intent_guess(user_input)
        return intent, entity, gemini_cooldown_until

    try:
        if gen_model is None:
            intent, entity = _fallback_intent_guess(user_input)
            return intent, entity, gemini_cooldown_until

        print("🧠 [analyze_intent] Prompt sent to Gemini.")
        resp = gen_model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data.get("intent"), data.get("entity"), gemini_cooldown_until
    except Exception as e:
        err_text = str(e)
        print("⚠️ Lỗi intent (Gemini):", err_text)
        m = re.search(r"retry_delay.*seconds\s*:\s*(\d+)", err_text) or re.search(
            r"seconds\s*:\s*(\d+)", err_text
        )
        try:
            retry_secs = int(m.group(1)) if m else 60
        except Exception:
            retry_secs = 60

        new_cooldown = time.time() + retry_secs
        print(f"⚠️ Setting Gemini cooldown for {retry_secs}s until {new_cooldown}")
        gemini_cooldown_until = new_cooldown

    intent, entity = _fallback_intent_guess(user_input)
    return intent, entity, gemini_cooldown_until


def _top_predicted_label_from_state(state: Dict[str, Any]) -> Optional[str]:
    try:
        diag = state.get("last_image_diagnosis") or {}
        preds = diag.get("predictions") or []
        return preds[0].get("label") if preds else None
    except Exception:
        return None

def _last_disease_from_state(state: Dict[str, Any]) -> Optional[str]:
    name = state.get("disease")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None

def _is_generic_entity(entity: Optional[str]) -> bool:
    """
    Xem entity có chỉ là câu hỏi chung chung kiểu:
    'nguyên nhân do đâu', 'triệu chứng là gì', 'phòng ngừa ra sao'... không.
    Nếu chỉ toàn từ khóa chung, không có tên bệnh -> coi là generic.
    """
    if not entity:
        return True

    norm = (entity or "").strip().lower()
    if norm in _GENERIC_REF_PHRASES or len(norm) <= 2:
        return True

    # Nếu entity chỉ toàn mấy từ kiểu "nguyên nhân / triệu chứng / do đâu / là gì"
    tokens = re.findall(r"\w+", norm)
    generic_tokens = {
        "bệnh", "benh",
        "triệu", "trieu", "chứng", "chung",
        "nguyên", "nguyen", "nhân", "nhan",
        "điều", "dieu", "trị", "tri",
        "phòng", "phong", "ngừa", "ngua",
        "do", "đâu", "dau",
        "gì", "gi",
        "là", "la",
        "này", "nay", "đó", "do", "kia"
    }
    remaining = [t for t in tokens if t not in generic_tokens]
    # Nếu sau khi bỏ hết từ generic mà không còn gì -> entity chung chung
    return len(remaining) == 0


def _last_disease_from_state(state: Dict[str, Any]) -> Optional[str]:
    name = state.get("disease")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def _resolve_entity_with_image_context(
    user_input: str,
    intent: Optional[str],
    entity: Optional[str],
    state: Dict[str, Any]
) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Dùng ngữ cảnh để suy ra entity khi user hỏi chung chung:
    - Ưu tiên bệnh đã lưu trong state["disease"] (từ câu hỏi trước)
    - Nếu không có thì dùng nhãn từ chẩn đoán ảnh gần nhất
    """
    used_image_ctx = False
    text = (user_input or "").lower()

    # 1) Nếu đã có intent thuộc nhóm bệnh nhưng entity chung chung
    if intent in DISEASE_INTENTS and _is_generic_entity(entity):
        # a) Ưu tiên bệnh từ câu trước
        last_disease = _last_disease_from_state(state)
        if last_disease:
            return intent, last_disease, False

        # b) Nếu không có, thử lấy từ ảnh
        top_label = _top_predicted_label_from_state(state)
        if top_label:
            return intent, top_label, True

    # 2) Nếu chưa có intent rõ ràng nhưng có context bệnh (state hoặc ảnh)
    has_any_disease_ctx = (
        _last_disease_from_state(state) is not None
        or state.get("last_image_diagnosis") is not None
    )
    if not intent and has_any_disease_ctx:
        if re.search(
            r"(triệu chứng|biểu hiện|điều trị|thuốc|nguyên nhân|tại sao|phòng ngừa|phòng tránh|tiêm|vaccine)",
            text,
        ):
            # Ưu tiên bệnh từ state, nếu không có mới lấy label từ ảnh
            disease = _last_disease_from_state(state) or _top_predicted_label_from_state(state)
            if not disease:
                return intent, entity, False

            used_image_ctx = (
                _last_disease_from_state(state) is None
                and _top_predicted_label_from_state(state) is not None
            )

            if re.search(r"triệu chứng|biểu hiện", text):
                return "hoi_trieu_chung", disease, used_image_ctx
            if re.search(r"điều trị|thuốc|chữa", text):
                return "hoi_cach_dieu_tri", disease, used_image_ctx
            if re.search(r"nguyên nhân|tại sao", text):
                return "hoi_nguyen_nhan", disease, used_image_ctx
            if re.search(r"phòng ngừa|phòng tránh|tiêm|vaccine", text):
                return "hoi_phong_ngua", disease, used_image_ctx

    # 3) Không suy ra thêm được gì
    return intent, entity, False
