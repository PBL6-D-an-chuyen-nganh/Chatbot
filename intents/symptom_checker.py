from intents.intent_utils import keyword_match_articles_by_symptoms

def handle_symptom_inquiry(user_input, model, index, articles, state):
    # Dùng LIKE trên trường symptoms
    matched = keyword_match_articles_by_symptoms(
        query=user_input,
        articles=articles,
        min_hits=1,
        top_k=3,   # lấy 3 bệnh khớp nhất
    )

    if not matched:
        return (
            "Em chưa đủ thông tin để tư vấn chính xác. "
            "Anh/chị mô tả rõ hơn vị trí, biểu hiện, thời gian bị nhé.",
            state,
        )

    # Lưu lại danh sách bệnh gợi ý để user hỏi sâu thêm
    state["suggested_diseases"] = [d.get("title") for d in matched]
    state["last_action"] = "symptom_check"

    # Tính max _hits để quy ra % tương đối
    max_hits = max(d.get("_hits", 1) for d in matched) or 1

    response = "Dựa trên triệu chứng anh/chị mô tả, các bệnh có khả năng gồm:\n\n"

    for i, disease in enumerate(matched, 1):
        title = disease.get("title", "Chưa rõ")
        symptoms = disease.get("symptoms", "") or ""
        preview = symptoms[:150] + ("..." if len(symptoms) > 150 else "")

        hits = int(disease.get("_hits", 0))
        percent = int(hits * 100 / max_hits) if max_hits > 0 else 0

        response += (
            f"{i}. **{title}**  (mức độ phù hợp ~ {percent}%)\n"
            f"   → {preview}\n\n"
        )

    response += (
        "⚠️ Đây chỉ là gợi ý tham khảo, **không thay thế chẩn đoán của bác sĩ**.\n"
        "→ Anh/chị có thể hỏi chi tiết hơn về **một trong các bệnh trên** "
        "(ví dụ: *\"triệu chứng của viêm da cơ địa\"*, *\"nguyên nhân của lang ben\"*), "
        "hoặc em hỗ trợ **đặt lịch khám** nếu cần."
    )

    return response, state
