from intents.intent_utils import find_article

def handle_disease_intent(intent, entity, user_input, model, index, articles, state):
    # Ưu tiên entity (tên bệnh) nếu có, fallback = cả câu hỏi
    query = (entity or user_input or "").strip()
    
    # 👇 MỚI: Nếu entity chung chung, kiểm tra context trong state
    # (thay vì chỉ dùng entity trực tiếp từ Gemini)
    if not entity and state.get("disease"):
        # Kiểm tra xem câu hỏi hiện tại có phải là follow-up không
        lower_input = user_input.lower()
        if any(kw in lower_input for kw in ["vậy", "nó", "cái này", "bệnh đó", "như thế nào", "thế nào"]):
            query = state["disease"]
    
    article = find_article(query, model, index, articles)

    if not article:
        # Không tìm được bài đủ tin cậy
        base = "Em chưa tìm được thông tin bệnh phù hợp từ dữ liệu hiện có."
        
        # 👇 MỚI: Gợi ý dựa trên suggested_diseases từ symptom_checker
        if state.get("suggested_diseases"):
            suggestions = ", ".join(state["suggested_diseases"])
            base += f"\n\n💡 Anh/chị đang muốn hỏi về: {suggestions} phải không ạ?"
        
        if intent == "hoi_trieu_chung":
            return base + " Anh/chị mô tả rõ hơn tên bệnh hoặc triệu chứng giúp em nhé.", state
        if intent == "hoi_cach_dieu_tri":
            return base + " Anh/chị cho em biết rõ tên bệnh để em hỗ trợ tốt hơn ạ.", state
        if intent == "hoi_nguyen_nhan":
            return base + " Anh/chị có thể nói rõ tên bệnh giúp em được không ạ?", state
        if intent == "hoi_phong_ngua":
            return base + " Anh/chị nói rõ bệnh nào để em tư vấn cách phòng ngừa chính xác hơn nhé.", state
        # fallback chung
        return base, state

    # Lưu lại bệnh đang nói đến trong state
    state["disease"] = article.get("title", "")

    title = article.get("title", "bệnh này")
    
    # 👇 MỚI: Thêm follow-up suggestions theo mỗi intent
    follow_up = "\n\n💡 "
    
    if intent == "hoi_trieu_chung":
        response = f"Triệu chứng của {title}: {article.get('symptoms','Chưa có dữ liệu.')}"
        follow_up += "Anh/chị muốn biết thêm về cách điều trị hoặc nguyên nhân không?"
        
    elif intent == "hoi_cach_dieu_tri":
        response = f"Cách điều trị {title}: {article.get('treatment','Chưa có dữ liệu.')}"
        follow_up += "Anh/chị có thể đặt lịch khám với bác sĩ chuyên khoa để được tư vấn chi tiết hơn nhé."
        
    elif intent == "hoi_nguyen_nhan":
        response = f"Nguyên nhân của {title}: {article.get('cause','Chưa có dữ liệu.')}"
        follow_up += "Anh/chị muốn biết cách phòng ngừa không ạ?"
        
    elif intent == "hoi_phong_ngua":
        response = f"Phòng ngừa {title}: {article.get('prevention','Chưa có dữ liệu.')}"
        follow_up += "Nếu đã có triệu chứng, anh/chị nên đặt lịch khám sớm nhé."
        
    else:
        response = f"Thông tin về {title}: {article.get('symptoms','Chưa có dữ liệu.')}"
        follow_up = ""
    
    return response + follow_up, state