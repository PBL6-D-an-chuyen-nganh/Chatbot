from intents.intent_utils import find_article

def handle_disease_intent(intent, entity, user_input, model, index, articles, state):
    article = find_article(entity or user_input, model, index, articles)
    state["disease"] = article["title"]
    if intent == "hoi_trieu_chung":
        return f"Triệu chứng của {article['title']}: {article.get('symptoms','Chưa có dữ liệu.')}", state
    if intent == "hoi_cach_dieu_tri":
        return f"Cách điều trị {article['title']}: {article.get('treatment','Chưa có dữ liệu.')}", state
    if intent == "hoi_nguyen_nhan":
        return f"Nguyên nhân của {article['title']}: {article.get('cause','Chưa có dữ liệu.')}", state
    if intent == "hoi_phong_ngua":
        return f"Phòng ngừa {article['title']}: {article.get('prevention','Chưa có dữ liệu.')}", state
