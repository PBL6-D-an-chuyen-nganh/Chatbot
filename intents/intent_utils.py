import requests
import numpy as np
from sentence_transformers import util


def find_article(query, model, index, articles):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, 1)
    return articles[indices[0][0]]


def find_doctor_by_name(name, doctors, model):
    names = [d["name"] for d in doctors]
    embeddings = model.encode(names, convert_to_numpy=True)
    query_vec = model.encode([name], convert_to_numpy=True)
    sims = util.cos_sim(query_vec, embeddings)[0]
    best_idx = int(np.argmax(sims))
    if sims[best_idx] > 0.6:
        return doctors[best_idx]
    return None


def get_available_slots(doctor_id, slots_api):
    try:
        url = slots_api.replace("{id}", str(doctor_id))
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        slots_data = resp.json()
        if not isinstance(slots_data, dict) or not slots_data:
            return []
        result = []
        for date_str, times in slots_data.items():
            for t in times:
                result.append(f"{date_str}T{t}")
        return result
    except Exception as e:
        print("⚠️ Lỗi lấy lịch trống:", e)
        return []


def create_appointment(data, appointments_api):
    try:
        resp = requests.post(appointments_api, json=data, timeout=15)
        if resp.status_code in [200, 201]:
            return True
        print("❌ Lỗi tạo lịch:", resp.status_code, resp.text)
    except Exception as e:
        print("⚠️ Không gửi được yêu cầu:", e)
    return False
