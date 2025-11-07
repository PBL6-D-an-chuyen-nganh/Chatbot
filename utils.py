import requests
import os
import json
import faiss
import numpy as np

def fetch_and_cache(api_url, cache_path):
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        resp = requests.get(api_url, timeout=5)
        resp.raise_for_status()
        data = resp.json().get("content", [])
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data
    except Exception as e:
        print(f"⚠️ Lỗi khi tải dữ liệu từ {api_url}: {e}")
        return []

def build_or_load_faiss(articles, model, faiss_path, titles_path):
    if os.path.exists(faiss_path) and os.path.exists(titles_path):
        index = faiss.read_index(faiss_path)
        titles = np.load(titles_path, allow_pickle=True).tolist()
        print("📦 Đã tải FAISS index từ cache.")
        return index, titles

    texts = [
        f"{a['title']} {a.get('symptoms','')} {a.get('treatment','')} "
        f"{a.get('cause','')} {a.get('prevention','')}" for a in articles
    ]
    titles = [a["title"] for a in articles]

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, faiss_path)
    np.save(titles_path, np.array(titles))
    print("✅ FAISS index created and saved.")
    return index, titles
