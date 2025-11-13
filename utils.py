import requests
import os
import json
import faiss
import numpy as np


def _extract_list_from_response(obj):
    """Try common wrapper keys to extract a list of items from API response."""
    if isinstance(obj, list):
        return obj
    if not isinstance(obj, dict):
        return [obj]
    for key in ("content", "data", "items", "articles", "results"):
        val = obj.get(key)
        if isinstance(val, list):
            return val
    # fallback: if dict contains numeric-indexed keys, try to collect
    seq = []
    for k in sorted(obj.keys()):
        try:
            int(k)
            seq.append(obj[k])
        except Exception:
            continue
    if seq:
        return seq
    return [obj]


def fetch_and_cache(api_url, cache_path):
    try:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        resp = requests.get(api_url, timeout=5)
        resp.raise_for_status()
        body = resp.json()
        data = _extract_list_from_response(body)
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
