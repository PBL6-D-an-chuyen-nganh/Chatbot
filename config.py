import os

# API
BASE_URL = "http://localhost:8080/api"
ARTICLES_API = f"{BASE_URL}/articles/all"
DOCTORS_API = f"{BASE_URL}/doctors"
SLOTS_API = f"{BASE_URL}/doctors/{{id}}/available-slots"
APPOINTMENTS_API = f"{BASE_URL}/user/appointments/create"
USER_APPOINTMENTS_API = f"{BASE_URL}/user/appointments/creator/{{creatorId}}"
CANCEL_APPOINTMENT_API = f"{BASE_URL}/user/appointments/{{id}}"
# Cache
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
ARTICLES_PATH = os.path.join(CACHE_DIR, "articles.json")
DOCTORS_PATH = os.path.join(CACHE_DIR, "doctors.json")
FAISS_PATH = os.path.join(CACHE_DIR, "articles.index")
TITLES_PATH = os.path.join(CACHE_DIR, "article_titles.npy")
