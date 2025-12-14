# intents/appointment/time_parses.py
from typing import Optional, Tuple
import re
from datetime import date, datetime


def minutes_from_time_str(time_str: str) -> Optional[int]:
    """
    '07:35:00' -> 7*60+35
    '07:35'    -> 7*60+35
    """
    if not time_str:
        return None
    parts = time_str.split(":")
    if len(parts) < 2:
        return None
    try:
        h = int(parts[0])
        m = int(parts[1])
        return h * 60 + m
    except ValueError:
        return None


def extract_date_and_time(user_text: str) -> Tuple[Optional[date], Optional[int], int, Optional[str]]:
    """
    Trả về:
      - day: datetime.date hoặc None
      - hour: int hoặc None
      - minute: int (mặc định 0)
      - half_day: 'morning' | 'afternoon' | 'evening' | None

    Hỗ trợ:
      - ngày: 15/12, 15-12, 15/12/2025, ...
      - giờ: 16:30, 16h30, 16h, 16 giờ, 7 giờ 35 phút, 7 gio 35 phut
    """
    text = (user_text or "").lower()

    # --- 1) Parse ngày: dd/mm(/yyyy) hoặc dd-mm(-yyyy) ---
    day: Optional[date] = None
    m_date = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{4}))?", text)
    if m_date:
        d = int(m_date.group(1))
        m = int(m_date.group(2))
        if m_date.group(3):
            y = int(m_date.group(3))
        else:
            y = datetime.now().year
        try:
            day = date(y, m, d)
        except ValueError:
            day = None

    # --- 2) Parse giờ ---
    hour: Optional[int] = None
    minute: int = 0

    # 2.1: dạng HH:MM
    m_time = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if m_time:
        hour = int(m_time.group(1))
        minute = int(m_time.group(2))
    else:
        # 2.2: dạng HHhMM
        m_time = re.search(r"\b(\d{1,2})h(\d{2})\b", text)
        if m_time:
            hour = int(m_time.group(1))
            minute = int(m_time.group(2))
        else:
            # 2.3: dạng "7 giờ 35 phút" / "7 gio 35 phut"
            m_time = re.search(
                r"\b(\d{1,2})\s*(?:giờ|gio|h)\s*(\d{1,2})\s*(?:phút|phut|p)?\b",
                text,
            )
            if m_time:
                hour = int(m_time.group(1))
                minute = int(m_time.group(2))
            else:
                # 2.4: dạng "16h" / "16 giờ"
                m_time = re.search(r"\b(\d{1,2})\s*(?:h|giờ|gio)\b", text)
                if m_time:
                    hour = int(m_time.group(1))
                    minute = 0

    # --- 3) Sáng / chiều / tối ---
    half_day: Optional[str] = None
    if "sáng" in text:
        half_day = "morning"
    elif "chiều" in text:
        half_day = "afternoon"
    elif "tối" in text or "đêm" in text:
        half_day = "evening"

    return day, hour, minute, half_day
