# intents/appointment/slots_service.py
from typing import Any, Dict, List, Optional
from datetime import datetime, date
import requests

from .time_parses import minutes_from_time_str

# Số slot tối đa gợi ý cho user
MAX_SUGGESTIONS = 10
# Khoảng lệch tối đa (phút) nếu không có giờ chính xác
NEAR_THRESHOLD_MINUTES = 35


def normalize_slots(slots_raw: Any) -> List[Dict[str, Any]]:
    """
    Chuẩn hoá dữ liệu lịch trống về dạng:
    {
        "date": "YYYY-MM-DD",
        "time": "HH:MM[:SS]",
        "datetime": "YYYY-MM-DD HH:MM:SS",
        "display": "...",
        "location": "...",
    }
    """
    out: List[Dict[str, Any]] = []

    if isinstance(slots_raw, list):
        for s in slots_raw:
            if not isinstance(s, dict):
                continue
            time_str = s.get("time") or s.get("startTime") or s.get("start")
            date_str = s.get("date")
            if not time_str or not date_str:
                continue

            if len(time_str) == 5:  # HH:MM
                dt_iso = f"{date_str} {time_str}:00"
            else:
                dt_iso = f"{date_str} {time_str}"

            out.append(
                {
                    "date": date_str,
                    "time": time_str,
                    "datetime": dt_iso,
                    "display": dt_iso,
                    "location": s.get("location") or s.get("room") or "",
                }
            )

    elif isinstance(slots_raw, dict):
        for date_str, times in slots_raw.items():
            if not isinstance(times, list):
                continue
            for t in times:
                time_str = str(t)
                if len(time_str) == 5:
                    dt_iso = f"{date_str} {time_str}:00"
                else:
                    dt_iso = f"{date_str} {time_str}"

                out.append(
                    {
                        "date": date_str,
                        "time": time_str,
                        "datetime": dt_iso,
                        "display": dt_iso,
                        "location": "",
                    }
                )

    def _to_key(s):
        try:
            return datetime.strptime(s.get("datetime", ""), "%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.max

    out.sort(key=_to_key)
    return out


def format_slots(slots: List[Dict[str, Any]]) -> str:
    if not slots:
        return "❌ Hiện chưa có lịch trống nào."
    lines = [
        "🗓️ Các lịch trống (chọn số):",
        "(Sau khi chọn giờ, bạn sẽ được hướng dẫn nhập Họ tên và SĐT bệnh nhân.)",
    ]
    for i, s in enumerate(slots, 1):
        lines.append(
            f"{i}. {s['display']}{(' • ' + s['location']) if s.get('location') else ''}"
        )
    return "\n".join(lines)


def find_slots_for_all_doctors(
    doctors: List[Dict[str, Any]],
    SLOTS_API: str,
    day: date,
    desired_minutes: Optional[int],
    half_day: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Quét tất cả bác sĩ, tìm các slot trong ngày 'day'
    gần với giờ mong muốn (desired_minutes) và buổi (half_day).

    Trả về danh sách slot đã sort + giới hạn MAX_SUGGESTIONS:
    {
       "date", "time", "datetime", "display",
       "location"  (tên bác sĩ),
       "doctorId",
       "doctorName",
       "is_exact",
       "diff_min",
    }
    """
    date_str = day.strftime("%Y-%m-%d")
    all_candidates: List[Dict[str, Any]] = []

    for d in doctors:
        d_id = (
            d.get("doctorID")
            or d.get("id")
            or d.get("doctorId")
            or d.get("userId")
        )
        if not d_id:
            continue

        try:
            url = SLOTS_API.format(id=d_id)
            print(f"[appointment] fetching slots for doctor {d_id} URL: {url}")
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            raw = (
                resp.json()
                if resp.headers.get("content-type", "").startswith("application/json")
                else {}
            )
            slots = normalize_slots(raw)
        except Exception as e:
            print(f"[appointment] fetch slots error for doctor {d_id}: {e}")
            continue

        for s in slots:
            if s.get("date") != date_str:
                continue

            slot_minutes = minutes_from_time_str(s.get("time", ""))
            if slot_minutes is None:
                continue

            # Lọc theo buổi nếu có
            h_slot = slot_minutes // 60
            if half_day == "morning" and not (0 <= h_slot < 12):
                continue
            if half_day == "afternoon" and not (12 <= h_slot < 18):
                continue
            if half_day == "evening" and not (18 <= h_slot <= 23):
                continue

            is_exact = False
            diff_min = None
            if desired_minutes is not None:
                diff_min = abs(slot_minutes - desired_minutes)
                is_exact = diff_min == 0
            else:
                diff_min = 0

            all_candidates.append(
                {
                    "date": s["date"],
                    "time": s["time"],
                    "datetime": s["datetime"],
                    "display": s["datetime"],
                    "location": d.get("name") or "",
                    "doctorId": d_id,
                    "doctorName": d.get("name") or "",
                    "is_exact": is_exact,
                    "diff_min": diff_min,
                }
            )

    if not all_candidates:
        return []

    # Ưu tiên slot khớp chính xác
    exact_slots = [c for c in all_candidates if c["is_exact"]]
    if exact_slots:
        chosen = sorted(exact_slots, key=lambda c: c["datetime"])
    else:
        near_slots = [
            c
            for c in all_candidates
            if c["diff_min"] is not None and c["diff_min"] <= NEAR_THRESHOLD_MINUTES
        ]
        if not near_slots:
            return []
        chosen = sorted(near_slots, key=lambda c: (c["diff_min"], c["datetime"]))

    return chosen[:MAX_SUGGESTIONS]
