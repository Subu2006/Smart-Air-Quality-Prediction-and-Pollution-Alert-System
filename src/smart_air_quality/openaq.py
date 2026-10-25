"""Optional OpenAQ v3 latest-measurements reader. It does not derive AQI from partial pollutants."""
from __future__ import annotations
import os
import requests

BASE_URL = "https://api.openaq.org/v3"


def fetch_location_latest(location_id: str | int | None = None, api_key: str | None = None, timeout: int = 15) -> dict:
    key = api_key or os.getenv("OPENAQ_API_KEY")
    location = str(location_id or os.getenv("OPENAQ_LOCATION_ID", "")).strip()
    if not key:
        raise RuntimeError("OpenAQ API key not configured. Set OPENAQ_API_KEY in .env.")
    if not location:
        raise RuntimeError("OpenAQ location ID not configured. Set OPENAQ_LOCATION_ID in .env.")
    response = requests.get(f"{BASE_URL}/locations/{location}/latest", headers={"X-API-Key": key}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return {"source": "OpenAQ API v3", "location_id": location, "retrieved_at_utc": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(), "payload": payload}
