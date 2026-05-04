"""
Live temperature scraper for NW3 Weather — Hampstead, North London.
Source: https://nw3weather.co.uk/

Server-rendered HTML, updates every ~60s on the site.
We make at most one request per call; caller is responsible for rate-limiting.
"""

import re
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

SOURCE_ID = "hampstead-nw3"
URL = "https://nw3weather.co.uk/"
TIMEOUT = 8  # seconds


class ScraperError(Exception):
    pass


def fetch() -> dict:
    """
    Fetch the current observation from nw3weather.co.uk.

    Returns:
        {
            "source":       "hampstead-nw3",
            "station":      "Hampstead NW3, London",
            "temp_c":       16.4,
            "humidity_pct": 48,
            "pressure_hpa": 1022,
            "feels_like_c": 16.4,
            "timestamp":    "2026-04-18T14:52:00+01:00",   # ISO 8601, local time
            "fetched_at":   "2026-04-18T13:52:34+00:00",   # UTC
        }

    Raises:
        ScraperError: on any network or parse failure.
    """
    html = _get_html()
    soup = BeautifulSoup(html, "html.parser")

    return {
        "source":       SOURCE_ID,
        "station":      "Hampstead NW3, London",
        "temp_c":       _parse_temperature(soup),
        "humidity_pct": _parse_humidity(soup),
        "pressure_hpa": _parse_pressure(soup),
        "feels_like_c": _parse_feels_like(soup),
        "obs_time":    _parse_site_timestamp(soup),
        "fetched_at":   datetime.now(timezone.utc).isoformat(),
    }


# ── private helpers ────────────────────────────────────────────────────────────

def _get_html() -> str:
    try:
        r = requests.get(URL, timeout=TIMEOUT)
        r.raise_for_status()
        return r.text
    except requests.RequestException as exc:
        raise ScraperError(f"{SOURCE_ID}: request failed — {exc}") from exc


def _parse_temperature(soup: BeautifulSoup) -> float:
    """
    The live table row looks like:
      <td><b>Temperature</b></td>
      <td><b>16.4 °C</b> <br/> falling </td>
    """
    return _find_value_after_label(soup, "Temperature", SOURCE_ID)


def _parse_humidity(soup: BeautifulSoup) -> int | None:
    try:
        val = _find_value_after_label(soup, "Relative Humidity", SOURCE_ID)
        return int(val)
    except ScraperError:
        return None


def _parse_pressure(soup: BeautifulSoup) -> int | None:
    try:
        val = _find_value_after_label(soup, "Pressure", SOURCE_ID)
        return int(val)
    except ScraperError:
        return None


def _parse_feels_like(soup: BeautifulSoup) -> float | None:
    """
    Feels Like appears in the summary bar as:
      Feels Like:  16.4 °C (Daily Min: 6.8 °C)
    """
    text = soup.get_text(" ", strip=True)
    m = re.search(r"Feels Like[:\s]+(-?\d+\.?\d*)\s*°C", text)
    if m:
        return float(m.group(1))
    return None


def _parse_site_timestamp(soup: BeautifulSoup) -> str | None:
    """
    The page prints: 'Data recorded at 14:52:34 BST'
    We combine that time with today's date (UTC-aware).
    """
    text = soup.get_text(" ", strip=True)
    m = re.search(r"Data recorded at\s+(\d{2}:\d{2}:\d{2})\s+(\w+)", text)
    if not m:
        return None
    time_str, _ = m.group(1), m.group(2)
    today = datetime.now().strftime("%Y-%m-%d")
    return f"{today}-{time_str}"


def _find_value_after_label(soup: BeautifulSoup, label: str, source: str) -> float:
    """
    Walk all <td> cells; when we find one containing `label`,
    the next sibling <td> holds the current value as the first number.
    """
    for td in soup.find_all("td"):
        if label in td.get_text():
            next_td = td.find_next_sibling("td")
            if next_td:
                m = re.search(r"(-?\d+\.?\d*)", next_td.get_text())
                if m:
                    return float(m.group(1))
    raise ScraperError(f"{source}: could not parse '{label}'")


# ── quick manual test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    try:
        data = fetch()
        print(json.dumps(data, indent=2))
    except ScraperError as e:
        print(f"Error: {e}")