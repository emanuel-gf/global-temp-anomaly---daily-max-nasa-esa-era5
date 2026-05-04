"""
Live weather scraper for LGfL Weather Monitoring System.
Source: https://weather.lgfl.org.uk/table.aspx

Returns one dict per station row found in the table.
Variables differ from nw3weather.py — this source has solar radiation,
UV index, wind direction, and rain rate, but no feels-like or dew point.
"""

import re
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

SOURCE_ID = "lgfl"
URL = "https://weather.lgfl.org.uk/table.aspx"
TIMEOUT = 8

# Column order as they appear in the table (left to right, after Station + Time)
COLUMNS = [
    "temp_c",
    "pressure_hpa",
    "rain_rate_mm_hr",
    "rain_day_mm",
    "wind_dir_deg",
    "wind_avg_ms",
    "humidity_pct",
    "solar_rad_wm2",
    "uv_index",
]


class ScraperError(Exception):
    pass


def fetch() -> list[dict]:
    """
    Fetch all station rows from the LGfL table view.

    Returns a list of dicts, one per station:
        [
            {
                "source":           "lgfl",
                "station":          "Bow, London",
                "temp_c":           18.7,
                "pressure_hpa":     1020.0,
                "rain_rate_mm_hr":  0.0,
                "rain_day_mm":      0.0,
                "wind_dir_deg":     15.0,
                "wind_avg_ms":      0.0,
                "humidity_pct":     41.0,
                "solar_rad_wm2":    258.0,
                "uv_index":         2.5,
                "obs_time":         "14:56",   # local time string from site
                "fetched_at":       "2026-04-18T13:56:00+00:00",
            },
            ...
        ]

    Raises:
        ScraperError: on network or parse failure.
    """
    html = _get_html()
    soup = BeautifulSoup(html, "html.parser")
    return _parse_table(soup)


# ── private helpers ────────────────────────────────────────────────────────────

def _get_html() -> str:
    try:
        r = requests.get(URL, timeout=TIMEOUT)
        r.raise_for_status()
        return r.text
    except requests.RequestException as exc:
        raise ScraperError(f"{SOURCE_ID}: request failed — {exc}") from exc


def _parse_table(soup: BeautifulSoup) -> list[dict]:
    """
    The table has a header row (Station | Time | Temp Out | Bar | ...)
    followed by one data row per station.
    We locate the table by its header content, then parse each data row.
    """
    table = _find_data_table(soup)
    if table is None:
        raise ScraperError(f"{SOURCE_ID}: data table not found in page")

    rows = table.find_all("tr")
    # first row is the header — skip it
    data_rows = [r for r in rows if r.find("td")]

    if not data_rows:
        raise ScraperError(f"{SOURCE_ID}: no data rows found in table")

    fetched_at = datetime.now().isoformat()
    results = []

    for row in data_rows:
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) < 2:
            continue
        record = _parse_row(cells, fetched_at)
        if record:
            results.append(record)

    return results


def _find_data_table(soup: BeautifulSoup) -> BeautifulSoup | None:
    """Find the table that contains 'Temp Out' in its header row."""
    for table in soup.find_all("table"):
        if "Temp Out" in table.get_text():
            return table
    return None


def _parse_row(cells: list[str], fetched_at: str) -> dict | None:
    """
    cells[0] = station name
    cells[1] = obs time  (e.g. "14:56")
    cells[2..] = numeric values matching COLUMNS order
    """
    if len(cells) < 2:
        return None

    station = cells[0].strip()
    obs_time_str = cells[1].strip()

    # need at least station + time + one numeric column
    if not obs_time_str or ":" not in obs_time_str:
        return None  # or continue if inside a loop

    print(obs_time_str)
    obs_time = datetime.strptime(obs_time_str, "%H:%M")

    # Attach today's date and respect that obs_time. Without using timezone.
    now = datetime.now()
    obs_time = obs_time.replace(year=now.year, month=now.month, day=now.day).strftime("%Y-%m-%d-%H:%M:%S")

    record = {
        "source":     SOURCE_ID,
        "station":    station,
        "obs_time":   obs_time,
        "fetched_at": fetched_at,
    }

    for i, col in enumerate(COLUMNS):
        cell_index = i + 2  # offset past station + time
        if cell_index < len(cells):
            record[col] = _to_float(cells[cell_index])
        else:
            record[col] = None

    return record


def _to_float(value: str) -> float | None:
    """Parse a cell value to float; return None if empty or non-numeric."""
    v = value.strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


# ── quick manual test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    try:
        stations = fetch()
        print(json.dumps(stations, indent=2))
        print(f"\n{len(stations)} stations fetched.")
    except ScraperError as e:
        print(f"Error: {e}")