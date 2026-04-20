"""
CGESP Weather Station Fetcher
==============================
Fetches observed temperature data from CGESP (Centro de Gerenciamento de
Emergências da Prefeitura de São Paulo).

Two data sources per station:
  1. processo_cge.jsp  – 24h hourly table (with ↑/↓ temperature arrows)
  2. estacao.jsp       – current summary (max/min/current per variable)

Changelog
---------
v2 – Fixed both parsers against real HTML:
  • Summary: regex approach replaced with column-section DOM traversal.
    The page uses 'Mãxima' (ã not á) and has no Wind section — only
    Chuva / Temperatura / Umidade / Pressão columns.
  • History: column indices are now derived from the actual <th> header row
    instead of being hardcoded, making the parser robust to column reordering.
  • Added per-request delay to avoid triggering rate-limiting.
"""

from __future__ import annotations

import re
import time
import logging
from datetime import datetime
from typing import Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

HISTORY_URL = "https://www.cgesp.org/processo_cge.jsp"
SUMMARY_URL = "https://www.cgesp.org/v3/estacao.jsp"

# Seconds to wait between consecutive requests (per station).
# Keeps us well below any rate-limit threshold.
REQUEST_DELAY = 1.0

_ARROW_CHARS = set("↑↗▲↓↘▼")

# Arrow image filename fragments (src / alt attributes)
_ARROW_UP_PATTERNS   = ("seta_cima", "arrow_up", "arrow-up", "alta", "subindo")
_ARROW_DOWN_PATTERNS = ("seta_baixo", "arrow_down", "arrow-down", "baixa", "caindo")

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8",
    "Referer": "https://www.cgesp.org/",
}

# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class CGESPError(Exception):
    """Raised for all CGESP fetch/parse failures."""


# ──────────────────────────────────────────────────────────────────────────────
# Low-level HTTP helper
# ──────────────────────────────────────────────────────────────────────────────

def _get(url: str, params: dict, retries: int = 3, timeout: int = 15) -> str:
    """GET with retry, block-detection, and exponential back-off."""
    last_exc: Exception = CGESPError("No attempts made")

    for attempt in range(retries):
        try:
            if attempt > 0:
                time.sleep(REQUEST_DELAY * (attempt + 1))

            r = requests.get(url, headers=_DEFAULT_HEADERS, params=params, timeout=timeout)

            if "Não é permitido acesso direto" in r.text:
                raise CGESPError("Blocked: missing/invalid Referer header")

            r.raise_for_status()
            return r.text

        except CGESPError:
            raise
        except Exception as exc:
            last_exc = exc
            logger.warning("Attempt %d/%d failed for %s params=%s: %s",
                           attempt + 1, retries, url, params, exc)

    raise CGESPError(f"All {retries} attempts failed: {last_exc}") from last_exc


# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────────

def _safe_float(val: str) -> Optional[float]:
    """Parse a numeric string to float, returning None on failure."""
    if val is None:
        return None
    try:
        # Keep only digits, dot, minus; replace comma decimal separator
        cleaned = re.sub(r"[^\d\.\-]", "", str(val).replace(",", "."))
        return float(cleaned) if cleaned else None
    except (ValueError, TypeError):
        return None


def _strip_arrows(s: str) -> str:
    """Remove arrow unicode characters so float parsing is not disrupted."""
    return "".join(ch for ch in s if ch not in _ARROW_CHARS).strip()


_PT_MONTHS = {
    "JAN": 1, "FEV": 2, "MAR": 3, "ABR": 4, "MAI": 5, "JUN": 6,
    "JUL": 7, "AGO": 8, "SET": 9, "OUT": 10, "NOV": 11, "DEZ": 12,
}


def _parse_pt_datetime(s: str) -> Optional[datetime]:
    """Parse a Portuguese date string like '16 ABR 2026 14:00'."""
    s = s.strip()
    m = re.match(
        r"(\d{1,2})\s+([A-Z]{3})\s+(\d{4})\s+(\d{2}):(\d{2})",
        s, re.IGNORECASE,
    )
    if not m:
        return None
    day, mon_str, year, hour, minute = m.groups()
    month = _PT_MONTHS.get(mon_str.upper())
    if month is None:
        return None
    try:
        return datetime(int(year), month, int(day), int(hour), int(minute))
    except ValueError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Arrow detection (history table)
# ──────────────────────────────────────────────────────────────────────────────

def _detect_arrow(cell: Tag) -> Optional[str]:
    """Return 'up', 'down', or None from a <td> BeautifulSoup tag."""
    text = cell.get_text()

    # 1. Unicode arrows in text
    if any(ch in text for ch in ("↑", "↗", "▲")):
        return "up"
    if any(ch in text for ch in ("↓", "↘", "▼")):
        return "down"

    # 2. <img> src / alt
    for img in cell.find_all("img"):
        combined = ((img.get("src") or "") + " " + (img.get("alt") or "")).lower()
        if any(p in combined for p in _ARROW_UP_PATTERNS):
            return "up"
        if any(p in combined for p in _ARROW_DOWN_PATTERNS):
            return "down"

    # 3. CSS class names on any child element
    for tag in cell.find_all(True):
        cls = " ".join(tag.get("class") or []).lower()
        if any(p in cls for p in _ARROW_UP_PATTERNS):
            return "up"
        if any(p in cls for p in _ARROW_DOWN_PATTERNS):
            return "down"

    return None


# ──────────────────────────────────────────────────────────────────────────────
# History table parser  (processo_cge.jsp)
# ──────────────────────────────────────────────────────────────────────────────

# Canonical column names we look for in the <th> header row.
# Each entry is (output_key, list_of_substrings_to_match_in_header_text).
_HISTORY_COL_MAP = [
    ("raw_date",     ["data", "hora", "date", "time"]),
    ("chuva_mm",     ["chuva", "rain", "precip"]),
    ("vel_vt_ms",    ["vel", "vento", "wind", "speed"]),
    ("dir_vt_graus", ["dir"]),
    ("temp_c",       ["temp"]),
    ("umid_rel_pct", ["umid", "humid"]),
    ("pressao_mb",   ["press"]),
]


def _match_col(header_text: str, keywords: list[str]) -> bool:
    h = header_text.lower()
    return any(k in h for k in keywords)


def _parse_history(html: str) -> pd.DataFrame:
    """
    Parse the 24h hourly table from processo_cge.jsp.

    Column positions are detected from the actual <th> header row so the
    parser remains correct even if CGESP reorders columns.

    Returns a DataFrame with columns:
        timestamp, chuva_mm, vel_vt_ms, dir_vt_graus,
        temp_c, umid_rel_pct, pressao_mb, temp_trend
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")

    if table is None:
        raise CGESPError("No <table> found in history page")

    rows = table.find_all("tr")
    if not rows:
        raise CGESPError("History table has no rows")

    # ── Detect header row and build col_index map ─────────────────────────
    # Find the first row that contains <th> or whose text matches our keywords
    col_index: dict[str, int] = {}
    header_row_idx = 0

    for row_idx, tr in enumerate(rows):
        cells = tr.find_all(["th", "td"])
        texts = [c.get_text(strip=True) for c in cells]
        # A header row has at least one cell matching a known column keyword
        matches = sum(
            1 for txt in texts
            if any(_match_col(txt, kws) for _, kws in _HISTORY_COL_MAP)
        )
        if matches >= 2:
            # Map each canonical column to its position
            for out_key, kws in _HISTORY_COL_MAP:
                for ci, txt in enumerate(texts):
                    if _match_col(txt, kws) and out_key not in col_index:
                        col_index[out_key] = ci
            header_row_idx = row_idx
            logger.debug("Header row %d detected: %s -> col_index=%s", row_idx, texts, col_index)
            break

    if not col_index:
        # Fallback: assume fixed order Data,Chuva,Vel,Dir,Temp,Umid,Pressão
        col_index = {
            "raw_date": 0, "chuva_mm": 1, "vel_vt_ms": 2,
            "dir_vt_graus": 3, "temp_c": 4, "umid_rel_pct": 5, "pressao_mb": 6,
        }
        logger.warning("Could not detect header row — using fallback column order")

    # Determine which index holds temperature (for arrow detection)
    temp_col_idx = col_index.get("temp_c", 4)

    # ── Parse data rows ───────────────────────────────────────────────────
    rows_data = []
    for tr in rows[header_row_idx + 1:]:
        cells = tr.find_all(["td", "th"])
        if len(cells) < 2:
            continue

        def _cell_text(key: str) -> str:
            idx = col_index.get(key)
            if idx is None or idx >= len(cells):
                return ""
            return _strip_arrows(cells[idx].get_text(strip=True))

        raw_date = _cell_text("raw_date")

        # Skip rows that don't look like timestamps
        if not _parse_pt_datetime(raw_date):
            logger.debug("Skipping non-date row: %s", [c.get_text(strip=True) for c in cells])
            continue

        trend = _detect_arrow(cells[temp_col_idx]) if temp_col_idx < len(cells) else None

        rows_data.append({
            "raw_date":     raw_date,
            "chuva_mm":     _safe_float(_cell_text("chuva_mm")),
            "vel_vt_ms":    _safe_float(_cell_text("vel_vt_ms")),
            "dir_vt_graus": _safe_float(_cell_text("dir_vt_graus")),
            "temp_c":       _safe_float(_cell_text("temp_c")),
            "umid_rel_pct": _safe_float(_cell_text("umid_rel_pct")),
            "pressao_mb":   _safe_float(_cell_text("pressao_mb")),
            "temp_trend":   trend,
        })

    if not rows_data:
        raise CGESPError("History table parsed but no data rows found")

    df = pd.DataFrame(rows_data)
    df["timestamp"] = df["raw_date"].apply(_parse_pt_datetime)
    df = df.drop(columns=["raw_date"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Summary page parser  (estacao.jsp)
# ──────────────────────────────────────────────────────────────────────────────

def _kv_section(section_td: Optional[Tag]) -> dict[str, str]:
    """
    Extract label→value pairs from an inner .dados_estacoes table cell.
    Returns {} if the section is missing.
    """
    if section_td is None:
        return {}
    inner = section_td.find("table", class_="dados_estacoes")
    if not inner:
        return {}
    result = {}
    for tr in inner.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) == 2:
            label = cells[0].get_text(strip=True).rstrip(":")
            value = cells[1].get_text(strip=True)
            result[label] = value
    return result


def _num(s: Optional[str]) -> Optional[float]:
    """Extract the first numeric value from a string like '26.3 °C'."""
    if not s:
        return None
    m = re.search(r"[\d,\.]+", s)
    return _safe_float(m.group(0)) if m else None


def _parse_summary(html: str) -> dict:
    """
    Parse the current-conditions summary from estacao.jsp.

    The page has four column sections in a single outer table row:
      Chuva (Por Período*) | Temperatura | Umidade | Pressão

    Note: the page spells 'Máxima' as 'Mãxima' (HTML entity &atilde;).
    We accept both variants.

    Wind data (Velocidade/Rajada/Direção) is NOT present on the estacao.jsp
    page — it is only shown on some station variants. We return None for those
    fields when absent so the schema stays consistent.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Locate the outer instruments table
    outer = soup.find("table")
    if not outer:
        raise CGESPError("No instruments table found in summary page")

    headers = [th.get_text(strip=True) for th in outer.find_all("th")]
    data_rows = outer.find_all("tr")
    if len(data_rows) < 2:
        raise CGESPError("Summary table has no data row")

    sections = data_rows[1].find_all("td", recursive=False)

    def _section(keyword: str) -> Optional[Tag]:
        """Find the section <td> whose column header contains keyword."""
        kw = keyword.lower()
        for i, h in enumerate(headers):
            if kw in h.lower() and i < len(sections):
                return sections[i]
        return None

    chuva   = _kv_section(_section("chuva"))
    temp    = _kv_section(_section("temperatura"))
    umid    = _kv_section(_section("umidade"))
    pressao = _kv_section(_section("pressão") or _section("pressao"))

    # 'Mãxima' (real HTML) or 'Máxima' (user-facing) — accept both
    def _max(d: dict) -> Optional[str]:
        return d.get("Mãxima") or d.get("Máxima") or d.get("Maxima")

    def _min(d: dict) -> Optional[str]:
        return d.get("Mínima") or d.get("Minima")

    # Wind section — present on some station pages, absent on others
    vento = _kv_section(_section("vento"))

    return {
        "temp_atual_c":             _num(temp.get("Atual")),
        "temp_max_c":               _num(_max(temp)),
        "temp_min_c":               _num(_min(temp)),
        "umid_atual_pct":           _num(umid.get("Atual")),
        "umid_max_pct":             _num(_max(umid)),
        "umid_min_pct":             _num(_min(umid)),
        "pressao_atual_hpa":        _num(pressao.get("Atual")),
        "pressao_max_hpa":          _num(_max(pressao)),
        "pressao_min_hpa":          _num(_min(pressao)),
        "chuva_periodo_atual_mm":   _num(chuva.get("Per. Atual")),
        "chuva_periodo_anterior_mm":_num(chuva.get("Per. Anterior")),
        # Wind — None when not on page
        "vento_vel_kmh":            _num(vento.get("Velocidade")),
        "vento_rajada_kmh":         _num(vento.get("Rajada")),
        "vento_direcao":            vento.get("Direção") or vento.get("Direcao") or None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def fetch_station_history(channel_id: str, station_name: str) -> pd.DataFrame:
    """
    Fetch the 24h hourly history for one station.

    Returns a DataFrame with columns:
        timestamp, chuva_mm, vel_vt_ms, dir_vt_graus,
        temp_c, umid_rel_pct, pressao_mb, temp_trend,
        station, channel_id
    """
    time.sleep(REQUEST_DELAY)
    html = _get(HISTORY_URL, params={"WHICHCHANNEL": channel_id})
    df   = _parse_history(html)
    df["station"]    = station_name
    df["channel_id"] = channel_id
    return df


def fetch_station_summary(posto_id: str, station_name: str) -> dict:
    """
    Fetch the current-conditions summary for one station.

    Returns a dict with current/max/min values and station metadata.
    """
    time.sleep(REQUEST_DELAY)
    html    = _get(SUMMARY_URL, params={"POSTO": posto_id})
    summary = _parse_summary(html)
    summary["station"]    = station_name
    summary["posto_id"]   = posto_id
    summary["fetched_at"] = datetime.utcnow().isoformat()
    return summary


def fetch_all_stations_history(station_map: dict[str, str]) -> pd.DataFrame:
    """
    Fetch 24h history for multiple stations and concatenate.
    Partial failures are logged as warnings; at least one must succeed.
    """
    dfs: list[pd.DataFrame] = []
    for name, cid in station_map.items():
        try:
            df = fetch_station_history(cid, name)
            dfs.append(df)
            logger.info("✓ History fetched for '%s' (%d rows)", name, len(df))
        except Exception as exc:
            logger.warning("✗ History fetch failed for '%s': %s", name, exc)

    if not dfs:
        raise CGESPError("No station history data could be fetched")
    return pd.concat(dfs, ignore_index=True)


def fetch_all_stations_summary(posto_map: dict[str, str]) -> pd.DataFrame:
    """
    Fetch current summary for multiple stations and return as DataFrame.
    Partial failures are logged as warnings; at least one must succeed.
    """
    rows: list[dict] = []
    for name, pid in posto_map.items():
        try:
            row = fetch_station_summary(pid, name)
            rows.append(row)
            logger.info("✓ Summary fetched for '%s'", name)
        except Exception as exc:
            logger.warning("✗ Summary fetch failed for '%s': %s", name, exc)

    if not rows:
        raise CGESPError("No station summary data could be fetched")
    return pd.DataFrame(rows)