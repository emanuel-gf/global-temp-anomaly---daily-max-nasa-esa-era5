"""
retrieve_metar.py
=================
Fetch METAR observations via AVWX (current) or AWC (historical).

Usage examples
--------------
# Latest observation, printed as Telegram-ready text
python retrieve_metar.py --station_id EGLC

# Last 12 hours, printed as JSON (no file written)
python retrieve_metar.py --station_id EGLC --hours_before 12 --output_format json

# Save to CSV
python retrieve_metar.py --station_id EGLC --hours_before 24 \
    --output_path ./data --endswith csv

# Save to Parquet
python retrieve_metar.py --station_id EGLC --hours_before 6 \
    --output_path ./data --endswith parquet

# Upload to GCS bucket (needs google-cloud-storage)
python retrieve_metar.py --station_id EGLC --hours_before 6 \
    --output_path gs://my-bucket/metar --endswith parquet

Environment variables
---------------------
AVWX_TOKEN  : Your avwx.rest API token (required for AVWX current endpoint).
              Get one free at https://account.avwx.rest/
              If not set, falls back to AWC for all requests.

Dependencies
------------
pip install requests python-dotenv
pip install pandas pyarrow          # for --endswith parquet
pip install google-cloud-storage    # only if writing to GCS
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()  # loads .env if present

# ── constants ────────────────────────────────────────────────────────────────
AVWX_BASE   = "https://avwx.rest/api/metar"
AWC_BASE    = "https://aviationweather.gov/api/data/metar"
USER_AGENT  = "retrieve-metar-pipeline/1.0"

# ── helpers ──────────────────────────────────────────────────────────────────

def _avwx_headers() -> dict:
    token = os.getenv("AVWX_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "AVWX_TOKEN not set. Export it or add it to a .env file.\n"
            "Get a free token at: https://account.avwx.rest/"
        )
    return {"Authorization": token, "User-Agent": USER_AGENT}


def fetch_current_avwx(station_id: str) -> dict:
    """
    Fetch the latest METAR from AVWX with translate + summary + speech options.
    Returns a single-record dict ready for display or downstream use.
    Requires AVWX_TOKEN.
    """
    url = f"{AVWX_BASE}/{station_id.upper()}"
    params = {"options": "translate,summary,speech", "format": "json"}
    resp = requests.get(url, params=params, headers=_avwx_headers(), timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    # flatten into a flat dict (memory-efficient, no nested objects kept)
    record = _flatten_avwx_record(raw)
    return record


def fetch_history_awc(station_id: str, hours_before: int) -> list[dict]:
    """
    Fetch up to `hours_before` hours of METAR history from AWC (free, no key).
    Returns a list of flat dicts.
    AWC keeps up to 15 days; rate-limit: 100 req/min.
    """
    params = {
        "ids":    station_id.upper(),
        "format": "json",
        "hours":  hours_before,
    }
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(AWC_BASE, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    raw_list = resp.json()

    records = [_flatten_awc_record(r) for r in raw_list]
    return records


# ── field extraction ─────────────────────────────────────────────────────────

def _flatten_avwx_record(r: dict) -> dict:
    """Extract the useful scalar fields from an AVWX METAR response."""
    def _val(field):
        """AVWX wraps numbers as {"value": x, "spoken": "..."}"""
        if isinstance(field, dict):
            return field.get("value")
        return field

    clouds = r.get("clouds") or []
    cloud_str = ", ".join(
        f"{c.get('type','?')} {c.get('altitude','?')}ft"
        for c in clouds if isinstance(c, dict)
    ) or "NCD"

    wx_codes = r.get("wx_codes") or []
    wx_str = " ".join(
        w.get("value", "") for w in wx_codes if isinstance(w, dict)
    ) or "None"

    return {
        "source":         "AVWX",
        "station":        r.get("station"),
        "raw":            r.get("raw"),
        "sanitized":      r.get("sanitized"),
        "time_utc":       r.get("time", {}).get("dt") if isinstance(r.get("time"), dict) else None,
        "flight_rules":   r.get("flight_rules"),
        "summary":        r.get("summary"),
        "speech":         r.get("speech"),
        # numeric fields
        "temp_c":         _val(r.get("temperature")),
        "dewpoint_c":     _val(r.get("dewpoint")),
        "wind_dir_deg":   _val(r.get("wind_direction")),
        "wind_speed_kt":  _val(r.get("wind_speed")),
        "wind_gust_kt":   _val(r.get("wind_gust")),
        "visibility":     _val(r.get("visibility")),
        "altimeter_hpa":  _val(r.get("altimeter")),
        "clouds":         cloud_str,
        "wx_codes":       wx_str,
        # translations (plain English per field)
        "trans_wind":     (r.get("translate") or {}).get("wind"),
        "trans_visibility": (r.get("translate") or {}).get("visibility"),
        "trans_clouds":   (r.get("translate") or {}).get("clouds"),
        "trans_wx":       (r.get("translate") or {}).get("wx_codes"),
    }


def _flatten_awc_record(r: dict) -> dict:
    """Extract useful scalar fields from an AWC METAR JSON record."""
    return {
        "source":         "AWC",
        "station":        r.get("icaoId"),
        "raw":            r.get("rawOb"),
        "sanitized":      None,
        "time_utc":       r.get("obsTime"),
        "flight_rules":   None,
        "summary":        None,
        "speech":         None,
        "temp_c":         r.get("temp"),
        "dewpoint_c":     r.get("dewp"),
        "wind_dir_deg":   r.get("wdir"),
        "wind_speed_kt":  r.get("wspd"),
        "wind_gust_kt":   r.get("wgst"),
        "visibility":     r.get("visib"),
        "altimeter_hpa":  r.get("altim"),
        "clouds":         str(r.get("clouds", "")) or "NCD",
        "wx_codes":       r.get("wxString") or "None",
        "trans_wind":     None,
        "trans_visibility": None,
        "trans_clouds":   None,
        "trans_wx":       None,
    }


# ── formatters ───────────────────────────────────────────────────────────────

def format_telegram(records: list[dict]) -> str:
    """
    Build a compact, readable Telegram message.
    Uses monospace blocks so it renders nicely in Telegram.
    """
    lines = []
    for r in records:
        station   = r.get("station", "?")
        time_utc  = r.get("time_utc", "?")
        rules     = r.get("flight_rules") or "N/A"
        raw       = r.get("raw", "")
        summary   = r.get("summary") or (
            f"T:{r.get('temp_c')}°C "
            f"Dp:{r.get('dewpoint_c')}°C "
            f"Wind:{r.get('wind_dir_deg')}°/{r.get('wind_speed_kt')}kt "
            f"Vis:{r.get('visibility')} "
            f"QNH:{r.get('altimeter_hpa')}hPa "
            f"Clouds:{r.get('clouds')} "
            f"WX:{r.get('wx_codes')}"
        )

        lines.append(
            f"✈️ *{station}* | {time_utc} UTC\n"
            f"🛩 Flight rules: `{rules}`\n"
            f"📋 `{raw}`\n"
            f"📝 {summary}"
        )
    return "\n\n---\n\n".join(lines)


def format_json(records: list[dict]) -> str:
    return json.dumps(records, indent=2, default=str)


def format_dict(records: list[dict]) -> str:
    """Compact Python-dict-style string."""
    return str(records)


# ── persistence (only imported when actually saving) ─────────────────────────

def save_records(records: list[dict], output_path: str, endswith: str):
    """
    Save records to output_path with extension `endswith`.
    Supports: csv, parquet.
    Supports GCS paths: gs://bucket/prefix
    Pandas is imported ONLY here to avoid resident memory when not saving.
    """
    import pandas as pd  # late import — not loaded unless --output_path used

    df = pd.DataFrame(records)

    station  = records[0].get("station", "UNKNOWN") if records else "UNKNOWN"
    ts       = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"metar_{station}_{ts}.{endswith.lstrip('.')}"

    is_gcs = output_path.startswith("gs://")

    if is_gcs:
        # write to a tmp buffer then upload
        import io
        from google.cloud import storage  # late import

        bucket_path = output_path[5:]  # strip gs://
        bucket_name, *prefix_parts = bucket_path.split("/")
        prefix       = "/".join(prefix_parts)
        blob_name    = f"{prefix}/{filename}".lstrip("/")

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob   = bucket.blob(blob_name)

        if endswith in ("parquet",):
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            buf.seek(0)
            blob.upload_from_file(buf, content_type="application/octet-stream")
        else:  # csv
            blob.upload_from_string(df.to_csv(index=False), content_type="text/csv")

        print(f"[saved] gs://{bucket_name}/{blob_name}", file=sys.stderr)

    else:
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)

        if endswith == "parquet":
            df.to_parquet(full_path, index=False)
        else:  # csv
            df.to_csv(full_path, index=False)

        print(f"[saved] {full_path}", file=sys.stderr)


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="retrieve_metar.py",
        description="Fetch METAR data from AVWX / AWC. Memory-safe for small VMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--station_id", required=True,
        help="ICAO station code, e.g. EGLC, EGLL, KJFK"
    )
    p.add_argument(
        "--hours_before", type=int, default=0,
        help="How many hours of history to retrieve (0 = latest only). "
             "0 uses AVWX (rich parsed data); >0 uses AWC free API."
    )
    p.add_argument(
        "--output_format", choices=["telegram", "json", "dict"], default="telegram",
        help="How to print results to stdout when NOT saving to file. Default: telegram"
    )
    p.add_argument(
        "--output_path", default=None,
        help="Directory (local) or GCS path (gs://bucket/prefix) to save the file. "
             "When set, nothing is printed to stdout (RAM-safe)."
    )
    p.add_argument(
        "--endswith", choices=["csv", "parquet"], default="csv",
        help="File format when --output_path is given. Default: csv"
    )
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── fetch ────────────────────────────────────────────────────────────────
    try:
        if args.hours_before == 0:
            # Latest only — use AVWX for rich decoded output
            try:
                records = [fetch_current_avwx(args.station_id)]
            except EnvironmentError as e:
                # No token: fall back to AWC for the last 1 hour
                print(f"[warn] {e}\n[warn] Falling back to AWC for latest obs.", file=sys.stderr)
                records = fetch_history_awc(args.station_id, hours_before=1)
                records = records[:1]  # keep only most recent
        else:
            if args.hours_before > 24:
                parser.error("--hours_before max is 24 (AWC free endpoint limit per best practice)")
            records = fetch_history_awc(args.station_id, args.hours_before)

    except requests.HTTPError as e:
        print(f"[error] HTTP {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as e:
        print(f"[error] Network error: {e}", file=sys.stderr)
        sys.exit(1)

    if not records:
        print(f"[warn] No METAR records returned for {args.station_id}", file=sys.stderr)
        sys.exit(0)

    # ── output ───────────────────────────────────────────────────────────────
    if args.output_path:
        # Save to disk / GCS — don't print, keep RAM free
        save_records(records, args.output_path, args.endswith)
    else:
        # Print to stdout (Telegram, JSON, or dict) — no pandas loaded
        if args.output_format == "telegram":
            print(format_telegram(records))
        elif args.output_format == "json":
            print(format_json(records))
        else:
            print(format_dict(records))


if __name__ == "__main__":
    main()