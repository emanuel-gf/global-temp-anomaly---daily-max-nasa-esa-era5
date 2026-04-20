"""
Run this from your project root:
  python debug_history.py

It prints the raw HTML of the history iframe so we can see the real table structure.
"""
import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8",
    "Referer": "https://www.cgesp.org/",
}

r = requests.get(
    "https://www.cgesp.org/processo_cge.jsp",
    headers=headers,
    params={"WHICHCHANNEL": "1000887"},
    timeout=15,
)
print(f"Status: {r.status_code}, Length: {len(r.text)}")

# Save full HTML
with open("debug_history.html", "w", encoding="utf-8") as f:
    f.write(r.text)
print("Full HTML saved → debug_history.html")

# Print ALL tables found
soup = BeautifulSoup(r.text, "html.parser")
tables = soup.find_all("table")
print(f"\nFound {len(tables)} table(s)")

for t_idx, table in enumerate(tables):
    rows = table.find_all("tr")
    print(f"\n=== TABLE {t_idx} ({len(rows)} rows) ===")
    for r_idx, tr in enumerate(rows[:8]):  # first 8 rows per table
        cells = tr.find_all(["td", "th"])
        raw   = [c.get_text(strip=True) for c in cells]
        tags  = [c.name for c in cells]
        print(f"  row {r_idx:02d} [{len(cells)} cells] tags={tags}")
        print(f"           values={raw}")
        # also show raw HTML of first data row to catch colspan/rowspan
        if r_idx <= 2:
            print(f"           raw_html={str(tr)[:300]}")