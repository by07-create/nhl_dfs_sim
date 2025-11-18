import re
import json
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

URL_TEMPLATE = "https://5v5hockey.com/line-matchups-detail-embedded/?date={date}"
OUTFILE = Path("5v5_matchups.csv")


def fetch_html(date: str):
    url = URL_TEMPLATE.format(date=date)
    print(f" Fetching: {url}")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text


def extract_matchup_blocks(html: str):
    pattern = r"JSON\.stringify\((\[.*?\])\)"
    blocks = re.findall(pattern, html, flags=re.DOTALL)
    print(f" Found {len(blocks)} matchup JSON blocks.")
    return blocks


def js_to_json(js_block: str) -> str:
    js_block = js_block.replace("'", '"')
    js_block = js_block.replace("True", "true").replace("False", "false")
    return js_block


def parse_matchups(blocks):
    rows = []

    for block in blocks:
        try:
            json_str = js_to_json(block)
            arr = json.loads(json_str)
            rows.extend(arr)
        except Exception as e:
            print(f" JSON parse error: {e}")
            print("Block preview:")
            print(block[:200])
            continue

    return rows


def save_csv(rows):
    df = pd.DataFrame(rows)
    df.to_csv(OUTFILE, index=False)
    print(f" Saved matchups -> {OUTFILE.resolve()}")
    print(df.head())
    return df


def main():
    #  Automatically use today's date (YYYY-MM-DD)
    date = datetime.now().strftime("%Y-%m-%d")

    html = fetch_html(date)
    blocks = extract_matchup_blocks(html)
    rows = parse_matchups(blocks)
    save_csv(rows)


if __name__ == "__main__":
    main()
