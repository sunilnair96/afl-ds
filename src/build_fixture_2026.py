import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup, NavigableString

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

YEAR = 2026
URL = f"https://www.footywire.com/afl/footy/ft_match_list?year={YEAR}"

OUTPUT_PATH = r"C:\Users\sunil\Projects\2026ws\afl-ds\data\raw\fixture_2026.xlsx"

TEAMS = [
    "ADE", "BRL", "CAR", "COL", "ESS", "FRE", "GCS", "GEE", "GWS",
    "HAW", "MEL", "NTH", "PTA", "RIC", "STK", "SYD", "WBD", "WCE"
]

# Map Footywire full names -> short codes
FULL_NAME_TO_CODE = {
    "Adelaide": "ADE",
    "Adelaide Crows": "ADE",

    "Brisbane": "BRL",
    "Brisbane Lions": "BRL",

    "Carlton": "CAR",

    "Collingwood": "COL",

    "Essendon": "ESS",

    "Fremantle": "FRE",
    "Fremantle Dockers": "FRE",

    "Gold Coast": "GCS",
    "Gold Coast Suns": "GCS",

    "Geelong": "GEE",
    "Geelong Cats": "GEE",

    "GWS": "GWS",
    "GWS Giants": "GWS",
    "Greater Western Sydney": "GWS",

    "Hawthorn": "HAW",
    "Hawthorn Hawks": "HAW",

    "Melbourne": "MEL",
    "Melbourne Demons": "MEL",

    "North Melbourne": "NTH",
    "North Melbourne Kangaroos": "NTH",

    "Port Adelaide": "PTA",
    "Port Adelaide Power": "PTA",

    "Richmond": "RIC",
    "Richmond Tigers": "RIC",

    "St Kilda": "STK",
    "St Kilda Saints": "STK",

    "Sydney": "SYD",
    "Sydney Swans": "SYD",

    "Western Bulldogs": "WBD",
    "Bulldogs": "WBD",

    "West Coast": "WCE",
    "West Coast Eagles": "WCE",
}

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def map_team_name_to_code(name: str) -> str:
    """Map a full team name (as shown on Footywire) to a short code."""
    name = name.strip()
    if name in FULL_NAME_TO_CODE:
        return FULL_NAME_TO_CODE[name]

    # Fallback: case-insensitive match
    for full, code in FULL_NAME_TO_CODE.items():
        if name.lower() == full.lower():
            return code

    raise ValueError(f"Unknown team name for mapping: '{name}'")


def fetch_fixture_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.text


def find_round_for_row(row_tag, round_pattern):
    """
    Walk backwards in the HTML from this row until we find text like 'Round 0',
    'Round 1', etc. Return the round number (int) or None.
    """
    for prev in row_tag.previous_elements:
        if isinstance(prev, NavigableString):
            text = prev.strip()
            if not text:
                continue
            m = round_pattern.search(text)
            if m:
                return int(m.group(1))
    return None


def parse_fixture(html: str) -> pd.DataFrame:
    """
    Parse the Footywire fixture HTML and return a DataFrame with:
        Round, Home, Away

    Uses any table where a cell looks like 'Sydney v Carlton', etc.
    """
    soup = BeautifulSoup(html, "html.parser")
    round_pattern = re.compile(r"Round\s+(\d+)")

    rows = []

    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue

            match_cell = None
            match_text = None

            # Find a cell that looks like 'TeamA v TeamB'
            for td in tds:
                text = td.get_text(" ", strip=True)
                if " v " in text:
                    parts = re.split(r"\s+v\s+", text)
                    if len(parts) == 2:
                        match_cell = td
                        match_text = text
                        break

            if match_cell is None:
                continue

            home_name, away_name = re.split(r"\s+v\s+", match_text)
            try:
                home_code = map_team_name_to_code(home_name)
                away_code = map_team_name_to_code(away_name)
            except ValueError:
                # Not a real AFL match row
                continue

            round_no = find_round_for_row(tr, round_pattern)
            if round_no is None:
                # Couldn't figure out which round this match belongs to
                continue

            rows.append(
                {
                    "Round": round_no,
                    "Home": home_code,
                    "Away": away_code,
                }
            )

    if not rows:
        print("No fixture rows parsed from HTML.")
        return pd.DataFrame(columns=["Round", "Home", "Away"])

    df = pd.DataFrame(rows, columns=["Round", "Home", "Away"])
    df.sort_values(by=["Round"], inplace=True, ignore_index=True)
    return df


def save_fixture(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_excel(path, index=False)
    print(f"Fixture written to {path}")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    html = fetch_fixture_html(URL)
    df = parse_fixture(html)
    save_fixture(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
