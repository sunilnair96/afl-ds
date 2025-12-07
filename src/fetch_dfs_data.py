import os
import time
import pandas as pd

import re  # add this import at the top of the file if not already there

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

OUTPUT_DIR = r"C:\Users\sunil\Projects\2026ws\afl-ds\data\dfs"

# Seasons to scrape for fantasy points
FANTASY_SEASONS = [2026]

# Season to scrape for match results
RESULTS_SEASONS = [2026]  # initial value as requested

TEAMS = [
    "ADE", "BRL", "CAR", "COL", "ESS", "FRE", "GCS", "GEE", "GWS",
    "HAW", "MEL", "NTH", "PTA", "RIC", "STK", "SYD", "WBD", "WCE"
]

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# SELENIUM SETUP
# -----------------------------------------------------------------------------

def init_driver(headless: bool = True) -> webdriver.Chrome:
    """
    Initialise a Chrome WebDriver.
    Requires chrome/chromium + matching chromedriver on PATH.
    """
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # If you use ChromeDriverManager, you can swap to that here.
    service = Service()  # assumes chromedriver is on PATH
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

import os  # you already have this
import pandas as pd  # already there

def build_round_opponent_labels(results_df: pd.DataFrame, team: str, max_round_index: int):
    """
    For a given team and results DataFrame, return a list of opponent
    codes for rounds 0..max_round_index.

    - If the team doesn't play in a round, label as 'BYE'.
    """
    games = results_df[(results_df["Home"] == team) | (results_df["Away"] == team)]

    labels = []
    for r in range(max_round_index + 1):
        subset = games[games["Round"] == r]
        if subset.empty:
            labels.append("BYE")
        else:
            row = subset.iloc[0]
            opponent = row["Away"] if row["Home"] == team else row["Home"]
            labels.append(opponent)

    return labels


def rename_fantasy_round_columns_with_opponents(df: pd.DataFrame, season: int, team: str) -> pd.DataFrame:
    """
    Rename the round columns in the fantasy points DataFrame so that:

        PLAYER, AVG, ADJ, 100, 120, PPM, <round 0>, <round 1>, ...

    become:

        PLAYER, AVG, ADJ, 100, 120, PPM, <opp in R0 or BYE>, <opp in R1>, ...

    Uses the saved afl_results_<season>.xlsx file in OUTPUT_DIR.
    """
    results_path = os.path.join(OUTPUT_DIR, f"afl_results_{season}.xlsx")
    if not os.path.exists(results_path):
        print(f"Results file {results_path} not found; leaving fantasy headers unchanged.")
        return df

    results_df = pd.read_excel(results_path)

    # First 6 columns are fixed: PLAYER, AVG, ADJ, 100, 120, PPM
    fixed_cols = list(df.columns[:6])
    num_round_cols = max(0, len(df.columns) - 6)
    if num_round_cols == 0:
        return df

    # Build opponent labels for rounds 0..num_round_cols-1
    labels = build_round_opponent_labels(results_df, team, num_round_cols - 1)

    # Make sure lengths match
    if len(labels) < num_round_cols:
        labels = labels + ["BYE"] * (num_round_cols - len(labels))
    labels = labels[:num_round_cols]

    new_columns = fixed_cols + labels
    df.columns = new_columns

    return df

# -----------------------------------------------------------------------------
# SCRAPE RESULTS (ROUND, HOME, HOMESCORE, AWAY, AWAYSCORE)
# -----------------------------------------------------------------------------

def scrape_results_for_season(driver: webdriver.Chrome, season: int) -> pd.DataFrame:
    """
    Scrape match results for a given season from DFS Australia.

    The page has headings like:
        'Opening Round', 'Round 1', 'Round 2', ...

    For each heading, the next <table> contains the games for that round.

    Output columns: Round, Home, HomeScore, Away, AwayScore
    - Round: 0 for 'Opening Round', 1 for 'Round 1', etc.
    """
    url = f"https://dfsaustralia.com/afl-results/?season={season}"
    print(f"Scraping results from {url}")
    driver.get(url)

    # Wait for JS to load the content
    time.sleep(5)

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")

    round_pattern = re.compile(r"^(Opening Round|Round\s+\d+)\s*$")

    rows = []
    seen_tables = set()  # avoid using same table twice

    # Find all text nodes that look like a round heading
    for text_node in soup.find_all(string=round_pattern):
        heading_text = text_node.strip()

        # Map heading text to numeric round
        if heading_text.startswith("Opening Round"):
            round_no = 0
        else:
            m = re.search(r"\d+", heading_text)
            if not m:
                continue
            round_no = int(m.group(0))

        heading_tag = text_node.parent
        table = heading_tag.find_next("table")
        if table is None:
            continue

        # Avoid processing the same table multiple times
        if id(table) in seen_tables:
            continue
        seen_tables.add(id(table))

        tbody = table.find("tbody") or table

        for tr in tbody.find_all("tr"):
            tds = tr.find_all("td")
            # Expect at least the columns:
            # MCH, VENUE, HOME, ODDS, LINE, SCORE, AWAY, ODDS, LINE, SCORE, ...
            if len(tds) < 10:
                continue

            home = tds[2].get_text(strip=True)
            home_score = tds[5].get_text(strip=True)
            away = tds[6].get_text(strip=True)
            away_score = tds[9].get_text(strip=True)

            # Skip any weird header rows accidentally caught
            if home.upper() == "HOME" and away.upper() == "AWAY":
                continue

            rows.append(
                {
                    "Round": round_no,
                    "Home": home,
                    "HomeScore": home_score,
                    "Away": away,
                    "AwayScore": away_score,
                }
            )

    if not rows:
        print(f"No usable result tables found for season {season}")
        return pd.DataFrame(columns=["Round", "Home", "HomeScore", "Away", "AwayScore"])

    df = pd.DataFrame(rows, columns=["Round", "Home", "HomeScore", "Away", "AwayScore"])

    # Scores are like '13.4.82' – keep as text for now.
    # If you later want just the total points, you can split on '.' and take the last part.

    return df

def save_results_for_season(driver: webdriver.Chrome, season: int, output_dir: str = OUTPUT_DIR) -> None:
    df = scrape_results_for_season(driver, season)
    if df.empty:
        print(f"No results to save for season {season}")
        return

    output_file = os.path.join(output_dir, f"afl_results_{season}.xlsx")
    df.to_excel(output_file, index=False)
    print(f"Results for season {season} saved to {output_file}")


# -----------------------------------------------------------------------------
# SCRAPE FANTASY POINTS (PLAYER STATS)
# -----------------------------------------------------------------------------

def parse_fantasy_table(table) -> pd.DataFrame:
    """
    Given the <table id='fantasyPoints'> element, parse it into a DataFrame.

    This function:
      - Reads header cells
      - Tries to replace <img> headers with some text (alt or title)
      - Reads body rows
    """
    # Get headers – some may contain images
    headers = []
    for th in table.find_all("th"):
        # If there is an image, use alt or title if possible
        img = th.find("img")
        if img:
            alt_text = img.get("alt") or img.get("title") or ""
            headers.append(alt_text.strip())
        else:
            headers.append(th.get_text(strip=True))

    # Rows
    tbody = table.find("tbody")
    rows = []
    if tbody:
        for row in tbody.find_all("tr"):
            cells = [cell.get_text(strip=True) for cell in row.find_all("td")]
            if cells:
                rows.append(cells)

    df = pd.DataFrame(rows, columns=headers)
    return df

def scrape_fantasy_points_for_team_season(
    driver: webdriver.Chrome,
    team: str,
    season: int
) -> pd.DataFrame:
    """
    Scrape the fantasy points table for a single team & season.
    """
    url = f"https://dfsaustralia.com/afl-fantasy-points/?team={team}&season={season}"
    print(f"Scraping fantasy points from {url}")
    driver.get(url)

    # Wait for page JS to load – adjust if needed
    time.sleep(5)

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")

    table = soup.find("table", {"id": "fantasyPoints"})
    if not table:
        print(f"Table with id 'fantasyPoints' not found for team {team}, season {season}")
        return pd.DataFrame()

    df = parse_fantasy_table(table)

    # Note:
    #  - Columns 0–28 (according to DFS layout) represent rounds.
    #  - DFS often uses team logos as headers; parse_fantasy_table()
    #    tries to convert logos to text via img alt/title. If the site
    #    doesn’t provide these, headers may be blank and you can adjust
    #    later, or enrich using the results file.

    # Use the results file to rename the round columns to opponent team codes
    df = rename_fantasy_round_columns_with_opponents(df, season, team)

    return df

def save_fantasy_points_for_all(
    driver: webdriver.Chrome,
    seasons,
    teams,
    output_dir: str = OUTPUT_DIR
) -> None:
    for season in seasons:
        for team in teams:
            df = scrape_fantasy_points_for_team_season(driver, team, season)
            if df.empty:
                print(f"No fantasy data for season {season}, team {team}")
                continue

            filename = f"afl_fantasy_points_{season}_{team}.xlsx"
            output_file = os.path.join(output_dir, filename)
            df.to_excel(output_file, index=False)
            print(f"Fantasy table for season {season}, team {team} saved to {output_file}")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    driver = None
    try:
        driver = init_driver(headless=True)

        # 1. Scrape & save results for all RESULTS_SEASONS
        for season in RESULTS_SEASONS:
            save_results_for_season(driver, season, OUTPUT_DIR)

        # 2. Scrape & save fantasy points for all seasons/teams
        save_fantasy_points_for_all(driver, FANTASY_SEASONS, TEAMS, OUTPUT_DIR)

    finally:
        if driver is not None:
            driver.quit()
            print("Selenium driver closed.")


if __name__ == "__main__":
    main()
