import os
import glob
import pandas as pd

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

DATA_DIR = r"C:\Users\sunil\Projects\2026ws\afl-ds\data\dfs"
OUTPUT_FILE = r"C:\Users\sunil\Projects\2026ws\afl-ds\data\raw\afl_fantasy_player_summary.xlsx"

TEAMS = [
    "ADE", "BRL", "CAR", "COL", "ESS", "FRE", "GCS", "GEE", "GWS",
    "HAW", "MEL", "NTH", "PTA", "RIC", "STK", "SYD", "WBD", "WCE"
]

BASE_COLS = ["PLAYER", "AVG", "ADJ", "100", "120", "PPM"]


# -------------------------------------------------------------------
# MAIN LOGIC
# -------------------------------------------------------------------

def consolidate_player_stats():
    # Dict: player -> aggregated stats
    player_stats = {}

    # Find all fantasy points files
    pattern = os.path.join(DATA_DIR, "afl_fantasy_points_*.xlsx")
    files = glob.glob(pattern)

    if not files:
        print(f"No files found matching {pattern}")
        return

    for file_path in files:
        print(f"Processing {file_path}")
        df = pd.read_excel(file_path)

        # Drop BYE columns
        df = df.loc[:, df.columns != "BYE"]

        # Ensure expected base columns exist
        missing = [c for c in BASE_COLS if c not in df.columns]
        if missing:
            print(f"Warning: {file_path} is missing columns {missing}, skipping.")
            continue

        # Identify opponent columns (all columns after BASE_COLS)
        opp_cols = [c for c in df.columns if c not in BASE_COLS]

        # Convert numeric columns where possible
        for col in BASE_COLS[1:] + opp_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for _, row in df.iterrows():
            player = row["PLAYER"]
            if pd.isna(player) or str(player).strip() == "":
                continue

            player = str(player).strip()

            if player not in player_stats:
                player_stats[player] = {
                    "avg_sum": 0.0,
                    "adj_sum": 0.0,
                    "sum_100": 0.0,
                    "sum_120": 0.0,
                    "ppm_sum": 0.0,
                    "row_count": 0,
                    "opp_sum": {},    # opp -> sum of scores
                    "opp_count": {},  # opp -> number of games
                }

            p = player_stats[player]

            # Aggregate base columns
            for col, key in [
                ("AVG", "avg_sum"),
                ("ADJ", "adj_sum"),
                ("100", "sum_100"),
                ("120", "sum_120"),
                ("PPM", "ppm_sum"),
            ]:
                val = row.get(col)
                if pd.notna(val):
                    p[key] += float(val)

            p["row_count"] += 1

            # Aggregate vs-opponent scores (each cell = one game)
            for col in opp_cols:
                opp_team = col  # header is team code
                val = row[col]
                if pd.isna(val):
                    continue
                try:
                    score = float(val)
                except (TypeError, ValueError):
                    continue

                p["opp_sum"][opp_team] = p["opp_sum"].get(opp_team, 0.0) + score
                p["opp_count"][opp_team] = p["opp_count"].get(opp_team, 0) + 1

    # Build final DataFrame
    rows = []
    for player, p in player_stats.items():
        rc = max(p["row_count"], 1)  # avoid divide-by-zero

        row = {
            "PLAYER": player,
            "avg": p["avg_sum"] / rc,
            "adj": p["adj_sum"] / rc,
            "avg(100)": p["sum_100"] / rc,
            "avg(120)": p["sum_120"] / rc,
            "avg(ppm)": p["ppm_sum"] / rc,
        }

        # Average vs each opponent team
        for team in TEAMS:
            s = p["opp_sum"].get(team, 0.0)
            c = p["opp_count"].get(team, 0)
            row[team] = (s / c) if c > 0 else None

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    # Sort by player name for readability
    summary_df.sort_values(by="PLAYER", inplace=True)

    # Save to Excel
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    summary_df.to_excel(OUTPUT_FILE, index=False)
    print(f"Summary written to {OUTPUT_FILE}")


if __name__ == "__main__":
    consolidate_player_stats()
