import os
import pandas as pd

import numpy as np
import math

# -----------------------------
# CONFIG – CHANGE IF NEEDED
# -----------------------------
RAW_DIR = r"C:\Users\sunil\Projects\2026ws\afl-ds\data\raw"
OUTPUT_DIR = r"C:\Users\sunil\Projects\2026ws\afl-ds\data\processed"
OUTPUT_FILE = "afl_fantasy_player_summary_2026.xlsx"

POSITIONS_FILE = "Positions2026withAvgs.xlsx"            # master list of all players
SUMMARY_FILE = "afl_fantasy_player_summary.xlsx"         # per-player vs-opponent averages
FIXTURE_FILE = "fixture_2026.xlsx"                       # fixture with Round, Home, Away

# If the sheet name is exactly "Positions 2025 with Avgs", we try that first
POSITIONS_SHEET = "Positions 2025 with Avgs"


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def make_key(x):
    """Normalise player name for joining."""
    if pd.isna(x):
        return None
    return str(x).strip().upper()


def load_positions(path, sheet_name=None):
    """
    Load Positions2026withAvgs and use the first data row as the actual header,
    like in your sample file.
    """
    # Try the given sheet name first, fall back to default
    if sheet_name:
        try:
            df = pd.read_excel(path, sheet_name=sheet_name)
        except Exception:
            df = pd.read_excel(path)
    else:
        df = pd.read_excel(path)

    # First row contains the real headers (Player, Team, Position, etc.)
    header_row = df.iloc[0]
    df = df.iloc[1:].copy()
    df.columns = header_row

    # Drop any completely empty rows (no Player)
    if "Player" in df.columns:
        df = df[df["Player"].notna()].copy()
    else:
        raise ValueError("Expected a 'Player' column in positions file after header fix.")

    # Add join key
    df["PLAYER_KEY"] = df["Player"].apply(make_key)

    return df


def load_summary(path):
    """Load afl_fantasy_player_summary and create join key."""
    df = pd.read_excel(path)
    if "PLAYER" not in df.columns:
        raise ValueError("Expected a 'PLAYER' column in afl_fantasy_player_summary.")
    df["PLAYER_KEY"] = df["PLAYER"].apply(make_key)
    return df


def load_fixture(path):
    """Load fixture_2026 with columns Round, Home, Away."""
    df = pd.read_excel(path)
    expected_cols = {"Round", "Home", "Away"}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"Expected columns {expected_cols} in fixture file.")
    # Ensure Round is integer
    df["Round"] = df["Round"].astype(int)
    df["Home"] = df["Home"].astype(str).str.upper()
    df["Away"] = df["Away"].astype(str).str.upper()
    return df


# -----------------------------
# MAIN LOGIC
# -----------------------------
def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build full paths
    positions_path = os.path.join(RAW_DIR, POSITIONS_FILE)
    summary_path = os.path.join(RAW_DIR, SUMMARY_FILE)
    fixture_path = os.path.join(RAW_DIR, FIXTURE_FILE)

    # Load inputs
    positions_df = load_positions(positions_path, sheet_name=POSITIONS_SHEET)
    summary_df = load_summary(summary_path)
    fixture_df = load_fixture(fixture_path)

    # Left join: keep ALL columns from positions, add columns from summary
    merged = positions_df.merge(
        summary_df,
        on="PLAYER_KEY",
        how="left",
        suffixes=("", "_stats")
    )

    # We need the five columns from afl_fantasy_player_summary
    # They will now be in 'merged' directly (avg, adj, avg(100), avg(120), avg(ppm))
    required_cols = ["avg", "adj", "avg(100)", "avg(120)", "avg(ppm)"]
    for col in required_cols:
        if col not in merged.columns:
            # If missing, create as NaN column to avoid errors
            merged[col] = pd.NA

    # Build (Round, Team) -> Opponent map from fixture
    fixture_long = pd.concat(
        [
            fixture_df.rename(columns={"Home": "Team", "Away": "Opponent"})[["Round", "Team", "Opponent"]],
            fixture_df.rename(columns={"Away": "Team", "Home": "Opponent"})[["Round", "Team", "Opponent"]],
        ],
        ignore_index=True,
    )

    fixture_map = {
        (int(r), str(t)): str(o)
        for r, t, o in fixture_long[["Round", "Team", "Opponent"]].itertuples(index=False)
    }

    # Determine rounds range from fixture
    min_round = int(fixture_df["Round"].min())
    max_round = int(fixture_df["Round"].max())

    # Function to get per-round score for a single row
    def get_round_score(row, rnd):
        team = row.get("Team")
        if pd.isna(team):
            return 0.0

        team = str(team).upper()
        opp = fixture_map.get((rnd, team))

        # If the team does not play in this round (bye), return 0
        if not opp:
            return 0.0

        # Opponent column must exist in merged – columns like ADE, BRL, etc.
        if opp not in merged.columns:
            return 0.0

        val = row.get(opp)
        if pd.isna(val):
            return 0.0

        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    # Add columns "round 0" to "round 24" (or whatever min/max are)
    for rnd in range(min_round, max_round + 1):
        col_name = f"round {rnd}"  # as requested: columns round 0 to round 24
        merged[col_name] = merged.apply(lambda r, rr=rnd: get_round_score(r, rr), axis=1)

    # -------------------------------------------------
    # FORMAT DOB FIELD TO DD/MM/YYYY
    # -------------------------------------------------
    if "DOB" in merged.columns:
        merged["DOB"] = pd.to_datetime(merged["DOB"], errors="coerce").dt.strftime("%d/%m/%Y")

    # -------------------------------------------------
    # ROUND UP ALL COLUMNS FROM COLUMN "ADE" ONWARDS
    # -------------------------------------------------
    if "ADE" not in merged.columns:
        raise ValueError("Column 'ADE' not found — verify exact spelling in Positions file.")

    # Find start index
    start_idx = merged.columns.get_loc("ADE")

    # Round up all numeric columns from ADE forward
    for col in merged.columns[start_idx:]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").apply(
            lambda x: math.ceil(x) if pd.notna(x) else x
        )

    # Write to Excel
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    merged.to_excel(output_path, index=False)
    print(f"Fixture-based summary written to: {output_path}")


if __name__ == "__main__":
    main()
