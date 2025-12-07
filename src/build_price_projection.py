import os
import pandas as pd

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
DIR = r"C:\Users\sunil\Projects\2026ws\afl-ds\data\processed"

ROUNDWISE_FILE = "afl_fantasy_roundwise_projection_2026_Edited.xlsx"
MAGIC_FILE = "MagicNumber.xlsx"

OUTPUT_FILE = "afl_fantasy_206_price_projection.xlsx"

ROUNDWISE_PATH = os.path.join(DIR, ROUNDWISE_FILE)
MAGIC_PATH = os.path.join(DIR, MAGIC_FILE)
OUTPUT_PATH = os.path.join(DIR, OUTPUT_FILE)

# We only care about magic numbers / prices up to Round 23
MAX_ROUND_FOR_MAGIC = 23          # last round that changes price
MAX_ROUND_SCORE_COL = 24          # score columns exist for rounds 0..24

# Smoothing factor for blending current price towards target price
ALPHA = 0.13                      # tweakable if you want slightly different behaviour


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def load_magic_numbers(path: str, max_round: int = MAX_ROUND_FOR_MAGIC) -> dict:
    """
    Load magic numbers by round from MagicNumber.xlsx.

    Assumptions (tries these in order):
    1) Single-row layout with columns like "Round 0", "Round 1", ..., "Round 23"
    2) Two-column layout with columns containing ["round", "magic"]

    Returns dict: {0: mn0, 1: mn1, ..., max_round: mnN}
    """
    mdf = pd.read_excel(path)

    magic_by_round = {}

    # Case 1: single row, columns "Round 0", "Round 1", ...
    if mdf.shape[0] == 1:
        row = mdf.iloc[0]
        for col in mdf.columns:
            col_str = str(col).strip().lower()
            if col_str.startswith("round "):
                try:
                    r = int(col_str.split()[1])
                except ValueError:
                    continue
                if 0 <= r <= max_round:
                    magic_by_round[r] = row[col]

    # Case 2: 2 columns, one for round, one for magic
    if not magic_by_round:
        round_col_candidates = [
            c for c in mdf.columns if "round" in str(c).lower()
        ]
        magic_col_candidates = [
            c for c in mdf.columns if "magic" in str(c).lower()
        ]
        if round_col_candidates and magic_col_candidates:
            rcol = round_col_candidates[0]
            mcol = magic_col_candidates[0]
            for _, row in mdf.iterrows():
                try:
                    r = int(row[rcol])
                except Exception:
                    continue
                if 0 <= r <= max_round:
                    magic_by_round[r] = row[mcol]

    missing = [r for r in range(0, max_round + 1) if r not in magic_by_round]
    if missing:
        raise ValueError(
            f"Missing magic number(s) for rounds: {missing}. "
            f"Check MagicNumber.xlsx format."
        )

    return magic_by_round


def detect_price_column(df: pd.DataFrame) -> str:
    """
    Try to detect the 'Priced At' column (starting price basis).

    Looks for common variants case-insensitively.
    """
    lowered = {str(c).strip().lower(): c for c in df.columns}

    for key in ("priced at", "price at", "price", "starting price"):
        if key in lowered:
            return lowered[key]

    raise ValueError(
        "Could not detect a 'Priced At' column. "
        "Please rename it to something like 'Priced At' or 'Price'."
    )


def detect_round_columns(df: pd.DataFrame, max_round: int = MAX_ROUND_SCORE_COL) -> dict:
    """
    Build mapping {round_number: column_name} for round score columns.

    Assumes columns named like "round 0", "Round 1", ..., case-insensitive.
    We expect at least round 0..max_round to exist for scores.
    """
    round_cols = {}
    for col in df.columns:
        col_str = str(col).strip().lower()
        if col_str.startswith("round "):
            parts = col_str.split()
            if len(parts) == 2:
                try:
                    r = int(parts[1])
                except ValueError:
                    continue
                if 0 <= r <= max_round:
                    round_cols[r] = col

    missing = [r for r in range(0, max_round + 1) if r not in round_cols]
    if missing:
        raise ValueError(
            f"Missing round score column(s) for rounds: {missing}. "
            f"Expected columns like 'round 0', 'round 1', etc."
        )

    return round_cols


def weighted_last_five_average(played_scores):
    """
    Compute weighted average of last up to 5 played scores.

    played_scores: list of all past played scores in chronological order.
    We take the last up to 5, with weights 5,4,3,2,1 for
    most recent -> oldest.
    """
    if not played_scores:
        return None

    tail = played_scores[-5:]           # last up to 5 scores
    # assign weights: most recent gets 5, then 4,3,2,1
    weights = []
    values = []
    # iterate from newest to oldest within the tail
    for i, score in enumerate(reversed(tail)):
        w = 5 - i
        if w <= 0:
            break
        weights.append(w)
        values.append(score)

    if not weights:
        return None

    num = sum(w * s for w, s in zip(weights, values))
    den = sum(weights)
    return num / den


# -------------------------------------------------------------------
# MAIN LOGIC
# -------------------------------------------------------------------

def main():
    # 1. Load roundwise projections (player scores per round)
    df = pd.read_excel(ROUNDWISE_PATH)

    # 2. Load magic numbers for rounds 0..23 only
    magic_by_round = load_magic_numbers(MAGIC_PATH, max_round=MAX_ROUND_FOR_MAGIC)

    # 3. Detect the "Priced At" column
    price_col = detect_price_column(df)

    # 4. Detect round score columns (0..24 exist, but we'll only use 0..23 for prices)
    round_cols = detect_round_columns(df, max_round=MAX_ROUND_SCORE_COL)

    # 5. Prepare containers for price projections (from round 2 to 23)
    num_rows = len(df)
    price_projection = {r: [None] * num_rows for r in range(2, MAX_ROUND_FOR_MAGIC + 1)}

    # ----------------------------------------------------------------
    # Pricing model & assumptions
    # ----------------------------------------------------------------
    # - StartingPrice = PricedAt * MagicNumber(Round 0)
    # - Price does NOT change after Round 0.
    # - Price starts changing from Round 1.
    #
    # For each player:
    #   - Maintain list of played_scores (only score > 0)
    #   - For each round r (1..23):
    #       * Add score_r to played_scores if > 0
    #       * Compute weighted last-5 average A_r
    #       * TargetPrice_r = A_r * MagicNumber_r
    #       * P_{r+1} = round_to_nearest_1000(
    #               (1 - ALPHA) * P_r + ALPHA * TargetPrice_r
    #         )
    #
    # - A score of 0 (or NaN) is treated as DNP: no new score added to
    #   history; if no games played yet, price does not move.
    # - We calculate internal price from Round 1 to Round 23.
    # - We do NOT calculate price for Round 24 (season ends).
    # ----------------------------------------------------------------

    mn0 = magic_by_round[0]

    for idx, row in df.iterrows():
        priced_at = row[price_col]
        if pd.isna(priced_at):
            priced_at = 0

        # Starting price based on Round 0 magic number
        starting_price = priced_at * mn0
        current_price = float(starting_price)

        # Collect played scores in chronological order
        played_scores = []

        # Optional: include Round 0 score in history (if you want it contributing)
        if 0 in round_cols:
            s0 = row[round_cols[0]]
            if pd.notna(s0) and float(s0) > 0:
                played_scores.append(float(s0))

        # Loop rounds 1..24 (we stop price at 23, but scores exist to 24)
        for r in range(1, MAX_ROUND_SCORE_COL + 1):
            score_col = round_cols[r]
            score = row[score_col]

            # Treat NaN as 0
            if pd.isna(score):
                score = 0.0
            score = float(score)

            # For r > MAX_ROUND_FOR_MAGIC (i.e. r == 24), we do NOT change price
            # but we could still collect score in history if you want for later analysis.
            if r > MAX_ROUND_FOR_MAGIC:
                if score > 0:
                    played_scores.append(score)
                continue

            # If the player played this round, add to history
            if score > 0:
                played_scores.append(score)

            # If still no played games, price does not move
            if not played_scores:
                # P_{r+1} equals P_r; but we only store from round 2 onwards
                if r >= 2:
                    price_projection[r][idx] = round(current_price, 2)
                continue

            # Weighted last-5 average A_r using all played games up to this round
            A_r = weighted_last_five_average(played_scores)

            if A_r is None:
                # safety: if something weird, keep price unchanged
                if r >= 2:
                    price_projection[r][idx] = round(current_price, 2)
                continue

            magic_r = magic_by_round[r]

            # Target price based on weighted average
            target_price = A_r * magic_r

            # Blend current price towards target, then round to nearest $1,000
            raw_price = (1.0 - ALPHA) * current_price + ALPHA * target_price
            new_price = round(raw_price / 1000.0) * 1000.0

            current_price = new_price

            # Store price projection FROM round 2 onwards, up to 23
            if 2 <= r <= MAX_ROUND_FOR_MAGIC:
                price_projection[r][idx] = current_price

    # 6. Append StartingPrice column first (price before Round 1)
    df["StartingPrice"] = df[price_col] * magic_by_round[0]

    # 7. Append new price columns to df
    for r in range(2, MAX_ROUND_FOR_MAGIC + 1):
        col_name = f"Price R{r}"
        df[col_name] = price_projection[r]

    # 8. Write output
    os.makedirs(DIR, exist_ok=True)
    df.to_excel(OUTPUT_PATH, index=False)
    print(f"Price projection file written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
