import os
import pandas as pd

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
DIR = r"C:\Users\sunil\Projects\2026ws\afl-ds\data\processed"
INPUT_FILE = "afl_fantasy_player_ceiling_2026.xlsx"
OUTPUT_FILE = "afl_fantasy_roundwise_projection_2026.xlsx"

INPUT_PATH = os.path.join(DIR, INPUT_FILE)
OUTPUT_PATH = os.path.join(DIR, OUTPUT_FILE)


def main():
    # -------------------------------------------------------------------------
    # 1. Read input file
    # -------------------------------------------------------------------------
    df = pd.read_excel(INPUT_PATH)

    # We assume the Excel columns are in the same order as:
    # A  B  C  D  E  F  G  H  I  J  K  L  M  N  O ... AO ... BO
    # i.e. 0-based indices:
    # A=0, B=1, C=2, D=3, E=4, F=5, G=6, H=7, I=8, J=9, K=10, L=11,
    # M=12 (DOB), N=13, O=14, ..., AO, ..., BO

    cols = list(df.columns)

    # -------------------------------------------------------------------------
    # 2. Rename E–J:
    #    E: "AVG" -> "AVG-2025"
    #    F: "GMS" -> "GMS-2025"
    #    G: "AVG" -> "AVG-2024"
    #    H: "GMS" -> "GMS-2024"
    #    I: "AVG" -> "AVG-2023"
    #    J: "GMS" -> "GMS-2023"
    #
    # NOTE: We rename by *position* (index) to avoid issues with duplicate names.
    # -------------------------------------------------------------------------
    if len(cols) >= 10:
        cols[4] = "AVG-2025"   # E
        cols[5] = "GMS-2025"   # F
        cols[6] = "AVG-2024"   # G
        cols[7] = "GMS-2024"   # H
        cols[8] = "AVG-2023"   # I
        cols[9] = "GMS-2023"   # J

    df.columns = cols

    # -------------------------------------------------------------------------
    # 3. Keep A–D, renamed E–J, K, L, M (DOB), N
    #    Then drop O–AP, but keep AQ–BO.
    #
    #    Practically:
    #    - Keep first 14 columns: indices 0..13 (A..N)
    #    - Keep the last 25 columns (AQ..BO) if present.
    #    This matches:
    #      * "Drop Columns O to AP"
    #      * "Keep Columns AO to BO"
    #
    #    So we treat the last 25 columns as AQ..BO.
    # -------------------------------------------------------------------------
    num_tail_cols = 25
    total_cols = len(df.columns)

    if total_cols > 14 + num_tail_cols:
        head_indices = list(range(0, 14))  # A..N (0..13)
        tail_start = total_cols - num_tail_cols
        tail_indices = list(range(tail_start, total_cols))
        keep_indices = head_indices + tail_indices
        df = df.iloc[:, keep_indices]
    else:
        # If the sheet has fewer columns than expected, just keep all
        # (failsafe so the script doesn't crash).
        pass

    # -------------------------------------------------------------------------
    # 4. Ensure DOB (Column M) stays as DD/MM/YYYY
    #    - Column M is index 12 in the original order (A=0,...,M=12),
    #      but we have already subset columns; still, the column name is "DOB".
    # -------------------------------------------------------------------------
    if "DOB" in df.columns:
        # Convert to datetime, then back to string "DD/MM/YYYY"
        df["DOB"] = pd.to_datetime(df["DOB"], dayfirst=True, errors="coerce").dt.strftime(
            "%d/%m/%Y"
        )

    # -------------------------------------------------------------------------
    # 5. Write output
    # -------------------------------------------------------------------------
    os.makedirs(DIR, exist_ok=True)
    df.to_excel(OUTPUT_PATH, index=False)
    print(f"Roundwise projection file written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
