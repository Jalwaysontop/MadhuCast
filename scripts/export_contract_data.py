"""Write the dashboard's per-patient files (R3.1-R3.3) from Person 3's pipeline.

    python scripts/export_contract_data.py            # -> data/processed/{patient_profiles,timeseries,meals}/
    MADHUTWIN_DATA=real streamlit run dashboard/app.py

Uses data/raw/cgmacros/ when present, otherwise P3's seeded demo cohort (all values then SYNTHETIC).
See src/serving/p3_export.py for the conversions and docs/decisions.md ADR-016 for why.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serving import p3_export  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=p3_export.CGMACROS_DIR, help="CGMacros raw directory")
    ap.add_argument("--out", type=Path, default=p3_export.OUT_DIR, help="output root (data/processed)")
    ap.add_argument("--seed", type=int, default=42, help="P3 pipeline seed (EHR, pairing, augmentation)")
    ap.add_argument("--ehr", type=int, default=100, help="synthetic EHR records to pair from")
    ap.add_argument("--augment", type=float, default=0.30, help="fraction of meals given Indian substitutes")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    exports, real = p3_export.build(args.raw, args.seed, args.ehr, args.augment)
    p3_export.write(exports, args.out)
    print(f"Wrote {len(exports)} patients to {args.out} ({'CGMacros' if real else 'P3 demo cohort, SYNTHETIC'}).")
    for e in exports:
        p = e.profile
        paired = f"EHR {p.pairing['synthetic_ehr_id']}" if p.pairing else "own record"
        n_ind = sum(m.items is not None for m in e.meals)
        print(
            f"  {e.pid} <- {e.source_id}: {p.cohort}, {p.history_days:g} d, {len(e.meals)} meals "
            f"({n_ind} Indian), {int(e.history['missing_mask'].sum())} missing CGM bins, {paired}"
        )


if __name__ == "__main__":
    main()
