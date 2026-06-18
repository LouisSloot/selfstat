"""Pool per-clip eval results (eval/clips/*/results.csv) into one table.

Per-clip rate metrics (IDF1/MOTA/IDR) are averaged unweighted across clips; error
counts are summed; MOTA is also reported "pooled" — recomputed from the summed
error counts, which weights by clip size. One firm number across the eval set.

    python eval/aggregate.py
"""

import argparse
import glob
import os

import pandas as pd

COLS = ["idf1", "mota", "idr", "num_switches",
        "num_false_positives", "num_misses", "num_objects"]


def load():
    rows = []
    for p in sorted(glob.glob("eval/clips/*/results.csv")):
        clip = os.path.basename(os.path.dirname(p))
        df = pd.read_csv(p, index_col=0)
        for backbone, r in df.iterrows():
            rows.append({"clip": clip, "backbone": backbone,
                         **{c: r[c] for c in COLS}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None, help="optional path to dump the long table")
    args = ap.parse_args()

    df = load()
    if df.empty:
        print("no results found — run eval/run_eval.py on some clips first")
        return
    clips = sorted(df["clip"].unique())
    backbones = sorted(df["backbone"].unique())

    print(f"=== per-clip IDF1 / MOTA  ({len(clips)} clips) ===")
    idf1 = df.pivot(index="clip", columns="backbone", values="idf1")
    mota = df.pivot(index="clip", columns="backbone", values="mota")
    for c in clips:
        parts = "   ".join(f"{b}: IDF1={idf1.loc[c, b]:.3f} MOTA={mota.loc[c, b]:.3f}"
                           for b in backbones)
        print(f"  {c}:  {parts}")

    print("\n=== aggregate across clips ===")
    hdr = f"{'backbone':<10}{'IDF1(mean)':>12}{'MOTA(mean)':>12}{'MOTA(pool)':>12}" \
          f"{'IDR(mean)':>11}{'IDsw':>6}{'FP':>5}{'FN':>5}"
    print(hdr)
    for b in backbones:
        g = df[df["backbone"] == b]
        sw, fp, fn = int(g.num_switches.sum()), int(g.num_false_positives.sum()), int(g.num_misses.sum())
        pooled = 1 - (fp + fn + sw) / g.num_objects.sum()
        print(f"{b:<10}{g.idf1.mean():>12.3f}{g.mota.mean():>12.3f}{pooled:>12.3f}"
              f"{g.idr.mean():>11.3f}{sw:>6}{fp:>5}{fn:>5}")

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
