"""Label-free sanity checks for the metric plumbing (no GUI, no hand labels).

Uses an existing pipeline tracks.json as a stand-in for both GT and prediction,
then perturbs the prediction in known ways and asserts the metrics react
correctly. This validates the tracks.json->MOT conversion + motmetrics wiring
before we trust any number computed on real hand-labeled ground truth.

Run from the repo root:  python eval/verify_metrics.py
"""

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `import mot_io` works
import metrics  # noqa: E402
import mot_io  # noqa: E402


def _val(summary, name, col):
    return float(summary.loc[name, col])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks",
                    default="annotated_replays/tracked_gameplay_sam2_default.tracks.json",
                    help="a pipeline tracks.json to use as the synthetic GT/pred")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    if not os.path.exists(args.tracks):
        sys.exit(f"tracks file not found: {args.tracks} (run a backbone first, or pass --tracks)")

    tmp = tempfile.mkdtemp(prefix="evalverify_")
    gt = os.path.join(tmp, "gt.txt")
    mot_io.tracks_json_to_mot(args.tracks, gt)
    rows = mot_io.read_mot(gt)
    frames = sorted({r.frame for r in rows})
    second_half = set(frames[len(frames) // 2:])
    ids = sorted({r.id for r in rows})
    print(f"[verify] {args.tracks}: {len(rows)} rows, {len(frames)} frames, ids {ids}")

    def score(pred_rows, name, iou=args.iou):
        p = os.path.join(tmp, name + ".txt")
        mot_io.write_mot(pred_rows, p)
        return metrics.evaluate(gt, {name: p}, iou=iou)

    checks = []

    # 1. self-consistency: identical pred -> perfect.
    s = score(rows, "identical")
    idf1, sw = _val(s, "identical", "idf1"), int(_val(s, "identical", "num_switches"))
    fp, fn = int(_val(s, "identical", "num_false_positives")), int(_val(s, "identical", "num_misses"))
    ok = abs(idf1 - 1.0) < 1e-6 and sw == 0 and fp == 0 and fn == 0
    checks.append(ok)
    print(f"[1] self-consistency      IDF1={idf1:.3f} sw={sw} fp={fp} fn={fn}  -> {'PASS' if ok else 'FAIL'}")

    # 2. frame-offset guard: a large index offset (the clip-vs-source bug) must
    # tank IDF1. (A 1-frame shift is too gentle on near-static footage.)
    shift = max(10, len(frames) // 2)
    s = score([r._replace(frame=r.frame + shift) for r in rows], "shifted")
    idf1 = _val(s, "shifted", "idf1")
    ok = idf1 < 0.9
    checks.append(ok)
    print(f"[2] frame-offset guard    IDF1={idf1:.3f} after +{shift}f (expect <0.9)    -> {'PASS' if ok else 'FAIL'}")

    # 3. id-swap: swap two ids over the second half -> switches detected.
    if len(ids) >= 2:
        a, b = ids[0], ids[1]
        swapped = [r._replace(id=(b if r.id == a else a)) if (r.frame in second_half and r.id in (a, b))
                   else r for r in rows]
        s = score(swapped, "swapped")
        sw, idf1 = int(_val(s, "swapped", "num_switches")), _val(s, "swapped", "idf1")
        ok = sw >= 2 and idf1 < 1.0
        checks.append(ok)
        print(f"[3] id-swap detection     switches={sw} IDF1={idf1:.3f} (expect sw>=2)  -> {'PASS' if ok else 'FAIL'}")
    else:
        print("[3] id-swap detection     skipped (need >=2 ids)")

    # 4. IoU polarity: a ~10%-of-box shift still matches at 0.5 but not at 0.95.
    jit = [r._replace(x=r.x + 0.1 * r.w, y=r.y + 0.1 * r.h) for r in rows]
    lo = _val(score(jit, "jit", iou=0.5), "jit", "idf1")
    hi = _val(score(jit, "jit95", iou=0.95), "jit95", "idf1")
    ok = lo > 0.9 and hi < lo
    checks.append(ok)
    print(f"[4] iou polarity          IDF1@0.5={lo:.3f} IDF1@0.95={hi:.3f}        -> {'PASS' if ok else 'FAIL'}")

    allok = all(checks)
    print("RESULT:", "ALL PASS ✅" if allok else "SOME FAILED ❌")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
