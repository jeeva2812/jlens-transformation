"""Apply re-audit rules 1 (grounding) + 2 (bands; mid deferred) + 3 (steering).

Writes shards in place. Mid-band survivors are flagged (needs_1000=True) for
the null re-run; their verdict stays validated until jlens/reaudit_null.py decides.
"""
from __future__ import annotations
import json
import glob
from pathlib import Path
from jlens.reaudit import grounding

SHARDS = sorted(glob.glob("out/labels/*.json"))
SHARDS = [f for f in SHARDS if Path(f).name != "labels.json"]


def ratio_of(r):
    return abs(r["score"]) / max(r["null_threshold"], 1e-9)


def void_steer(r, why):
    s = r.get("steer") or {}
    prior = s.get("steers", "unsteered")
    r["steer"] = None
    return prior


def main():
    counts = {"ground_withdraw": 0, "lo_reject": 0, "mid_defer": 0, "hi_keep": 0,
              "steer_flipped": 0}
    mids = []
    for f in SHARDS:
        D = json.loads(Path(f).read_text())
        dirty = False
        for r in D:
            # strength on every record (300-draw basis for now)
            r["strength"] = round(ratio_of(r), 3) if r["null_threshold"] else 0.0
            if r["verdict"] != "validated":
                continue
            g = grounding(r["hypothesis"], r)
            if g < 4:
                prior_steer = void_steer(r, "grounding")
                r["verdict"] = "no-hypothesis"
                r["audit_note"] = (f"RE-AUDIT withdraw (rule 1): max-pole theme support {g}/15 < 4 "
                                   f"for {r['hypothesis']!r}; prior ratio {ratio_of(r):.2f}, "
                                   f"prior steer {prior_steer}. " + (r.get("audit_note") or ""))
                r["hypothesis"] = None
                r["pairs"] = []
                counts["ground_withdraw"] += 1
                dirty = True
                continue
            ratio = ratio_of(r)
            if ratio >= 1.40:
                counts["hi_keep"] += 1
            elif ratio < 1.15:
                prior_steer = void_steer(r, "lo")
                r["verdict"] = "rejected"
                r["audit_note"] = (f"RE-AUDIT reject (rule 2): ratio {ratio:.3f} < 1.15; "
                                   f"prior steer {prior_steer}. " + (r.get("audit_note") or ""))
                counts["lo_reject"] += 1
                dirty = True
            else:
                r["needs_1000"] = True
                mids.append((r["model"], r["matrix"], r["layer"], r["family"], r["index"]))
                counts["mid_defer"] += 1
                dirty = True
            # rule 3 on surviving validated with steering
            if r["verdict"] == "validated" and r.get("steer"):
                s = r["steer"]
                sh, rnd = abs(s.get("shift", 0.0)), abs(s.get("random_shift", 0.0))
                if s.get("steers") != "no" and (sh <= rnd or max(sh, rnd) < 0.005):
                    s["steers"] = "no"
                    s["steer_note"] = "lost to random control"
                    counts["steer_flipped"] += 1
                    dirty = True
        if dirty:
            Path(f).write_text(json.dumps(D, indent=1))
    print(counts)
    print(len(mids), "mid-band deferred:")
    for m in mids:
        print(" ", m)


if __name__ == "__main__":
    main()
