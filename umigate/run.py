"""python -m umigate.run <first_seed> <n_seeds> <profile> [out.json]   profile: dev | held"""
import json, sys
from collections import Counter
from .gen import clean_episode, plant, KINDS, HELD_ONLY
from .check import check, schema_check

WANT = dict(dropped="TIME_GAP", clock_offset="CLOCK_OFFSET", pose_jump="POSE_JUMP", rot_jump="ROT_JUMP", scale="SCALE", stuck_gripper="STUCK_GRIPPER",
            clipped_gripper="CLIPPED_GRIPPER", never_closes="LABEL_MISMATCH", lr_swapped="LR_SWAPPED", interpenetration="INTERPENETRATION",
            nan_burst="NAN", t_rewind="TIME_GAP", drift="SCALE")
PROFILES = dict(dev=dict(rate=20, one_handed=False, kinds=KINDS), held=dict(rate=10, one_handed=None, kinds=KINDS + HELD_ONLY))


def run(first, n, profile):
    P = PROFILES[profile]
    R = dict(planted=Counter(), stopped=Counter(), named=Counter(), schema=Counter(), clean=0, clean_stopped=0, schema_clean_stopped=0, clean_codes=Counter())
    for seed in range(first, first + n):
        one = P["one_handed"] if P["one_handed"] is not None else seed % 4 == 0
        base = clean_episode(seed, rate=P["rate"], one_handed=one)
        v, f, _ = check(base)
        R["clean"] += 1
        if v != "ACCEPT":
            R["clean_stopped"] += 1
            R["clean_codes"].update(x["code"] for x in f)
        R["schema_clean_stopped"] += schema_check(base) != "ACCEPT"
        for k in P["kinds"]:
            bad = plant(base, k, seed)
            v, f, _ = check(bad)
            R["planted"][k] += 1
            R["stopped"][k] += v != "ACCEPT"
            R["named"][k] += any(x["code"] == WANT[k] for x in f)
            R["schema"][k] += schema_check(bad) != "ACCEPT"
    return R


if __name__ == "__main__":
    first, n, profile = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    R = run(first, n, profile)
    print(f"profile {profile}, seeds {first}..{first + n - 1}")
    print(f"planted {sum(R['planted'].values())}: gate stopped {sum(R['stopped'].values())}, named {sum(R['named'].values())}; schema check stopped {sum(R['schema'].values())}")
    for k, c in R["planted"].items():
        print(f"  {k:16s} planted {c:3d}  stopped {R['stopped'][k]:3d}  named {R['named'][k]:3d}  schema {R['schema'][k]:3d}")
    print(f"clean episodes {R['clean']}: gate stopped {R['clean_stopped']} {dict(R['clean_codes'])}; schema check stopped {R['schema_clean_stopped']}")
    if len(sys.argv) > 4:
        json.dump({k: (dict(v) if isinstance(v, Counter) else v) for k, v in R.items()}, open(sys.argv[4], "w"), indent=1)
