"""Pre-delivery gate for bimanual UMI-style episodes. Findings carry a code, the hand, and the frame range.

ACCEPT / FIX_TRIM (a clean contiguous stretch >= 70% of the episode and >= 8 s is kept) / REJECT.
Codes: NAN, TIME_GAP, CLOCK_OFFSET, POSE_JUMP, ROT_JUMP, SCALE, STUCK_GRIPPER, CLIPPED_GRIPPER, LABEL_MISMATCH, LR_SWAPPED, INTERPENETRATION.
"""
import numpy as np
from .gen import rotmat, G, MAXW

WHOLE = {"CLOCK_OFFSET", "SCALE", "STUCK_GRIPPER", "CLIPPED_GRIPPER", "LABEL_MISMATCH", "LR_SWAPPED"}
MAX_SPEED = 2.0            # m/s, one frame
MAX_ROT = 0.5              # rad, one frame
BIMANUAL_SYNC = 1 / 60


def runs(mask):
    """[(start, end_exclusive)] of True runs."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def check(ep):
    f, rate = [], ep["rate"]
    dt = 1 / rate
    n = min(len(ep["L"]["t"]), len(ep["R"]["t"]))
    bad = np.zeros(n, bool)

    def add(code, hand, a=None, b=None, msg=""):
        f.append(dict(code=code, hand=hand, frames=None if a is None else (int(a), int(b)), msg=msg))
        if a is not None:
            bad[a:b] = True

    for name in ("L", "R"):
        h = ep[name]
        fin = np.isfinite(h["pos"]).all(1) & np.isfinite(h["quat"]).all(1) & np.isfinite(h["width"]) & np.isfinite(h["imu_acc"]).all(1)
        for a, b in runs(~fin[:n]):
            add("NAN", name, a, b, f"{b - a} non-finite frames")
        for i in np.where(np.diff(h["t"][:n]) > 1.5 * dt)[0]:
            add("TIME_GAP", name, i, i + 2, f"{(h['t'][i + 1] - h['t'][i]) / dt:.1f} frame intervals")
        sp = np.linalg.norm(np.diff(h["pos"][:n], axis=0), axis=1) / dt
        for i in np.where(sp > MAX_SPEED)[0]:
            add("POSE_JUMP", name, i, i + 2, f"{sp[i]:.1f} m/s between frames")
        dot = np.abs(np.sum(h["quat"][1:n] * h["quat"][:n - 1], 1))     # |dot| so q and -q are the same rotation
        ang = 2 * np.arccos(np.clip(dot, 0, 1))
        for i in np.where(ang > MAX_ROT)[0]:
            add("ROT_JUMP", name, i, i + 2, f"{np.degrees(ang[i]):.0f} deg between frames")
        w = h["width"][:n]
        if np.nanstd(w) < 0.002:
            add("STUCK_GRIPPER", name, msg="gripper width does not change")
        elif np.nanmax(w) < 0.04:
            add("CLIPPED_GRIPPER", name, msg=f"gripper never opens past {np.nanmax(w) * 1000:.0f} mm")
        elif ep["success"] and np.nanmin(w) > 0.06:
            add("LABEL_MISMATCH", name, msg="labelled success but the gripper never closes")
        # scale: pose acceleration vs IMU world acceleration (R f + g), away from jumps and gaps
        ok = fin[:n].copy()
        ok[1:] &= ~bad[1:]
        p = h["pos"][:n]
        a_pose = (p[2:] - 2 * p[1:-1] + p[:-2]) * rate**2
        a_imu = np.einsum("nij,nj->ni", rotmat(h["quat"][:n]), h["imu_acc"][:n]) + G
        m = ok[1:-1] & ok[:-2] & ok[2:]
        for a, b in runs(bad):                                          # keep a margin around anything already flagged
            m[max(0, a - 3):max(0, b - 1)] = False
        if m.sum() > 20 and np.sqrt(np.mean(a_imu[1:-1][m] ** 2)) > 0.3:
            s = np.sum(a_pose[m] * a_imu[1:-1][m]) / np.sum(a_imu[1:-1][m] ** 2)
            if abs(s - 1) > 0.15:
                add("SCALE", name, msg=f"pose motion is {s:.2f}x what the IMU says")
    t0 = np.nanmedian(ep["L"]["t"][:n] - ep["R"]["t"][:n])
    if abs(t0) > BIMANUAL_SYNC:
        add("CLOCK_OFFSET", "L-R", msg=f"hands are {t0 * 1000:+.0f} ms apart (limit {BIMANUAL_SYNC * 1000:.1f} ms)")
    if np.nanmedian(ep["L"]["pos"][:n, 1] - ep["R"]["pos"][:n, 1]) < 0:
        add("LR_SWAPPED", "L-R", msg="left gripper sits right of the right gripper")
    close = np.linalg.norm(ep["L"]["pos"][:n] - ep["R"]["pos"][:n], axis=1) < 0.05
    for a, b in runs(close):
        if b - a >= 10:
            add("INTERPENETRATION", "L-R", a, b, f"grippers within 5 cm for {b - a} frames")
    if not f:
        return "ACCEPT", f, (0, n)
    if any(x["code"] in WHOLE for x in f):
        return "REJECT", f, None
    good = ~bad
    best = max(runs(good), key=lambda r: r[1] - r[0], default=(0, 0))
    if (best[1] - best[0]) >= 0.7 * n and (best[1] - best[0]) * dt >= 8:
        return "FIX_TRIM", f, best
    return "REJECT", f, None


def schema_check(ep):
    """What a loader-level check gives: arrays present, equal length, finite."""
    for h in (ep["L"], ep["R"]):
        if not (np.isfinite(h["pos"]).all() and np.isfinite(h["quat"]).all()):
            return "REJECT"
    return "ACCEPT" if len(ep["L"]["t"]) == len(ep["R"]["t"]) else "REJECT"
