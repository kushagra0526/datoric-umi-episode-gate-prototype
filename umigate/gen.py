"""Synthetic bimanual UMI-style episodes with the fields Datoric's public UMI spec lists, and planted defects.

Per hand: t (s), pos (N,3 m), quat (N,4 w,x,y,z), width (N, m), imu_acc (N,3 m/s^2 specific force in the gripper frame).
Episode: dict(L=hand, R=hand, rate, success, task, defects=[...]). Fictional data only.
"""
import numpy as np

G = np.array([0.0, 0.0, -9.81])
MAXW = 0.085


def quat_from_rotvec(v):
    a = np.linalg.norm(v, axis=-1, keepdims=True)
    h = a / 2
    axis = np.where(a > 1e-9, v / np.maximum(a, 1e-9), 0)
    return np.concatenate([np.cos(h), axis * np.sin(h)], -1)


def rotmat(q):
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def minjerk(r, waypoints, seg_t, fs):
    """Piecewise minimum-jerk path through waypoints; returns positions sampled at fs and exact accelerations."""
    pos, acc = [], []
    for a, b, T in zip(waypoints[:-1], waypoints[1:], seg_t):
        n = int(T * fs)
        s = np.arange(n) / fs / T
        p = 10 * s**3 - 15 * s**4 + 6 * s**5
        ddp = (60 * s - 180 * s**2 + 120 * s**3) / T**2
        pos.append(a + np.outer(p, b - a)); acc.append(np.outer(ddp, b - a))
    return np.concatenate(pos), np.concatenate(acc)


def hand(r, side, fs, dur, active=True):
    y0 = 0.2 * side
    k = max(3, int(dur / 1.2))
    wp = [np.array([0.4, y0, 0.3])]
    for _ in range(k):
        wp.append(wp[-1] + (r.normal(0, 0.2, 3) * np.array([1, 1, 0.7]) if active else r.normal(0, 0.004, 3)))
        wp[-1][1] = np.clip(wp[-1][1], y0 - 0.15, y0 + 0.15) if side > 0 else np.clip(wp[-1][1], y0 - 0.15, y0 + 0.15)
    seg = np.full(k, dur / k)
    pos, acc = minjerk(r, wp, seg, 200)                         # simulate at 200 Hz, decimate to the recording rate
    n = len(pos)
    rv = np.cumsum(r.normal(0, 0.002, (n, 3)), 0)
    rv -= np.linspace(0, 1, n)[:, None] * rv[-1]                # smooth, bounded orientation wander
    q = quat_from_rotvec(rv)
    R = rotmat(q)
    f = np.einsum("nji,nj->ni", R, acc - G)                     # specific force in the gripper frame: R^T (a - g)
    width = np.full(n, MAXW)
    idx = np.linspace(0, n, 2 * (k // 2) + 3).astype(int)
    for i in range(1, len(idx) - 1, 2):                         # close, hold, open
        a, b = idx[i], idx[i + 1]
        width[a:b] = np.interp(np.arange(a, b), [a, a + 15, b - 15, b], [MAXW, 0.03, 0.03, MAXW])
    step = 200 // fs
    sl = slice(0, None, step)
    t = np.arange(len(pos[sl])) / fs
    return dict(t=t, pos=pos[sl].copy(), quat=q[sl].copy(), width=width[sl].copy() + r.normal(0, 0.0005, len(t)),
                imu_acc=f[sl] + r.normal(0, 0.05, (len(t), 3)))


def clean_episode(seed, rate=20, dur=None, one_handed=False, benign=True):
    r = np.random.default_rng(seed)
    dur = dur or float(r.uniform(12, 40))
    L, R = hand(r, +1, rate, dur), hand(r, -1, rate, dur, active=not one_handed)
    n = min(len(L["t"]), len(R["t"]))
    for h in (L, R):
        for k in h:
            h[k] = h[k][:n]
    if benign:                                                   # q and -q are the same rotation: legitimate sign flips
        for h in (L, R):
            for a in r.choice(n, size=r.integers(0, 4), replace=False):
                h["quat"][a:] *= -1
    return dict(L=L, R=R, rate=rate, success=1, task="pick_place", defects=[], one_handed=one_handed)


def plant(ep, kind, seed):
    r = np.random.default_rng(seed * 31 + 7)
    ep = {**ep, "L": {k: v.copy() for k, v in ep["L"].items()}, "R": {k: v.copy() for k, v in ep["R"].items()}, "defects": [kind]}
    h = ep["L" if r.random() < 0.5 else "R"]
    n = len(h["t"])
    k = int(r.integers(n // 4, 3 * n // 4))
    if kind == "dropped":                                        # lost frames: timestamps jump while the data continues
        gap = int(r.integers(3, 8))
        for hh in (h,):
            hh["t"][k:] += gap / ep["rate"]
    elif kind == "clock_offset":                                 # one hand's clock is off by 40-90 ms
        h["t"] = h["t"] + float(r.choice([-1, 1]) * r.uniform(0.04, 0.09))
    elif kind == "pose_jump":                                    # SLAM relocalization: position teleports
        h["pos"][k:] += r.normal(0, 1, 3) / np.linalg.norm(r.normal(0, 1, 3)) * r.uniform(0.3, 0.8)
    elif kind == "rot_jump":                                     # orientation snaps by > 60 degrees
        ax = r.normal(0, 1, 3); ax /= np.linalg.norm(ax)
        dq = quat_from_rotvec(ax * r.uniform(1.2, 2.5))
        w1, v1 = dq[0], dq[1:]
        q = h["quat"][k:]
        w2, v2 = q[:, :1], q[:, 1:]
        h["quat"][k:] = np.concatenate([w1 * w2 - v2 @ v1[:, None], w1 * v2 + w2 * v1 + np.cross(v1, v2)], 1)
    elif kind == "scale":                                        # monocular SLAM scale error
        c = h["pos"][0].copy()
        h["pos"] = c + (h["pos"] - c) * float(r.choice([0.65, 1.4]))
    elif kind == "stuck_gripper":
        h["width"][:] = h["width"][0]
    elif kind == "clipped_gripper":
        h["width"] = h["width"] * 0.35
    elif kind == "never_closes":                                 # success label says a pick happened; width never closed
        h["width"] = 0.075 + 0.01 * np.sin(np.arange(n) / 7.0)
    elif kind == "lr_swapped":
        ep["L"], ep["R"] = ep["R"], ep["L"]
    elif kind == "interpenetration":                             # tracking collapses: both hands at one point for a stretch
        o = ep["R"] if h is ep["L"] else ep["L"]
        a, b = k, min(n, k + 40)
        o["pos"][a:b] = h["pos"][a:b] + r.normal(0, 0.005, (b - a, 3))
    elif kind == "nan_burst":
        h["pos"][k:k + 5] = np.nan
    elif kind == "t_rewind":                                     # held-out only: clock reset, timestamps go backwards
        h["t"][k:] -= h["t"][k] - h["t"][k - 1] + float(r.uniform(2, 6))
    elif kind == "drift":                                        # held-out only: slow SLAM drift, 1.5-3 cm/s
        v = r.normal(0, 1, 3); v /= np.linalg.norm(v)
        h["pos"] = h["pos"] + np.outer(h["t"] - h["t"][0], v) * float(r.uniform(0.015, 0.03))
    return ep


KINDS = ["dropped", "clock_offset", "pose_jump", "rot_jump", "scale", "stuck_gripper", "clipped_gripper", "never_closes", "lr_swapped",
         "interpenetration", "nan_burst"]
HELD_ONLY = ["t_rewind", "drift"]
