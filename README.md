# umigate: a pre-delivery gate for bimanual UMI-style episodes

Datoric's public UMI dataset page lists dual wrist GoPro video with IMU, 6-DoF end-effector pose, gripper width, 10-20 Hz alignment, bimanual clock matching within 1/60 s, and reviews for SLAM and calibration, dropped episodes, trajectory integrity and bimanual alignment (https://www.datoric.com/datasets/umi-dual-camera-manipulation). This prototype is one way to turn those review items into a per-episode check that runs before delivery and says **why** an episode is stopped.

Input: one episode with, per hand, timestamps, pose (position + quaternion), gripper width, IMU specific force, plus a success label. Output: `ACCEPT`, `FIX_TRIM` (keep the longest clean stretch if it is >= 70% of the episode and >= 8 s) or `REJECT`, with codes and frame ranges.

| Code | What it catches |
|---|---|
| `NAN`, `TIME_GAP`, `CLOCK_REWIND` | non-finite values, dropped frames, timestamps that go backwards |
| `CLOCK_OFFSET` | left/right clocks further apart than 1/60 s |
| `POSE_JUMP`, `ROT_JUMP` | SLAM relocalisation: > 2 m/s or > 0.5 rad in one frame (rotation angle uses \|q.q'\|, so a harmless q / -q sign flip is not a jump) |
| `SCALE` | pose acceleration vs the IMU's world acceleration (R f + g); ratio off by > 15% |
| `STUCK_GRIPPER`, `CLIPPED_GRIPPER`, `LABEL_MISMATCH` | width flat, never opens, or "success" on a pick where it never closed |
| `LR_SWAPPED`, `INTERPENETRATION` | hands swapped; grippers within 5 cm for 10+ frames |

## Results (synthetic episodes, one planted defect each)
Dev: 20 Hz, two-handed, seeds 0-39: 440/440 planted defects stopped and named, 0/40 clean stopped. A loader-level schema check (finite + equal length) stopped 40/440 (only the NaN burst).

Held-out: 10 Hz, 25% one-handed tasks, two defect kinds the gate was not written for (`t_rewind`, `drift`), seeds 1000-1099, profile committed before the first run:

| | schema check | umigate (run 1) | umigate (run 2, after one fix) |
|---|---|---|---|
| planted defects stopped | 100 / 1,300 | 1,125 / 1,300 | 1,177 / 1,300 |
| named with the right code | - | 1,076 | 1,176 |
| clean episodes stopped | 0 / 100 | 0 / 100 | 0 / 100 |

Run 1 missed `t_rewind` entirely (no monotonic-time check); I added `CLOCK_REWIND`. What is still missed, and why:
- `drift` 99/100 missed: a constant-velocity SLAM drift (1.5-3 cm/s) is invisible to a second-difference check against the IMU. It needs a different signal (loop closure to a known start pose, or video).
- `scale` 13/100 missed: all 13 are one-handed episodes where the defect landed on the idle hand, which has no motion for the IMU to compare against.
- `pose_jump` 11/100 missed: jumps under 0.2 m (my planted jump size is drawn loosely), below the 2 m/s limit at 10 Hz where normal motion reaches about 1 m/s.

## Limits
- Synthetic motion and noise that I wrote; the gate's thresholds and the defect shapes come from the same author, so it is near-circular. Real GoPro/SLAM failures are messier.
- No video: marker detection, calibration and blur are not covered.
- Stored as numpy arrays with the field names of their spec, not Zarr.

## Run
```
python -m unittest tests.test_umigate
python -m umigate.run 0 40 dev
python -m umigate.run 1000 100 held
```
numpy only.

Unofficial prototype by Kushagra Pratap Singh. Not affiliated with Datoric.
