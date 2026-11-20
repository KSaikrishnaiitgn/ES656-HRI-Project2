"""Per-stride ankle angle summary from the CSVs written by s1_track_ankle_angle.py.

theta = included shank-sole angle at the ankle (90 deg = neutral, <90 dorsiflexion, >90 plantarflexion).
A stride is taken between successive push-off peaks (maximum theta = peak plantarflexion).
Only steady walking is used (start/stop seconds per file).

Usage: python stride_summary.py OUT_TXT "file.csv|t0|t1" ["file2.csv|t0|t1" ...]
"""
import sys
import numpy as np

lines = []
for arg in sys.argv[2:]:
    f, t0, t1 = arg.split('|'); t0, t1 = float(t0), float(t1)
    d = np.genfromtxt(f, delimiter=',', names=True)
    t, th = d['time_s'], d['theta_shank_sole_filt_deg']
    m = (t >= t0) & (t <= t1); t, th = t[m], th[m]
    # push-off peaks: local maxima above the 75th percentile, at least 0.8 s apart
    thr = np.percentile(th, 75); pk = []
    for i in range(1, len(th) - 1):
        if th[i] > th[i-1] and th[i] >= th[i+1] and th[i] > thr:
            if pk and t[i] - t[pk[-1]] < 0.8:
                if th[i] > th[pk[-1]]: pk[-1] = i
            else:
                pk.append(i)
    th_max, th_min, T = [], [], []
    for s, e in zip(pk[:-1], pk[1:]):
        th_max.append(th[s]); th_min.append(th[s:e + 1].min()); T.append(t[e] - t[s])
    th_max, th_min, T = map(np.array, (th_max, th_min, T))
    name = f.replace('/', '\\').split('\\')[-1].replace('_ankle.csv', '')
    lines += [f"{name}: {len(T)} strides ({t0:g}-{t1:g} s)",
              f"  stride time                         {T.mean():.2f} +/- {T.std():.2f} s  (cadence {120/T.mean():.0f} steps/min)",
              f"  min theta  (peak dorsiflexion)      {th_min.mean():5.1f} +/- {th_min.std():.1f} deg  -> {90-th_min.mean():4.1f} deg DF from 90",
              f"  max theta  (peak plantarflexion)    {th_max.mean():5.1f} +/- {th_max.std():.1f} deg  -> {th_max.mean()-90:4.1f} deg PF from 90",
              f"  range of motion per stride          {(th_max - th_min).mean():5.1f} +/- {(th_max - th_min).std():.1f} deg",
              f"  absolute min / max over the trial   {th.min():5.1f} / {th.max():5.1f} deg",
              ""]
lines += ["theta = shank-sole angle at the ankle; 90 deg = neutral, <90 dorsiflexion, >90 plantarflexion.",
          "OpenSim ankle_angle_r (.mot files) = 90 - theta."]
txt = "\n".join(lines)
print(txt)
with open(sys.argv[1], 'w') as fh:
    fh.write(txt + "\n")
