import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # 02_code
"""Diagnostic: is the large TA burst at heel strike in the constrained trials EMG or movement artefact?
For each trial: median frequency and share of power < 20 Hz in 200 ms windows centred on the
largest TA-envelope peaks, vs. quiet windows. Also plots raw TA around one burst."""
import sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import s2_emg_processing_alignment as E

fig, axes = plt.subplots(2, 2, figsize=(14, 6)); axes = axes.ravel()
for ax, (tag, (vid, ef, *_)) in zip(axes, E.TRIALS.items()):
    d = np.genfromtxt(f"{E.EMG_DIR}/{ef}", delimiter=',', names=True)
    raw = d['Tibialis_Raw_V'] - d['Tibialis_Raw_V'].mean()
    t, ta, gas = E.load_emg(ef)
    pk = E.local_extrema(ta, t, 'max', 0.8, np.percentile(ta, 90))
    def stats(i0):
        seg = raw[max(0, i0 - 200):i0 + 200]
        X = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2; f = np.fft.rfftfreq(len(seg), 1 / 2000)
        keep = (f > 2) & (np.min(np.abs(f[:, None] - np.arange(50, 500, 50)), axis=1) > 4)  # drop mains bins
        X, f = X[keep], f[keep]
        c = np.cumsum(X); mdf = f[np.searchsorted(c, c[-1] / 2)]
        return mdf, X[f < 20].sum() / X.sum() * 100, np.sqrt(np.mean(seg ** 2))
    b = np.array([stats(i) for i in pk])
    q = np.array([stats(i) for i in np.linspace(1000, len(raw) - 1000, 60).astype(int)])
    print(f"{tag}: bursts n={len(pk)}  median freq {b[:,0].mean():5.1f} Hz, <20 Hz share {b[:,1].mean():4.1f} %, "
          f"RMS {b[:,2].mean()*1000:5.0f} mV | all windows: MDF {q[:,0].mean():5.1f} Hz, <20 Hz {q[:,1].mean():4.1f} %, "
          f"RMS {q[:,2].mean()*1000:4.0f} mV")
    i0 = pk[len(pk) // 2]
    s = slice(i0 - 600, i0 + 600)
    ax.plot(t[s], raw[s], lw=0.5); ax.set_title(f"{tag} raw TA around a burst peak"); ax.set_ylim(-1.5, 1.5)
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=80)
