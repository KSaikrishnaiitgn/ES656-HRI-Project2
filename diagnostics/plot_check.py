"""Diagnostic plot: marker y-positions, segment tilts and ankle angle over time."""
import sys, csv
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, len(sys.argv) - 2, figsize=(9 * (len(sys.argv) - 2), 9), squeeze=False)
for c, f in enumerate(sys.argv[2:]):
    d = np.genfromtxt(f, delimiter=',', names=True)
    t = d['time_s']
    ax = axes[:, c]
    for k in ['pinkUp', 'pinkLow', 'heel', 'toe']:
        ax[0].plot(t, d[k + '_y'], label=k)
    ax[0].invert_yaxis(); ax[0].legend(); ax[0].set_title(f.split('\\')[-1] + '  image y (px)')
    sh = np.degrees(np.arctan2(d['pinkUp_x'] - d['pinkLow_x'], d['pinkLow_y'] - d['pinkUp_y']))
    ft = np.degrees(np.arctan2(d['heel_y'] - d['toe_y'], d['toe_x'] - d['heel_x']))
    ax[1].plot(t, sh, label='shank forward lean'); ax[1].plot(t, ft, label='foot sticker line toe-up tilt')
    ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot(t, d['ankle_raw_deg'], lw=.6, label='raw'); ax[2].plot(t, d['ankle_filt_deg'], label='filtered')
    ax[2].legend(); ax[2].grid(alpha=.3); ax[2].set_xlabel('time (s)')
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=80)
