import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # 02_code
"""Diagnostic: theta, sole pitch, detected gait events and EMG envelopes per trial (one row per trial)."""
import sys, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import s2_emg_processing_alignment as E

t0, t1 = float(sys.argv[2]), float(sys.argv[3])     # EMG-time window to show (s)
fig, axes = plt.subplots(4, 1, figsize=(16, 13))
for ax, (tag, (vid, ef, ct, vd, cond)) in zip(axes, E.TRIALS.items()):
    t_e, ta, gas = E.load_emg(ef)
    t_v, th, pitch, fps = E.load_kin(tag)
    hs, to = E.gait_events(t_v, th, pitch)
    D = dict(U1=7.54, U2=15.46, C1=45.26, C2=21.81)[tag]
    m = (t_v - D >= t0) & (t_v - D <= t1)
    ax.plot(t_v[m] - D, th[m], 'k', label='theta')
    ax.plot(t_v[m] - D, 90 + pitch[m], c='gray', lw=0.8, label='90 + sole pitch')
    for i in hs:
        if m[i]: ax.axvline(t_v[i] - D, c='g', lw=0.8)
    for i in to:
        if m[i]: ax.axvline(t_v[i] - D, c='m', lw=0.8, ls='--')
    ax2 = ax.twinx()
    me = (t_e >= t0) & (t_e <= t1)
    ax2.plot(t_e[me], ta[me] * 1000, c='tab:blue', lw=1, label='TA env (mV)')
    ax2.plot(t_e[me], gas[me] * 1000, c='tab:orange', lw=1, label='GAS env (mV)')
    ax.set_title(f'{tag} (green = HS, magenta = TO)'); ax.legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8)
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=80)
