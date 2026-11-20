"""
Build every figure used by the LaTeX report (PartB_LaTeX_Report/main.tex) into
PartB_LaTeX_Report/figures/ with clear, numbered names.

Re-uses the processing in s2_emg_processing_alignment.py (same filters, artefact masks, gait events)
and the final EMG-video offsets in results/emg/alignment.csv, so every figure shows the
same data the report's numbers come from.
"""
import os, sys, csv, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # 02_code
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import s2_emg_processing_alignment as E
import s4_spring_shoe_model as M

import config_paths as P   # legacy: builds the figures of the archived first LaTeX draft
BASE = P.ROOT
OUT = P.OLD_REPORT_FIG
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3, 'savefig.dpi': 200,
                     'savefig.bbox': 'tight'})
COL = {'U': 'tab:blue', 'C': 'tab:red'}
D = {r['trial']: float(r['D_refined_s']) for r in csv.DictReader(open(os.path.join(P.RES_EMG, 'alignment.csv')))}
TAGS = list(E.TRIALS)
LABEL = {'U1': 'U1 unconstrained', 'U2': 'U2 unconstrained', 'C1': 'C1 constrained', 'C2': 'C2 constrained'}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print('wrote', name)


# ---------------------------------------------------------------- fig01 device geometry
def fig01():
    rot = M.rot
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for th_deg, ls, alpha, lab in [(0, '-', 1.0, 'neutral (0 deg)'), (-15, '--', 0.6, '15 deg plantarflexion')]:
        th = np.radians(th_deg)
        Pf, Pb = rot(th) @ M.P_f0, rot(th) @ M.P_b0
        ax.plot([Pb[0] * 1e3, Pf[0] * 1e3], [Pb[1] * 1e3, Pf[1] * 1e3], 'k', ls=ls, lw=3, alpha=alpha, label=f'foot/projections, {lab}')
        for A, P, c, nm in [(M.A_f, Pf, 'tab:orange', 'front link S1: L1 = 340 mm (wire + spring, k = 754.6 N/m)'),
                            (M.A_b, Pb, 'tab:green', 'back link S2: L2 = 330 mm (wire + spring, k = 754.6 N/m)')]:
            ax.plot([P[0] * 1e3, A[0] * 1e3], [P[1] * 1e3, A[1] * 1e3], c=c, ls=ls, lw=1.8, alpha=alpha,
                    label=nm if th_deg == 0 else None)
    ax.plot([0, 0], [0, 411], c='tab:pink', lw=6, alpha=0.5, label='shank (ankle to knee, 411 mm)')
    ax.plot([M.A_b[0] * 1e3, M.A_f[0] * 1e3], [M.A_f[1] * 1e3] * 2, c='k', lw=6, label='shank band')
    ax.plot(0, 0, 'ko', ms=8); ax.annotate('ankle axis O', (0, 0), (15, -35), fontsize=9)
    ax.annotate('L3 = 170 mm', (60, 12), fontsize=8); ax.annotate('L4 = 130 mm', (-125, 12), fontsize=8)
    ax.annotate(f'band points: x = {M.A_f[0]*1e3:+.1f} / {M.A_b[0]*1e3:+.1f} mm, y = {M.A_f[1]*1e3:.1f} mm',
                (-210, 345), fontsize=8)
    ax.set_aspect('equal'); ax.set_xlim(-230, 260); ax.set_ylim(-200, 430)
    ax.set_xlabel('x, forward (mm)'); ax.set_ylabel('y, up the shank (mm)')
    ax.set_title('Spring-shoe constraint geometry (sagittal plane)'); ax.legend(fontsize=7, loc='lower left')
    save(fig, 'fig01_device_geometry_schematic.png')


# ---------------------------------------------------------------- fig02 torque-angle + stiffness
def fig02():
    ths = np.radians(np.linspace(-25, 25, 501))
    tau = np.array([M.device_torque(t)[0] for t in ths])
    ff = np.array([M.device_torque(t)[1][0] for t in ths]); fb = np.array([M.device_torque(t)[2][0] for t in ths])
    K = -np.gradient(tau, ths)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(np.degrees(ths), tau, lw=2); ax[0].axhline(0, c='k', lw=.6); ax[0].axvspan(-1.33, 0.66, color='grey', alpha=.25)
    ax[0].set_xlabel('ankle angle theta (deg), + DF'); ax[0].set_ylabel('device torque on foot (N m), + DF'); ax[0].set_title('(a) restoring torque')
    ax[1].plot(np.degrees(ths), ff, label='front spring S1', c='tab:orange'); ax[1].plot(np.degrees(ths), fb, label='back spring S2', c='tab:green')
    ax[1].set_xlabel('ankle angle theta (deg)'); ax[1].set_ylabel('spring force (N)'); ax[1].legend(); ax[1].set_title('(b) wire tension')
    ax[2].plot(np.degrees(ths), K, c='tab:purple'); ax[2].set_xlabel('ankle angle theta (deg)'); ax[2].set_ylabel('K = -d tau / d theta (N m/rad)')
    ax[2].set_title('(c) local rotational stiffness')
    save(fig, 'fig02_device_torque_force_stiffness_model.png')


# ---------------------------------------------------------------- fig04 ankle angle time series
def fig04():
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for ax, tag in zip(axes, TAGS):
        t, th, pitch, fps = E.load_kin(tag)
        hs, to = E.gait_events(t, th, pitch)
        m = (t >= D[tag]) & (t <= D[tag] + 8)
        ax.plot(t[m] - D[tag], th[m], c=COL[tag[0]], lw=1.6)
        for i in hs:
            if m[i]: ax.axvline(t[i] - D[tag], c='k', lw=.6, ls=':')
        ax.axhline(90, c='k', lw=.8); ax.set_ylim(72, 112); ax.set_ylabel('theta (deg)'); ax.set_title(LABEL[tag], fontsize=9, loc='left')
    axes[-1].set_xlabel('time from EMG start (s)   [dotted = heel strike; 90 deg = neutral; <90 DF, >90 PF]')
    save(fig, 'fig04_ankle_angle_timeseries_all_trials.png')


# ---------------------------------------------------------------- fig05 spectrum / interference
def fig05():
    ef = E.TRIALS['U2'][1]
    d = np.genfromtxt(os.path.join(E.EMG_DIR, ef), delimiter=',', names=True)
    x = d['Tibialis_Raw_V'] - d['Tibialis_Raw_V'].mean()
    f = np.fft.rfftfreq(len(x), 1 / 2000); X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    def gain(f):
        g = E.butter_mag2(f, E.HP_CUT, 4, 'high') * E.butter_mag2(f, 450, 4, 'low')
        for h in range(50, 500, 50): g = g * (1 - np.exp(-0.5 * ((f - h) / 0.75) ** 2))
        return g
    y = E.fft_filter(x, 2000, gain); Y = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    k = np.ones(40) / 40
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.semilogy(f, np.convolve(X, k, 'same'), c='grey', lw=.8, label='raw TA (U2)')
    ax.semilogy(f, np.convolve(Y, k, 'same') + 1e-9, c='tab:blue', lw=.8, label='after notch + 40-450 Hz band-pass')
    for h in (100, 200, 400): ax.axvline(h, c='r', ls=':', lw=.8)
    ax.set_xlim(0, 600); ax.set_xlabel('frequency (Hz)'); ax.set_ylabel('power (a.u.)'); ax.legend()
    ax.set_title('EMG power spectrum: mains-related interference at 100/200/400 Hz removed by filtering')
    save(fig, 'fig05_emg_spectrum_interference_and_filter.png')


# ---------------------------------------------------------------- fig06 raw EMG + envelope per trial
def fig06():
    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    for row, tag in enumerate(TAGS):
        ef = E.TRIALS[tag][1]
        d = np.genfromtxt(os.path.join(E.EMG_DIR, ef), delimiter=',', names=True)
        t, ta, gas = E.load_emg(ef)
        tv, th, pitch, fps = E.load_kin(tag); hs, _ = E.gait_events(tv, th, pitch)
        for c, (ch, env, name) in enumerate([('Tibialis_Raw_V', ta, 'Tibialis anterior'), ('Gastro_Raw_V', gas, 'Gastrocnemius')]):
            ax = axes[row, c]
            raw = d[ch] - d[ch].mean()
            m = (t >= 4) & (t <= 10)
            ax.plot(t[m], raw[m], c='lightgrey', lw=.4, label='raw (mean removed)')
            ax.plot(t[m], env[m] * 4, c=COL[tag[0]], lw=1.4, label='linear envelope x4')
            bad = np.isnan(env) & m
            if bad.any():
                ax.fill_between(t[m], -1.6, 1.6, where=bad[m], color='red', alpha=.12, label='rejected artefact')
            for i in hs:
                te = tv[i] - D[tag]
                if 4 <= te <= 10: ax.axvline(te, c='k', lw=.6, ls=':')
            ax.set_ylim(-1.6, 1.6); ax.set_title(f'{LABEL[tag]} - {name}', fontsize=9, loc='left')
            if row == 0: ax.legend(fontsize=7, loc='upper right')
    for ax in axes[:, 0]: ax.set_ylabel('V')
    for ax in axes[-1]: ax.set_xlabel('EMG time (s)   [dotted = heel strike from video]')
    save(fig, 'fig06_emg_raw_and_envelope_all_trials.png')


# ---------------------------------------------------------------- fig07 artefact evidence
def fig07():
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for a, tag, title in [(ax[0], 'U2', 'U2 (unconstrained): genuine TA burst'), (ax[1], 'C1', 'C1 (constrained): TA at heel strike, clipped')]:
        ef = E.TRIALS[tag][1]
        d = np.genfromtxt(os.path.join(E.EMG_DIR, ef), delimiter=',', names=True)
        t, ta, gas = E.load_emg(ef); raw = d['Tibialis_Raw_V']
        env_full = E.emg_envelope(raw)
        pk = E.local_extrema(env_full, t, 'max', 0.8, np.percentile(env_full, 90))
        i0 = pk[len(pk) // 2]; s = slice(i0 - 600, i0 + 600)
        a.plot(t[s], raw[s], lw=.5, c=COL[tag[0]]); a.axhline(3.3, c='k', ls='--', lw=.7); a.axhline(0, c='k', ls='--', lw=.7)
        a.set_ylim(-0.1, 3.4); a.set_title(title, fontsize=9); a.set_xlabel('EMG time (s)'); a.set_ylabel('raw ADC (V), rails 0 / 3.3 V')
    for tag in ('U2', 'C1'):
        ef = E.TRIALS[tag][1]
        d = np.genfromtxt(os.path.join(E.EMG_DIR, ef), delimiter=',', names=True)
        raw = d['Tibialis_Raw_V'] - d['Tibialis_Raw_V'].mean(); t = d['Time_s']
        env_full = E.emg_envelope(d['Tibialis_Raw_V'])
        pk = E.local_extrema(env_full, t, 'max', 0.8, np.percentile(env_full, 90))
        P = []
        for i0 in pk:
            seg = raw[max(0, i0 - 200):i0 + 200]
            if len(seg) < 400: continue
            P.append(np.abs(np.fft.rfft(seg * np.hanning(400))) ** 2)
        f = np.fft.rfftfreq(400, 1 / 2000); Pm = np.mean(P, 0); Pm /= Pm.sum()
        ax[2].plot(f, Pm, c=COL[tag[0]], label=f'{tag} TA burst windows')
    ax[2].set_xlim(0, 500); ax[2].set_xlabel('frequency (Hz)'); ax[2].set_ylabel('normalised power'); ax[2].legend()
    ax[2].set_title('burst spectra: artefact is low-frequency', fontsize=9)
    save(fig, 'fig07_emg_artefact_clean_vs_clipped.png')


# ---------------------------------------------------------------- fig11 per-trial EMG gait profiles
def fig11():
    rows = list(csv.DictReader(open(os.path.join(P.RES_EMG, 'per_stride_metrics.csv'))))
    keys = [('po_exc', 'push-off PF excursion (deg)'), ('T', 'stride time (s)'), ('ta_swing', 'TA mean in swing (%)'),
            ('gas_peak', 'GAS stance peak (%)'), ('cci', 'co-contraction index (%)')]
    fig, axes = plt.subplots(1, len(keys), figsize=(16, 3.6))
    for ax, (k, name) in zip(axes, keys):
        for j, tag in enumerate(TAGS):
            v = np.array([float(r[k]) for r in rows if r['trial'] == tag and r[k] not in ('nan', '')])
            v = v[np.isfinite(v)]
            ax.scatter(np.full(len(v), j) + np.random.default_rng(j).uniform(-.12, .12, len(v)), v, s=12, c=COL[tag[0]], alpha=.6)
            if len(v): ax.plot([j - .25, j + .25], [v.mean()] * 2, c='k', lw=2)
        ax.set_xticks(range(4)); ax.set_xticklabels(TAGS); ax.set_title(name, fontsize=9)
    save(fig, 'fig11_per_stride_metrics_by_trial.png')


def copy_existing():
    pairs = [(os.path.join(P.RES_FIG, 'video_frames_U2_vs_C1.png'), 'fig03_annotated_video_frames_U2_vs_C1.png'),
             (os.path.join(P.RES_FIG, 'emg_video_alignment.png'), 'fig08_emg_video_alignment_correlation.png'),
             (os.path.join(P.RES_FIG, 'gait_cycle_theta_TA_GAS.png'), 'fig09_gait_cycle_theta_TA_GAS_mean_sd.png'),
             (os.path.join(P.RES_FIG, 'condition_comparison.png'), 'fig10_condition_comparison_bars.png')]
    for tag, (vid, ef, *_) in E.TRIALS.items():
        pairs.append((os.path.join(E.EMG_DIR, ef.replace('.csv', '.png')), f'figA_{tag}_emg_recording_system_plot.png'))
    for src, dst in pairs:
        shutil.copyfile(src, os.path.join(OUT, dst)); print('copied', dst)


if __name__ == '__main__':
    fig01(); fig02(); fig04(); fig05(); fig06(); fig07(); fig11(); copy_existing()
