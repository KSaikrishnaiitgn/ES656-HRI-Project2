"""
HRI Class Project 2 - Part B: detailed phase-by-phase analysis for the report.

Re-uses the processing of s2_emg_processing_alignment.py (same filters, artefact masks, gait events,
EMG-video alignment and normalisation) and adds:
  1. Raw-signal statistics of the actual EMG recordings (01_raw_data/emg/*.csv) per trial:
     DC level, clipped samples, masked samples, band-passed RMS and median frequency of the
     clean signal, and burst amplitudes of the envelope in mV.
  2. Walking-pattern timing: stance/swing durations, cadence, timing of the ankle events.
  3. Ankle angular velocity (push-off plantarflexion and swing dorsiflexion speeds).
  4. Phase-averaged ankle angle and TA/GAS activation (loading, mid/terminal stance,
     push-off, swing), using only artefact-free samples.
  5. Effect sizes (Cohen's d, stride level, descriptive only).
  6. Figures: fig12 (representative single strides: raw EMG, envelope and angle),
     fig13 (angular velocity over the gait cycle), fig14 (phase-averaged bars).

Run from the 02_code folder after s2_emg_processing_alignment.py:
    python s3_phase_analysis.py
Outputs: 03_results/emg/detailed_phase_summary.txt and three PNGs in 03_results/figures/.
"""
import os, csv
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import s2_emg_processing_alignment as E

import config_paths as P
BASE = E.BASE
FIG_OUT = P.RES_FIG                     # draft-style figures; the report figures come from s7_report_figures.py
TXT_OUT = os.path.join(P.RES_EMG, 'detailed_phase_summary.txt')
COND = {'U1': 'Unconstrained', 'U2': 'Unconstrained', 'C1': 'Constrained', 'C2': 'Constrained'}
COL = {'Unconstrained': 'tab:blue', 'Constrained': 'tab:red'}
FS = E.FS_EMG


def bandpass(raw):
    """Notch + 40-450 Hz band-pass (same as the envelope pipeline, before rectification)."""
    x = raw - raw.mean()

    def gain(f):
        g = E.butter_mag2(f, E.HP_CUT, 4, 'high') * E.butter_mag2(f, 450, 4, 'low')
        for h in range(50, 500, 50):
            g = g * (1 - np.exp(-0.5 * ((f - h) / 0.75) ** 2))
        return g
    return E.fft_filter(x, FS, gain)


def median_freq(x):
    f = np.fft.rfftfreq(len(x), 1 / FS); P = np.abs(np.fft.rfft(x - x.mean())) ** 2
    c = np.cumsum(P); return f[np.searchsorted(c, c[-1] / 2)]


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (b.mean() - a.mean()) / sp


def main():
    al = {r['trial']: float(r['D_refined_s']) for r in csv.DictReader(open(os.path.join(P.RES_EMG, 'alignment.csv')))}
    lines = []

    # ---------------------------------------------------------------- 1. raw-signal statistics
    lines += ["1. Actual EMG recordings: raw-signal statistics (0-3.3 V ADC, 2000 Hz, 20 s)",
              f"{'trial':5s} {'ch':4s} {'DC (V)':>7s} {'clip %':>7s} {'masked %':>9s} {'RMS clean (mV)':>15s} "
              f"{'MDF clean (Hz)':>15s} {'env p95 (mV)':>13s}"]
    raw_cache, env_cache = {}, {}
    for tag, (vid, ef, *_r) in E.TRIALS.items():
        d = np.genfromtxt(os.path.join(E.EMG_DIR, ef), delimiter=',', names=True)
        raw_cache[tag] = d
        for ch, name in (('Tibialis_Raw_V', 'TA'), ('Gastro_Raw_V', 'GAS')):
            raw = d[ch]
            m, sat = E.saturation_mask(raw); m = m | E.lowfreq_mask(raw)
            bp = bandpass(raw)
            env = E.emg_envelope(raw)
            env_cache[(tag, name)] = (env, m, bp)
            clean = bp[~m]
            # median frequency over clean 1 s chunks
            mdfs = []
            for s in range(0, len(raw) - 2000, 2000):
                if not m[s:s + 2000].any():
                    mdfs.append(median_freq(bp[s:s + 2000]))
            lines.append(f"{tag:5s} {name:4s} {raw.mean():7.3f} {100 * ((raw >= E.SAT_HI) | (raw <= E.SAT_LO)).mean():7.2f} "
                         f"{100 * m.mean():9.1f} {1000 * np.sqrt(np.mean(clean ** 2)):15.1f} "
                         f"{(np.median(mdfs) if mdfs else np.nan):15.0f} {1000 * np.nanpercentile(env[~m], 95):13.1f}")
    lines.append("RMS = band-passed (40-450 Hz, notched) signal over artefact-free samples; MDF = median frequency of "
                 "fully clean 1 s windows; env p95 = 95th percentile of the clean envelope.")

    # ---------------------------------------------------------------- strides (same logic as s2_emg_processing_alignment)
    emg = {}
    for tag, (vid, ef, *_r) in E.TRIALS.items():
        t_e, ta, gas = E.load_emg(ef)
        emg[tag] = (t_e, ta, gas)
    strides = {}
    for tag in E.TRIALS:
        t_v, theta, pitch, fps = E.load_kin(tag); t_e, ta, gas = emg[tag]; D = al[tag]
        hs, to = E.gait_events(t_v, theta, pitch)
        omega = np.gradient(theta, t_v)                       # deg/s, + = towards plantarflexion
        rows = []
        for a, b in zip(hs[:-1], hs[1:]):
            if t_v[a] < D or t_v[b] > D + t_e[-1]:
                continue
            tt = np.linspace(t_v[a], t_v[b], 101)
            to_in = to[(to > a) & (to < b)]
            to_pct = 100 * (t_v[to_in[0]] - t_v[a]) / (t_v[b] - t_v[a]) if len(to_in) else np.nan
            rows.append(dict(t0=t_v[a], t1=t_v[b], T=t_v[b] - t_v[a], to_pct=to_pct,
                             th=np.interp(tt, t_v, theta), om=np.interp(tt, t_v, omega),
                             ta=np.interp(tt - D, t_e, ta), gas=np.interp(tt - D, t_e, gas)))
        strides[tag] = rows
    norm_ta = np.nanmean([np.nanmax(s['ta']) for t in ('U1', 'U2') for s in strides[t]])
    norm_gas = np.nanmean([np.nanmax(s['gas']) for t in ('U1', 'U2') for s in strides[t]])
    for tag in strides:
        for s in strides[tag]:
            s['ta_n'] = 100 * s['ta'] / norm_ta; s['gas_n'] = 100 * s['gas'] / norm_gas
    lines += ["", f"Normalisation check: TA {1000 * norm_ta:.1f} mV, GAS {1000 * norm_gas:.1f} mV"]

    # ---------------------------------------------------------------- 2-3. walking pattern and velocity
    def kin_metrics(s):
        th, om, T, to = s['th'], s['om'], s['T'], s['to_pct']
        to_i = int(round(to))
        st_min_i = 15 + int(np.argmin(th[15:61]))
        po_i = 50 + int(np.argmax(th[50:86]))
        sw_min_i = to_i + int(np.argmin(th[to_i:96])) if to_i < 95 else np.nan
        return dict(T=T, cadence=120 / T, stance=T * to / 100, swing=T * (1 - to / 100), to=to,
                    th_hs=th[0], th_to=th[to_i], th_dfmin=th[st_min_i], t_dfmin=st_min_i,
                    th_po=th[po_i], t_po=po_i, th_sw=th[sw_min_i] if np.isfinite(sw_min_i) else np.nan,
                    t_sw=sw_min_i, df_exc_stance=th[0] - th[st_min_i],
                    sw_return=th[po_i] - (th[sw_min_i] if np.isfinite(sw_min_i) else np.nan),
                    om_pf=om[40:86].max(), om_df=-om[to_i:96].min() if to_i < 95 else np.nan,
                    om_load=-om[0:20].min())
    km = {tag: [kin_metrics(s) for s in strides[tag]] for tag in strides}
    kkeys = [('T', 'stride time (s)'), ('cadence', 'cadence (steps/min)'), ('stance', 'stance duration (s)'),
             ('swing', 'swing duration (s)'), ('to', 'toe-off (% GC)'),
             ('th_hs', 'theta at heel strike (deg)'), ('df_exc_stance', 'stance DF excursion HS->min (deg)'),
             ('th_dfmin', 'peak stance DF, min theta (deg)'), ('t_dfmin', 'timing of peak DF (% GC)'),
             ('th_po', 'push-off peak theta (deg)'), ('t_po', 'timing of push-off peak (% GC)'),
             ('th_to', 'theta at toe-off (deg)'),
             ('th_sw', 'swing minimum theta (deg)'), ('t_sw', 'timing of swing minimum (% GC)'),
             ('sw_return', 'swing DF return, push-off peak -> swing min (deg)'),
             ('om_load', 'peak DF velocity, loading 0-20% (deg/s)'),
             ('om_pf', 'peak PF velocity, push-off (deg/s)'), ('om_df', 'peak DF velocity, swing (deg/s)')]

    def ms_(vals):
        v = np.asarray(vals, float); v = v[~np.isnan(v)]
        return (v.mean(), v.std(), len(v)) if len(v) else (np.nan, np.nan, 0)

    lines += ["", "2-3. Walking pattern, ankle events and angular velocity (mean +/- SD over strides)",
              f"{'metric':52s}" + ''.join(f"{t:>14s}" for t in E.TRIALS) + f"{'Uncon':>14s}{'Con':>14s}{'d':>7s}"]
    for k, name in kkeys:
        row = f"{name:52s}"
        for tag in E.TRIALS:
            m, s, n = ms_([x[k] for x in km[tag]]); row += f"{m:8.2f}+/-{s:4.2f}"
        u = [x[k] for t in ('U1', 'U2') for x in km[t]]; c = [x[k] for t in ('C1', 'C2') for x in km[t]]
        mu, su, _ = ms_(u); mc, sc, _ = ms_(c)
        row += f"{mu:8.2f}+/-{su:4.2f}{mc:8.2f}+/-{sc:4.2f}{cohen_d(u, c):7.2f}"
        lines.append(row)
    lines.append("d = Cohen's d (constrained - unconstrained)/pooled SD, strides treated as independent: descriptive only.")

    # ---------------------------------------------------------------- 4. phase-averaged values
    def phase_windows(to):
        to_i = int(round(to))
        return {'Loading (0-15%)': (0, 16), 'Mid/terminal stance (15-50%)': (15, 51),
                'Push-off (50%-TO)': (50, to_i + 1), 'Swing (TO-100%)': (to_i, 101)}
    lines += ["", "4. Phase-averaged values (clean samples only; n = strides with >= 50 % of the phase clean)",
              f"{'phase':30s} {'quantity':10s} {'Unconstrained':>22s} {'Constrained':>22s}"]
    phase_tab = {}
    for ph in ['Loading (0-15%)', 'Mid/terminal stance (15-50%)', 'Push-off (50%-TO)', 'Swing (TO-100%)']:
        for q in ('theta', 'TA', 'GAS', 'CCI'):
            out = []
            for cond, tags in (('Unconstrained', ('U1', 'U2')), ('Constrained', ('C1', 'C2'))):
                vals = []
                for t in tags:
                    for s in strides[t]:
                        a, b = phase_windows(s['to_pct'])[ph]
                        if q == 'theta':
                            vals.append(s['th'][a:b].mean()); continue
                        ta, gas = s['ta_n'][a:b], s['gas_n'][a:b]
                        if q == 'TA':
                            x = ta
                        elif q == 'GAS':
                            x = gas
                        else:
                            both = ~np.isnan(ta) & ~np.isnan(gas)
                            vals.append(200 * np.minimum(ta, gas)[both].sum() / (ta + gas)[both].sum()
                                        if both.mean() >= 0.5 else np.nan); continue
                        v = ~np.isnan(x)
                        vals.append(x[v].mean() if v.mean() >= 0.5 else np.nan)
                m, sd, n = ms_(vals); out.append((m, sd, n))
            phase_tab[(ph, q)] = out
            lines.append(f"{ph:30s} {q:10s} " + ''.join(f"{m:9.1f}+/-{sd:5.1f} (n={n:2d})" for m, sd, n in out))

    # ---------------------------------------------------------------- 5. burst amplitudes in mV (actual units)
    lines += ["", "5. Burst amplitudes of the envelope in mV (clean strides only)"]
    for cond, tags in (('Unconstrained', ('U1', 'U2')), ('Constrained', ('C1', 'C2'))):
        tsw, gst = [], []
        for t in tags:
            for s in strides[t]:
                to_i = int(round(s['to_pct']))
                x = s['ta'][to_i:]; v = ~np.isnan(x)
                if v.mean() >= 0.5: tsw.append(1000 * np.nanmax(x))
                x = s['gas'][:to_i + 1]; v = ~np.isnan(x)
                if v.mean() >= 0.5: gst.append(1000 * np.nanmax(x))
        a, b = ms_(tsw), ms_(gst)
        lines.append(f"{cond:14s} TA swing burst peak {a[0]:6.1f} +/- {a[1]:5.1f} mV (n={a[2]}),  "
                     f"GAS stance burst peak {b[0]:6.1f} +/- {b[1]:5.1f} mV (n={b[2]})")

    open(TXT_OUT, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    # ---------------------------------------------------------------- fig12: representative single strides
    def pick(tag):
        """Stride with the most clean EMG (ties -> closest push-off excursion to the trial mean)."""
        best = None
        for s in strides[tag]:
            clean = (~np.isnan(s['ta']) & ~np.isnan(s['gas'])).mean()
            if best is None or clean > best[0] + 1e-9:
                best = (clean, s)
        return best[1]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.5), sharex='col')
    for col, tag in enumerate(('U2', 'C1')):
        s = pick(tag); D = al[tag]; d = raw_cache[tag]
        te = d['Time_s']; m = (te >= s['t0'] - D) & (te <= s['t1'] - D)
        gc = 100 * (te[m] - (s['t0'] - D)) / s['T']
        cond = COND[tag]
        axes[0, col].plot(np.linspace(0, 100, 101), s['th'], c=COL[cond], lw=2)
        axes[0, col].axhline(90, c='k', lw=0.7); axes[0, col].axvline(s['to_pct'], c='k', ls=':', lw=1)
        axes[0, col].set_title(f"{tag} ({cond.lower()}), stride at video t = {s['t0']:.2f} s")
        for row, (ch, name) in enumerate((('Tibialis_Raw_V', 'TA'), ('Gastro_Raw_V', 'GAS')), start=1):
            env, mk, bp = env_cache[(tag, name)]
            ax = axes[row, col]
            ax.plot(gc, d[ch][m], c='0.6', lw=0.4, label='raw (V)')
            ax2 = ax.twinx()
            e = env[m] * 1000; e_clean = np.where(mk[m], np.nan, e)
            ax2.plot(gc, e_clean, c=COL[cond], lw=2, label='envelope (mV)')
            ax2.set_ylim(0, 450)
            if mk[m].any():
                ax.fill_between(gc, 0, 3.3, where=mk[m], color='red', alpha=0.15, lw=0)
            ax.set_ylim(0, 3.3); ax.axvline(s['to_pct'], c='k', ls=':', lw=1)
            ax.set_ylabel(f'{name} raw (V)'); ax2.set_ylabel(f'{name} envelope (mV)')
        axes[0, col].set_ylabel('theta shank-sole (deg)')
        axes[2, col].set_xlabel('gait cycle (%)  [0 = heel strike, dotted = toe-off]')
    plt.tight_layout(); plt.subplots_adjust(wspace=0.45)
    plt.savefig(os.path.join(FIG_OUT, 'fig12_single_stride_raw_emg_U2_vs_C1.png'), dpi=130); plt.close()

    # ---------------------------------------------------------------- fig13: angular velocity over the gait cycle
    gcx = np.linspace(0, 100, 101)
    fig, ax = plt.subplots(figsize=(9, 4))
    for cond, tags in (('Unconstrained', ('U1', 'U2')), ('Constrained', ('C1', 'C2'))):
        M = np.array([s['om'] for t in tags for s in strides[t]])
        mu, sd = M.mean(0), M.std(0)
        ax.plot(gcx, mu, c=COL[cond], lw=2, label=f'{cond} (n={len(M)})')
        ax.fill_between(gcx, mu - sd, mu + sd, color=COL[cond], alpha=0.2)
        to_m = np.nanmean([s['to_pct'] for t in tags for s in strides[t]])
        ax.axvline(to_m, c=COL[cond], ls=':', lw=1)
    ax.axhline(0, c='k', lw=0.7)
    ax.set_xlabel('gait cycle (%)  [0 = heel strike, dotted = toe-off]')
    ax.set_ylabel('d(theta)/dt (deg/s)\n+ = towards PF, - = towards DF')
    ax.set_title('Ankle angular velocity: mean +/- SD over strides'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_OUT, 'fig13_ankle_angular_velocity_gait_cycle.png'), dpi=130); plt.close()

    # ---------------------------------------------------------------- fig14: phase-averaged bars
    phases = ['Loading (0-15%)', 'Mid/terminal stance (15-50%)', 'Push-off (50%-TO)', 'Swing (TO-100%)']
    short = ['Loading', 'Mid/term.\nstance', 'Push-off', 'Swing']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, q, lab in zip(axes, ('theta', 'TA', 'GAS'), ('mean theta (deg)', 'TA (% uncon. peak)', 'GAS (% uncon. peak)')):
        x = np.arange(4)
        for j, cond in enumerate(('Unconstrained', 'Constrained')):
            vals = [phase_tab[(p, q)][j] for p in phases]
            ax.bar(x + (j - 0.5) * 0.38, [v[0] for v in vals], 0.38, yerr=[v[1] for v in vals],
                   color=COL[cond], capsize=3, label=cond)
            if q != 'theta':                                   # angle: every stride counts (32 / 30)
                for xi, v in zip(x, vals):
                    ax.text(xi + (j - 0.5) * 0.38, 1, f'n={v[2]}', ha='center', va='bottom', fontsize=7,
                            color='white', rotation=90)
        ax.set_xticks(x); ax.set_xticklabels(short); ax.set_ylabel(lab); ax.grid(axis='y', alpha=0.3)
        if q == 'theta':
            ax.set_ylim(75, 105); ax.axhline(90, c='k', lw=0.7)
    axes[0].legend(fontsize=8, loc='upper left')
    plt.tight_layout(); plt.savefig(os.path.join(FIG_OUT, 'fig14_phase_averaged_theta_TA_GAS.png'), dpi=130); plt.close()


if __name__ == '__main__':
    main()
