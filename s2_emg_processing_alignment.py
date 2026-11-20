"""
HRI Class Project 2 - Part B: EMG processing, EMG-video alignment and gait-cycle analysis.

Trials (videos started first, then EMG (20 s), EMG stopped, then video stopped):
    U1 = Video1 + Walking_no_constraint_marker_20261116_150748.csv     (unconstrained, trial 1)
    U2 = Video2 + Walking_no_constraint_marker2_20261116_151014.csv    (unconstrained, trial 2)
    C1 = Video3 + Walking_constraint_marker_20261116_153838.csv        (constrained,   trial 1)
    C2 = Video4 + Walking_constraint_marker2_20261116_154034.csv       (constrained,   trial 2)
(Walking_no_constraint_20261116_145923.csv has no matching video -> not used.)

1. EMG processing (raw channel, fs = 2000 Hz), zero-phase FFT-domain filters:
     remove mean -> notch 50 Hz and harmonics up to 450 Hz (strong 100/200/400 Hz interference)
     -> band-pass 40-450 Hz -> full-wave rectify -> low-pass 6 Hz (2nd-order Butterworth magnitude,
        applied forward-backward) = linear envelope.
   Artefact rejection: (a) amplifier clipping held >= 5 ms at a rail (+-100 ms), (b) 100 ms windows
   where RMS(<20 Hz) > RMS(EMG band) (+-50 ms). Masked samples are excluded from all metrics.
   Normalisation (no MVC was recorded): each muscle is divided by the mean per-stride peak envelope
   of the two unconstrained trials -> 100 % = typical unconstrained peak.

2. Alignment (offset D = time of EMG sample 0 on the video clock):
   a) Timestamps: EMG file name = EMG start time (laptop clock). MP4 'mvhd' creation time = end of
      the video recording (phone clock) -> video start = creation - duration.
   b) Refinement: cross-correlation of the gastrocnemius envelope with the ankle plantarflexion
      velocity from the video (push-off), allowing an electromechanical delay of 0.10 s
      (GAS activity leads plantarflexion). Searched within +-2 s of the timestamp estimate
      (for U1, whose file was re-exported, over every lag that keeps the EMG inside the video).

3. Gait events from the video: heel strike (HS) = maximum toe-up pitch of the sole after swing;
   toe-off (TO) approximated by peak plantarflexion (maximum theta). Strides HS->HS, 0-100 %.
   Metrics per stride: theta peaks, TA mean in swing, TA peak in loading (0-15 %), GAS peak and
   mean in stance, GAS peak timing, co-contraction index CCI = 2*sum(min(TA,GAS))/sum(TA+GAS)*100.
"""
import os, csv, datetime
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config_paths as P
BASE = P.ROOT
EMG_DIR = P.RAW_EMG
KIN_DIR = P.RES_KIN
OUT_EMG = P.RES_EMG; os.makedirs(OUT_EMG, exist_ok=True)
OUT_FIG = P.RES_FIG; os.makedirs(OUT_FIG, exist_ok=True)

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
TRIALS = {
    # tag: (video, emg file, video mvhd creation time (IST), mvhd duration s, condition)
    'U1': ('Video1.mp4', 'Walking_no_constraint_marker_20261116_150748.csv', '15:07:40', 28.78, 'Unconstrained'),
    'U2': ('Video2.mp4', 'Walking_no_constraint_marker2_20261116_151014.csv', '15:10:43', 43.53, 'Unconstrained'),
    'C1': ('Video3.mp4', 'Walking_constraint_marker_20261116_153838.csv', '15:39:06', 71.26, 'Constrained'),
    'C2': ('Video4.mp4', 'Walking_constraint_marker2_20261116_154034.csv', '15:41:05', 53.03, 'Constrained'),
}
EMD = 0.10          # s, assumed electromechanical delay (GAS activity -> plantarflexion velocity)
HP_CUT = 40.0       # Hz, EMG high-pass. 40 Hz instead of the usual 20 Hz: the constrained trials carry
                    # wire/cable movement artefact up to ~40 Hz, while clean EMG here has a median
                    # frequency of 100-200 Hz. Applied identically to all trials.
FS_EMG = 2000.0


# ------------------------------------------------------------------ filtering helpers
def fft_filter(x, fs, gain_fn):
    """Zero-phase filtering by multiplying the spectrum with a real gain (reflect-padded)."""
    n = len(x); pad = n // 4
    xp = np.r_[x[pad:0:-1], x, x[-2:-pad - 2:-1]]
    X = np.fft.rfft(xp); f = np.fft.rfftfreq(len(xp), 1 / fs)
    y = np.fft.irfft(X * gain_fn(f), len(xp))
    return y[pad:pad + n]


def butter_mag2(f, fc, order=2, kind='low'):
    r = (f / fc) if kind == 'low' else (fc / np.maximum(f, 1e-9))
    return 1.0 / (1.0 + r ** (2 * order))            # |H|^2 = forward-backward Butterworth


def emg_envelope(raw, fs=FS_EMG):
    x = raw - raw.mean()

    def gain(f):
        g = butter_mag2(f, HP_CUT, 4, 'high') * butter_mag2(f, 450, 4, 'low')
        for h in range(50, 500, 50):                 # notch 50 Hz and harmonics (+-1.5 Hz)
            g = g * (1 - np.exp(-0.5 * ((f - h) / 0.75) ** 2))
        return g
    x = fft_filter(x, fs, gain)
    env = fft_filter(np.abs(x), fs, lambda f: butter_mag2(f, 6, 2, 'low'))
    return np.maximum(env, 0)


SAT_HI, SAT_LO, SAT_PAD = 3.28, 0.02, 0.10      # amplifier rails (0-3.3 V) and mask padding (s)


SAT_MIN_RUN = 10    # samples (5 ms): a sustained flat stretch at a rail = artefact


def saturation_mask(raw, fs=FS_EMG):
    """Movement-artefact mask. Strong EMG occasionally touches the 3.3 V rail for 1-2 samples
    (kept: that is real muscle activity). Artefacts in the constrained trials (heel strike) hold
    the amplifier at a rail for many milliseconds; only runs of >= SAT_MIN_RUN clipped samples
    are treated as artefact, and the mask is dilated by SAT_PAD on both sides."""
    sat = (raw >= SAT_HI) | (raw <= SAT_LO)
    edges = np.flatnonzero(np.diff(np.r_[0, sat.astype(int), 0]))
    art = np.zeros_like(sat)
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s >= SAT_MIN_RUN:
            art[s:e] = True
    k = int(SAT_PAD * fs)
    return np.convolve(art.astype(float), np.ones(2 * k + 1), 'same') > 0, art


LF_RATIO, LF_WIN, LF_PAD = 1.0, 0.10, 0.05   # low-frequency artefact test: RMS(<20 Hz)/RMS(20-450 Hz)


def lowfreq_mask(raw, fs=FS_EMG):
    """Movement artefact that does not clip: windows (100 ms) where the < 20 Hz content is large
    relative to the EMG band. Clean EMG in these recordings has ~1 % of its power below 20 Hz
    (ratio ~0.1); impact artefacts have 35-60 % (ratio ~0.7-1.2). The constrained recordings also
    carry a lower-level cable/wire artefact through the whole stride, so only windows where the
    artefact dominates (ratio > 1.0) are masked; the rest is cleaned by the 40 Hz high-pass."""
    x = raw - raw.mean()
    lf = fft_filter(x, fs, lambda f: butter_mag2(f, 20, 4, 'low') * (f > 1))
    hf = fft_filter(x, fs, lambda f: butter_mag2(f, 20, 4, 'high') * butter_mag2(f, 450, 4, 'low')
                    * np.prod([1 - np.exp(-0.5 * ((f - h) / 0.75) ** 2) for h in range(50, 500, 50)], axis=0))
    k = int(LF_WIN * fs)
    box = np.ones(k) / k
    ratio = np.sqrt(np.convolve(lf ** 2, box, 'same') / (np.convolve(hf ** 2, box, 'same') + 1e-12))
    p = int(LF_PAD * fs)
    return np.convolve((ratio > LF_RATIO).astype(float), np.ones(2 * p + 1), 'same') > 0


def load_emg(fname, with_masks=False):
    """Return time and the TA/GAS linear envelopes (V); artefact regions (clipping or
    low-frequency movement artefact) are set to NaN."""
    d = np.genfromtxt(os.path.join(EMG_DIR, fname), delimiter=',', names=True)
    t = d['Time_s']
    out, masks = [], []
    for ch in ('Tibialis_Raw_V', 'Gastro_Raw_V'):
        env = emg_envelope(d[ch])
        m, sat = saturation_mask(d[ch])
        m = m | lowfreq_mask(d[ch])
        env[m] = np.nan
        out.append(env); masks.append((m, sat))
    if with_masks:
        return t, out[0], out[1], masks
    return t, out[0], out[1]


def load_kin(tag):
    d = np.genfromtxt(os.path.join(KIN_DIR, f'{tag}_ankle.csv'), delimiter=',', names=True)
    cal = dict(l.strip().split('=') for l in open(os.path.join(KIN_DIR, f'{tag}_calibration.txt')))
    fps = float(cal['fps']); off = float(cal['sticker_to_sole_offset_deg'])
    pitch = np.degrees(np.arctan2(d['heel_y'] - d['toe_y'], d['toe_x'] - d['heel_x'])) - off
    return d['time_s'], d['theta_shank_sole_filt_deg'], pitch, fps


def local_extrema(x, t, mode, min_sep, thr):
    idx = []
    for i in range(1, len(x) - 1):
        ok = (x[i] >= x[i - 1] and x[i] > x[i + 1]) if mode == 'max' else (x[i] <= x[i - 1] and x[i] < x[i + 1])
        if ok and ((x[i] > thr) if mode == 'max' else (x[i] < thr)):
            if idx and t[i] - t[idx[-1]] < min_sep:
                better = x[i] > x[idx[-1]] if mode == 'max' else x[i] < x[idx[-1]]
                if better:
                    idx[-1] = i
            else:
                idx.append(i)
    return np.array(idx, int)


def gait_events(t, theta, pitch):
    """Events from the sole pitch (sole angle to the belt, + toe-up):
    HS = maximum toe-up pitch at the end of swing (initial contact);
    TO = minimum pitch (most toe-down) between successive heel strikes (approximate toe-off).
    theta is not used: with the constraint the largest plantarflexion occurs at loading, not push-off."""
    hs = local_extrema(pitch, t, 'max', 0.8, np.nanpercentile(pitch, 75))
    to = []
    for a, b in zip(hs[:-1], hs[1:]):
        to.append(a + int(np.nanargmin(pitch[a:b])))
    return hs, np.array(to, int)


def nan_corr(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 50 else np.nan


# ------------------------------------------------------------------ main
def main():
    emg, kin, align = {}, {}, {}
    sat_info = {}
    for tag, (vid, ef, ctime, vdur, cond) in TRIALS.items():
        t_e, ta, gas, masks = load_emg(ef, with_masks=True)
        emg[tag] = (t_e, ta, gas)
        sat_info[tag] = masks
        kin[tag] = load_kin(tag)

        # ---- timestamp estimate of D (EMG start on the video clock)
        emg_start = datetime.datetime.strptime(ef.rsplit('_', 2)[-2] + ef.rsplit('_', 1)[-1][:6], '%Y%m%d%H%M%S')
        vend = datetime.datetime.strptime('20261116' + ctime.replace(':', ''), '%Y%m%d%H%M%S')
        D_ts = (emg_start - vend).total_seconds() + vdur          # creation = end of recording
        t_v, theta, pitch, fps = kin[tag]
        vlen = t_v[-1]
        emg_len = t_e[-1]
        ts_valid = 0 <= D_ts <= vlen - emg_len
        if ts_valid:
            grid = np.arange(D_ts - 2.5, D_ts + 2.5001, 0.01)
        else:
            grid = np.arange(0, vlen - emg_len + 0.0001, 0.01)
        grid = grid[(grid >= 0) & (grid <= vlen - emg_len)]

        # ---- refinement
        # Method A (all trials): corr( GAS(t - D), PF velocity(t + EMD) )
        pfv = np.maximum(np.gradient(theta, t_v), 0)               # deg/s, plantarflexion velocity
        pfv_shift = np.interp(t_v + EMD, t_v, pfv)
        # Method B (trials with heel-strike artefacts): corr( clipping rate(t - D), heel-strike train(t) )
        hs, _ = gait_events(t_v, theta, pitch)
        hs_train = np.zeros_like(t_v)
        for i in hs:
            hs_train += np.exp(-0.5 * ((t_v - t_v[i]) / 0.05) ** 2)
        sat_any = (sat_info[tag][0][1] | sat_info[tag][1][1]).astype(float)
        k = int(0.05 * FS_EMG)
        sat_rate = np.convolve(sat_any, np.ones(2 * k + 1) / (2 * k + 1), 'same')
        use_artefact = sat_any.mean() > 0.01                       # >1 % clipped samples
        rA, rB = [], []
        for D in grid:
            m = (t_v >= D) & (t_v <= D + emg_len)
            rA.append(nan_corr(np.interp(t_v[m] - D, t_e, gas), pfv_shift[m]))
            rB.append(nan_corr(np.interp(t_v[m] - D, t_e, sat_rate), hs_train[m]) if use_artefact else np.nan)
        rA, rB = np.array(rA), np.array(rB)
        r = rB if use_artefact else rA
        D_best = grid[np.nanargmax(r)]
        DA = grid[np.nanargmax(rA)]
        align[tag] = dict(D_ts=D_ts, ts_valid=ts_valid, D=D_best, r=np.nanmax(r), grid=grid, rcurve=r,
                          rA=rA, rB=rB, DA=DA, method='heel-strike artefact' if use_artefact else 'GAS vs PF velocity',
                          sat_pct=100 * sat_any.mean())

    # ---- normalisation constants from unconstrained strides
    def stride_table(tag):
        t_v, theta, pitch, fps = kin[tag]; t_e, ta, gas = emg[tag]; D = align[tag]['D']
        hs, to = gait_events(t_v, theta, pitch)
        rows = []
        for a, b in zip(hs[:-1], hs[1:]):
            if t_v[a] < D or t_v[b] > D + t_e[-1]:
                continue                                           # stride must be inside EMG window
            tt = np.linspace(t_v[a], t_v[b], 101)
            th = np.interp(tt, t_v, theta)
            ta_s = np.interp(tt - D, t_e, ta); gas_s = np.interp(tt - D, t_e, gas)
            to_in = to[(to > a) & (to < b)]
            to_pct = 100 * (t_v[to_in[0]] - t_v[a]) / (t_v[b] - t_v[a]) if len(to_in) else np.nan
            rows.append(dict(t0=t_v[a], T=t_v[b] - t_v[a], th=th, ta=ta_s, gas=gas_s, to_pct=to_pct))
        return rows

    strides = {tag: stride_table(tag) for tag in TRIALS}
    norm_ta = np.nanmean([np.nanmax(s['ta']) for tag in ('U1', 'U2') for s in strides[tag]])
    norm_gas = np.nanmean([np.nanmax(s['gas']) for tag in ('U1', 'U2') for s in strides[tag]])
    for tag in TRIALS:
        for s in strides[tag]:
            s['ta'] = 100 * s['ta'] / norm_ta; s['gas'] = 100 * s['gas'] / norm_gas

    # ---- write aligned EMG on the video timeline (for the rendered videos)
    for tag in TRIALS:
        t_v, theta, pitch, fps = kin[tag]; t_e, ta, gas = emg[tag]; D = align[tag]['D']
        inside = (t_v >= D) & (t_v <= D + t_e[-1])
        ta_v = np.where(inside, 100 * np.interp(t_v - D, t_e, ta) / norm_ta, np.nan)
        gas_v = np.where(inside, 100 * np.interp(t_v - D, t_e, gas) / norm_gas, np.nan)
        art = inside & (np.isnan(ta_v) | np.isnan(gas_v))
        hs, to = gait_events(t_v, theta, pitch)
        with open(os.path.join(OUT_EMG, f'{tag}_emg_on_video_timeline.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['frame', 'video_time_s', 'emg_time_s', 'theta_deg', 'TA_pct', 'GAS_pct',
                        'emg_artefact', 'heel_strike', 'toe_off'])
            hs_set, to_set = set(hs.tolist()), set(to.tolist())
            f2 = lambda v: '' if np.isnan(v) else f'{v:.2f}'
            for i in range(len(t_v)):
                w.writerow([i, f'{t_v[i]:.4f}', f'{t_v[i]-D:.4f}' if inside[i] else '', f'{theta[i]:.2f}',
                            f2(ta_v[i]), f2(gas_v[i]), int(art[i]), int(i in hs_set), int(i in to_set)])

    # ---- metrics per stride and per trial
    def win_stat(x, fn, min_valid=0.5):
        """Statistic over the valid (non-artefact) samples; NaN if < min_valid of the window is valid."""
        v = ~np.isnan(x)
        return fn(x[v]) if v.mean() >= min_valid else np.nan

    def metrics(s):
        ta, gas, th = s['ta'], s['gas'], s['th']
        to = int(round(s['to_pct'])) if np.isfinite(s['to_pct']) else 62
        both = ~np.isnan(ta) & ~np.isnan(gas)
        cci = 200 * np.minimum(ta, gas)[both].sum() / (ta + gas)[both].sum() if both.mean() >= 0.5 else np.nan
        g_st = gas[:to + 1]
        po_peak = th[50:86].max()                         # push-off plantarflexion peak (50-85 % GC)
        st_min = th[15:61].min()                          # late-stance dorsiflexion peak
        return dict(T=s['T'], th_min=th.min(), th_max=th.max(), rom=th.max() - th.min(),
                    po_peak=po_peak, po_exc=po_peak - st_min, lr_pf=th[:16].max() - th[0],
                    to_pct=s['to_pct'],
                    ta_swing=win_stat(ta[to:], np.mean), ta_load=win_stat(ta[:16], np.max),
                    ta_mean=win_stat(ta, np.mean),
                    gas_peak=win_stat(g_st, np.max), gas_stance=win_stat(g_st, np.mean),
                    gas_peak_pct=float(np.nanargmax(g_st)) if (~np.isnan(g_st)).mean() >= 0.5 else np.nan,
                    gas_mean=win_stat(gas, np.mean), cci=cci,
                    artefact_pct=100 * (1 - both.mean()))

    lines = ["EMG-video alignment (D = EMG start on the video clock, s)",
             "tag  timestamp-D valid  final-D  shift   method                 r      clipped%  (GAS-method D)"]
    for tag, a in align.items():
        lines.append(f"{tag}   {a['D_ts']:8.2f}   {str(a['ts_valid']):5s} {a['D']:8.2f}  "
                     f"{(a['D']-a['D_ts']) if a['ts_valid'] else float('nan'):+6.2f}   {a['method']:21s} {a['r']:.3f}"
                     f"   {a['sat_pct']:5.2f}     ({a['DA']:.2f})")
    lines += ["", f"Normalisation: 100 % = mean per-stride peak of unconstrained trials "
              f"(TA {norm_ta*1000:.1f} mV, GAS {norm_gas*1000:.1f} mV envelope)", ""]
    keys = [('T', 'stride time (s)'), ('th_min', 'min theta = peak DF (deg)'),
            ('th_max', 'max theta = peak PF (deg)'), ('rom', 'ankle ROM per stride (deg)'),
            ('po_peak', 'push-off peak theta 50-85% (deg)'), ('po_exc', 'push-off PF excursion (deg)'),
            ('lr_pf', 'loading-response PF 0-15% (deg)'),
            ('to_pct', 'toe-off (~peak PF) (% GC)'),
            ('ta_swing', 'TA mean in swing (%)'), ('ta_load', 'TA peak 0-15 % GC (%)'),
            ('ta_mean', 'TA mean whole cycle (%)'), ('gas_peak', 'GAS peak in stance (%)'),
            ('gas_stance', 'GAS mean in stance (%)'), ('gas_peak_pct', 'GAS peak timing (% GC)'),
            ('gas_mean', 'GAS mean whole cycle (%)'), ('cci', 'co-contraction index CCI (%)'),
            ('artefact_pct', 'EMG masked as artefact (% GC)')]
    table = {}
    for tag in TRIALS:
        ms = [metrics(s) for s in strides[tag]]
        table[tag] = {k: (np.nanmean([m[k] for m in ms]), np.nanstd([m[k] for m in ms])) for k, _ in keys}
        table[tag]['n'] = len(ms)
    cond_ms = {}
    for cond, tags in (('Unconstrained', ('U1', 'U2')), ('Constrained', ('C1', 'C2'))):
        ms = [metrics(s) for tag in tags for s in strides[tag]]
        cond_ms[cond] = ms
    hdr = f"{'metric':34s}" + ''.join(f"{t:>14s}" for t in TRIALS) + f"{'Uncon (all)':>16s}{'Con (all)':>16s}{'change':>10s}"
    lines += ["Per-stride metrics, mean +/- SD", hdr]
    lines.append(f"{'strides in EMG window':34s}" + ''.join(f"{table[t]['n']:>14d}" for t in TRIALS)
                 + f"{len(cond_ms['Unconstrained']):>16d}{len(cond_ms['Constrained']):>16d}")
    for k, name in keys:
        u = np.array([m[k] for m in cond_ms['Unconstrained']]); c = np.array([m[k] for m in cond_ms['Constrained']])
        row = f"{name:34s}" + ''.join(f"{table[t][k][0]:8.1f}+/-{table[t][k][1]:<4.1f}" for t in TRIALS)
        row += f"{np.nanmean(u):9.1f}+/-{np.nanstd(u):<5.1f}{np.nanmean(c):9.1f}+/-{np.nanstd(c):<5.1f}"
        row += f"{np.nanmean(c)-np.nanmean(u):+9.1f}"
        lines.append(row)
    lines += ["", "Clean (non-artefact) strides used per metric (unconstrained / constrained):"]
    for k, name in keys:
        nu = int(np.sum(np.isfinite([m[k] for m in cond_ms['Unconstrained']])))
        nc = int(np.sum(np.isfinite([m[k] for m in cond_ms['Constrained']])))
        lines.append(f"  {name:34s} {nu:3d} / {nc:3d}")
    lines += ["", "theta = shank-sole angle (90 = neutral, <90 DF, >90 PF). GC = gait cycle, 0 % = heel strike.",
              "TO is approximated by the minimum sole pitch (most toe-down), so swing = TO..100 %.",
              "Constrained theta is measured against the shoe outsole; any heel-to-toe drop of the shoe makes the",
              "constrained absolute angles read more dorsiflexed than the anatomical angle (excursions are unaffected).",
              f"Alignment assumes an electromechanical delay of {EMD:.2f} s between GAS activity and plantarflexion velocity."]
    txt = "\n".join(lines); print(txt)
    open(os.path.join(OUT_EMG, 'emg_summary.txt'), 'w').write(txt + "\n")

    # per-stride CSV
    with open(os.path.join(OUT_EMG, 'per_stride_metrics.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); ks = [k for k, _ in keys]
        w.writerow(['trial', 'condition', 'stride', 'start_video_s'] + ks)
        for tag in TRIALS:
            for i, s in enumerate(strides[tag]):
                m = metrics(s)
                w.writerow([tag, TRIALS[tag][4], i + 1, f"{s['t0']:.3f}"] + [f"{m[k]:.3f}" for k in ks])

    # ---- figures
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.5))
    for ax, (tag, a) in zip(axes, align.items()):
        ax.plot(a['grid'], a['rA'], label='GAS vs PF velocity')
        if np.isfinite(a['rB']).any():
            ax.plot(a['grid'], a['rB'], label='clipping vs heel strikes')
        ax.axvline(a['D'], c='r', label=f"final {a['D']:.2f} s")
        if a['ts_valid']:
            ax.axvline(a['D_ts'], c='k', ls='--', label=f"timestamp {a['D_ts']:.2f} s")
        ax.set_title(f"{tag}: alignment correlation"); ax.set_xlabel('EMG start on video clock (s)'); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_FIG, 'emg_video_alignment.png'), dpi=110); plt.close()

    gc = np.arange(101)
    cols = {'Unconstrained': 'tab:blue', 'Constrained': 'tab:red'}
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for cond, tags in (('Unconstrained', ('U1', 'U2')), ('Constrained', ('C1', 'C2'))):
        S = [s for tag in tags for s in strides[tag]]
        for ax, key, lab in zip(axes, ('th', 'ta', 'gas'),
                                ('theta shank-sole (deg)', 'Tibialis anterior (% uncon. peak)', 'Gastrocnemius (% uncon. peak)')):
            M = np.array([s[key] for s in S]); mu, sd = np.nanmean(M, 0), np.nanstd(M, 0)
            few = (~np.isnan(M)).mean(0) < 0.4             # hide points with < 40 % clean strides
            mu[few] = np.nan; sd[few] = np.nan
            ax.plot(gc, mu, c=cols[cond], lw=2, label=f"{cond} (n={len(S)})")
            ax.fill_between(gc, mu - sd, mu + sd, color=cols[cond], alpha=0.2)
            ax.set_ylabel(lab); ax.grid(alpha=0.3)
        to_mean = np.nanmean([s['to_pct'] for s in S])
        for ax in axes:
            ax.axvline(to_mean, c=cols[cond], ls=':', lw=1)
    axes[0].axhline(90, c='k', lw=0.8); axes[0].legend(); axes[2].set_xlabel('gait cycle (%)  [0 = heel strike, dotted = toe-off]')
    axes[0].set_title('Right ankle: mean +/- SD over strides')
    plt.tight_layout(); plt.savefig(os.path.join(OUT_FIG, 'gait_cycle_theta_TA_GAS.png'), dpi=120); plt.close()

    # bar chart of key metrics
    sel = [('ta_swing', 'TA swing mean (%)'), ('ta_load', 'TA loading peak (%)'), ('gas_peak', 'GAS stance peak (%)'),
           ('cci', 'CCI (%)'), ('th_max', 'peak PF theta (deg)'), ('th_min', 'peak DF theta (deg)')]
    fig, axes = plt.subplots(1, len(sel), figsize=(18, 3.5))
    for ax, (k, name) in zip(axes, sel):
        # strides whose EMG window was masked as artefact carry NaN: drop them (nan-aware stats)
        vals = [np.array([m[k] for m in cond_ms[c]], dtype=float) for c in ('Unconstrained', 'Constrained')]
        vals = [v[~np.isnan(v)] for v in vals]
        means = [np.mean(v) if len(v) else np.nan for v in vals]
        sds = [np.std(v) if len(v) > 1 else 0.0 for v in vals]
        ax.bar([0, 1], means, yerr=sds, color=['tab:blue', 'tab:red'], capsize=4)
        ax.set_xticks([0, 1]); ax.set_xticklabels([f'Uncon.\nn={len(vals[0])}', f'Con.\nn={len(vals[1])}'])
        ax.set_title(name, fontsize=10)
        if k.startswith('th'):
            ax.set_ylim(np.nanmin(means) - 15, np.nanmax(means) + 8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_FIG, 'condition_comparison.png'), dpi=110); plt.close()

    # save alignment numbers for the renderer
    with open(os.path.join(OUT_EMG, 'alignment.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['trial', 'D_timestamp_s', 'timestamp_valid', 'D_refined_s', 'r'])
        for tag, a in align.items():
            w.writerow([tag, f"{a['D_ts']:.3f}", a['ts_valid'], f"{a['D']:.3f}", f"{a['r']:.4f}"])


if __name__ == '__main__':
    main()
