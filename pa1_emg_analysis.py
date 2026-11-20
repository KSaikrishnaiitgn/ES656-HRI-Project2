"""
HRI Class Project 2 - Part A: agonist-antagonist activation at the ankle (EMG analysis).

Task: standing on the left leg, the subject repeatedly dorsiflexed (DF) and plantarflexed (PF) the
unloaded right ankle. Conditions (15 Nov 2026):
    (a) speed: normal (Trial_1), fast (Trial_2), slow (Trial_3)
    (b) stable load: 1 kg hung from the toes
    (c) unstable load: the same 1 kg weight pulled randomly by an assistant
    (d) limb configuration: three knee-angle / foot positions (Config1-3)
EMG: TA (DF agonist) and medial gastrocnemius GAS (PF agonist), same hardware and electrode sites
as Part B; recorded with the same firmware / recording script (15 s windows, 30 s for unstable).

Processing (identical to Part B, s2_emg_processing_alignment.py):
    mean removal -> notches 50 Hz + harmonics -> band-pass 40-450 Hz -> rectify -> 6 Hz low-pass.
    Only the low-frequency movement-artefact mask is applied. ADC clipping is NOT masked here: the
    clipped TA bursts are genuine EMG (median frequency 100-155 Hz, ~1 % power < 20 Hz) that
    saturated the ADC, so their envelope peaks are lower bounds.
Normalisation (no MVC): each muscle / mean per-cycle peak envelope of the normal-speed trial.
Cycles (no kinematics yet; Vicon pending): bursts are detected on each envelope
    (threshold = baseline + 20 % of (p99 - baseline), gaps < 0.15 s merged, bursts < 0.10 s dropped);
    one movement cycle = one TA burst onset to the next TA burst onset (each DF movement starts
    with a TA burst). Per cycle: period, burst peak/mean/duration, GAS timing relative to TA onset,
    antagonist co-activation and the co-contraction index CCI (Falconer & Winter 1985).

Run from 02_code:  python pa1_emg_analysis.py
Outputs: 03_results/partA/ (per-cycle CSV, summary txt), 04_report_PartA/figures/
"""
import os, csv
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config_paths as P
import s2_emg_processing_alignment as E
import s7_report_figures as F            # shared plot style

os.makedirs(P.PA_RES, exist_ok=True); os.makedirs(P.PA_REPORT_FIG, exist_ok=True)
FS = E.FS_EMG
CTA, CGAS = '#1f77b4', '#ff7f0e'

TRIALS = [  # tag, csv (relative to PA_EMG), group, label, video
    ('SN', 'Varying_Speed/Vicon_Trial_1_20261115_174208.csv', 'speed', 'Normal speed', 'V1_normal.mp4'),
    ('SS', 'Varying_Speed/Vicon_Trial_3_20261115_174638.csv', 'speed', 'Slow speed', 'V1_slow.mp4'),
    ('SF', 'Varying_Speed/Vicon_Trial_2_20261115_174443.csv', 'speed', 'Fast speed', 'V1_fast.mp4'),
    ('K1', 'Configuration/Vicon_Trial_Config1_20261115_175057.csv', 'config', 'Configuration 1', '--'),
    ('K2', 'Configuration/Vicon_Trial_Config2_20261115_180058.csv', 'config', 'Configuration 2', 'V2_config1.mp4'),
    ('K3', 'Configuration/Vicon_Trial_Config3_20261115_180236.csv', 'config', 'Configuration 3', 'V2_config_2.mp4'),
    ('LS', 'Stable_Loading/Vicon_Stable_load_20261115_180507.csv', 'load', 'Stable load (1 kg)', 'V3_stable_loading.mp4'),
    ('LU', 'Unstable_Loading/Vicon_Untable_load_20261115_180836.csv', 'load', 'Unstable load (1 kg, pulled)', 'V4_Unstable_loading.mp4'),
]
TH_FRAC, GAP = 0.20, 0.15               # onset threshold (fraction of range), merge gap (s)
MIN_DUR = {'ta': 0.20, 'gas': 0.10}      # minimum burst duration (s)
MIN_PEAK = 0.40                          # a burst must reach base + 40 % of the range


def load(rel):
    d = np.genfromtxt(os.path.join(P.PA_EMG, rel), delimiter=',', names=True)
    out = {'t': d['Time_s']}
    for ch, k in (('Tibialis_Raw_V', 'ta'), ('Gastro_Raw_V', 'gas')):
        raw = d[ch]
        env = E.emg_envelope(raw); m = E.lowfreq_mask(raw)
        sat = (raw >= E.SAT_HI) | (raw <= E.SAT_LO)
        out[k + '_raw'] = raw; out[k] = np.where(m, np.nan, env); out[k + '_mask'] = m; out[k + '_sat'] = sat
    return out


def bursts(env, t, k):
    x = np.nan_to_num(env, nan=0.0)
    base = np.percentile(x, 5); top = np.percentile(x, 99)
    thr = base + TH_FRAC * (top - base); pk = base + MIN_PEAK * (top - base)
    on = x > thr
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    seg = [[s, e] for s, e in zip(edges[::2], edges[1::2])]
    merged = []
    for s, e in seg:
        if merged and (s - merged[-1][1]) / FS < GAP:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if (e - s) / FS >= MIN_DUR[k] and x[s:e].max() >= pk], thr


def cci(a, b):
    v = ~np.isnan(a) & ~np.isnan(b)
    return 200 * np.minimum(a[v], b[v]).sum() / (a[v] + b[v]).sum() if v.mean() >= 0.5 else np.nan


def analyse():
    data, B = {}, {}
    for tag, rel, *_ in TRIALS:
        d = load(rel); data[tag] = d
        B[tag] = {k: bursts(d[k], d['t'], k) for k in ('ta', 'gas')}
    # normalisation reference: mean per-cycle TA / GAS peak in the normal-speed trial
    def cycles(tag):
        tb = B[tag]['ta'][0]
        return [(tb[i][0], tb[i + 1][0]) for i in range(len(tb) - 1)]
    ref = {}
    for k in ('ta', 'gas'):
        ref[k] = np.mean([np.nanmax(data['SN'][k][s:e]) for s, e in cycles('SN')])
    rows = []
    for tag, rel, grp, lab, vid in TRIALS:
        d = data[tag]; ta = 100 * d['ta'] / ref['ta']; gas = 100 * d['gas'] / ref['gas']
        d['ta_n'], d['gas_n'] = ta, gas
        gb = B[tag]['gas'][0]; tb = B[tag]['ta'][0]
        for c, (s, e) in enumerate(cycles(tag)):
            T = (e - s) / FS
            if np.isnan(ta[s:e]).mean() > 0.5 or np.isnan(gas[s:e]).mean() > 0.5:
                continue
            ts, te = tb[c]                                  # the TA (DF agonist) burst that opens this cycle
            # PF burst = the GAS burst that starts after the TA burst has ended (GAS acting as agonist)
            g_pf = [(a, b) for a, b in gb if te <= a < e]
            pf = g_pf[0] if g_pf else None
            pf_mask = np.zeros(e - s, bool)
            if pf:
                pf_mask[pf[0] - s:min(pf[1], e) - s] = True
            rows.append(dict(
                tag=tag, group=grp, label=lab, cycle=c + 1, period=T,
                ta_peak=np.nanmax(ta[ts:te]), ta_dur=(te - ts) / FS,
                gas_co_peak=np.nanmax(gas[ts:te]),                    # GAS (antagonist) during the TA burst
                gas_co_mean=np.nanmean(gas[ts:te]),
                gas_pf_peak=np.nanmax(gas[pf[0]:pf[1]]) if pf else np.nan,
                gas_pf_dur=(min(pf[1], e) - pf[0]) / FS if pf else np.nan,
                pf_onset_pct=100 * (pf[0] - s) / (e - s) if pf else np.nan,
                ta_during_pf=np.nanmean(ta[s:e][pf_mask]) if pf else np.nan,   # TA (antagonist) during the PF burst
                cci=cci(ta[s:e], gas[s:e]),
                cci_df=cci(ta[ts:te], gas[ts:te]),
                cci_pf=cci(ta[pf[0]:pf[1]], gas[pf[0]:pf[1]]) if pf else np.nan,
                ta_sat=100 * d['ta_sat'][s:e].mean()))
    return data, B, ref, rows, cycles


KEYS = [('period', 'Cycle period (s)', '{:.2f}'), ('ta_peak', 'TA burst peak (%)', '{:.0f}'),
        ('ta_dur', 'TA burst duration (s)', '{:.2f}'), ('gas_co_peak', 'GAS peak during TA (%)', '{:.0f}'),
        ('gas_co_mean', 'GAS mean during TA (%)', '{:.1f}'), ('gas_pf_peak', 'GAS PF-burst peak (%)', '{:.0f}'),
        ('gas_pf_dur', 'GAS PF-burst duration (s)', '{:.2f}'), ('pf_onset_pct', 'PF burst onset (% cycle)', '{:.0f}'),
        ('ta_during_pf', 'TA mean during PF burst (%)', '{:.1f}'), ('cci', 'CCI whole cycle (%)', '{:.1f}'),
        ('cci_df', 'CCI during TA burst (%)', '{:.1f}'), ('cci_pf', 'CCI during PF burst (%)', '{:.1f}'),
        ('ta_sat', 'TA samples clipped (%)', '{:.1f}')]


def summarise(rows, ref, data):
    lines = [f"Normalisation reference (normal speed, mean per-cycle peak): TA {1000 * ref['ta']:.1f} mV, GAS {1000 * ref['gas']:.1f} mV", '']
    lines.append(f"{'trial':6s}{'n':>4s}" + ''.join(f"{k:>16s}" for k, _, _ in KEYS))
    summ = {}
    for tag, rel, grp, lab, vid in TRIALS:
        R = [r for r in rows if r['tag'] == tag]; summ[tag] = {}
        line = f"{tag:6s}{len(R):4d}"
        for k, _, fmt in KEYS:
            v = np.array([r[k] for r in R], float); v = v[~np.isnan(v)]
            m, s = (v.mean(), v.std()) if len(v) else (np.nan, np.nan)
            summ[tag][k] = (m, s, len(v))
            line += f"{fmt.format(m):>9s}+/-{fmt.format(s):<5s}"[:16].rjust(16)
        lines.append(line)
    lines.append('')
    for tag, *_ in TRIALS:
        d = data[tag]
        lines.append(f"{tag}: duration {len(d['t']) / FS:.1f} s; low-frequency artefact masked TA {100 * d['ta_mask'].mean():.1f} %, "
                     f"GAS {100 * d['gas_mask'].mean():.1f} %; clipped samples TA {100 * d['ta_sat'].mean():.1f} %, GAS {100 * d['gas_sat'].mean():.2f} %")
    open(os.path.join(P.PA_RES, 'partA_summary.txt'), 'w').write('\n'.join(lines) + '\n')
    with open(os.path.join(P.PA_RES, 'partA_per_cycle.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('\n'.join(lines))
    return summ


# --------------------------------------------------------------------------------------- figures
def fig_method(data, B):
    d = data['SN']; t = d['t']; sel = (t >= 3) & (t <= 11)
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True)
    for ax, k, name, col in ((axs[0], 'ta', 'TA', CTA), (axs[1], 'gas', 'GAS', CGAS)):
        ax.plot(t[sel], d[k + '_raw'][sel], c='0.6', lw=0.3)
        sat = d[k + '_sat'] & sel
        ax.plot(t[sat], d[k + '_raw'][sat], '.', ms=1.5, c='k', label='clipped samples (ADC rail)')
        ax.set_ylim(0, 3.3); ax.set_ylabel(f'{name} raw (V)')
        ax2 = ax.twinx(); ax2.grid(False); ax2.spines['right'].set_visible(True)
        ax2.plot(t[sel], 1000 * d[k][sel], c=col, lw=1.5)
        ax2.set_ylabel(f'{name} envelope (mV)')
        bl, thr = B['SN'][k]
        ax2.axhline(1000 * thr, c=col, ls=':', lw=0.9)
        for s, e in bl:
            if t[e - 1] >= 3 and t[s] <= 11:
                ax.axvspan(t[s], t[e - 1], color=col, alpha=0.10, lw=0)
        if k == 'ta':
            ax.legend(loc='upper left', fontsize=7, frameon=False)
    axs[0].set_title('(a) tibialis anterior', loc='left', fontsize=9)
    axs[1].set_title('(b) medial gastrocnemius', loc='left', fontsize=9)
    axs[1].set_xlabel('Time (s), normal-speed trial'); axs[1].set_xlim(3, 11)
    fig.tight_layout(h_pad=0.6); F.save_to(fig, P.PA_REPORT_FIG, 'figA_method_example.png')


def fig_overview(data, B):
    fig, axs = plt.subplots(len(TRIALS), 1, figsize=(7.0, 9.0), sharex=True)
    for ax, (tag, rel, grp, lab, vid) in zip(axs, TRIALS):
        d = data[tag]; t = d['t']
        ax.plot(t, d['ta_n'], c=CTA, lw=1.0, label='TA'); ax.plot(t, d['gas_n'], c=CGAS, lw=1.0, label='GAS')
        for s, e in B[tag]['ta'][0]:
            ax.axvline(t[s], c=CTA, lw=0.4, alpha=0.6)
        ax.set_ylabel(lab.replace(' (1 kg, pulled)', '\n(pulled)').replace(' (1 kg)', '').replace('Configuration ', 'Config. '),
                      fontsize=7, rotation=0, ha='right', va='center')
        ax.set_ylim(0, 260); ax.tick_params(labelsize=7)
    axs[0].legend(loc='upper right', ncol=2, fontsize=7, frameon=False)
    axs[-1].set_xlabel('Time (s)'); axs[-1].set_xlim(0, 30)
    fig.text(0.015, 0.5, 'Envelope (% of normal-speed reference peak)', rotation=90, va='center', fontsize=8)
    fig.tight_layout(h_pad=0.25, rect=(0.03, 0, 1, 1)); F.save_to(fig, P.PA_REPORT_FIG, 'figA_envelopes_all_trials.png')


def cycle_profiles(d, B, cycles_fn, tag):
    out = {'ta': [], 'gas': []}
    for s, e in cycles_fn(tag):
        x = np.linspace(s, e - 1, 101)
        for k in ('ta', 'gas'):
            seg = d[k + '_n']
            if np.isnan(seg[s:e]).mean() > 0.5:
                continue
            out[k].append(np.interp(x, np.arange(len(seg)), seg))
    return {k: np.array(v) for k, v in out.items()}


def fig_profiles(data, B, cycles_fn):
    groups = [('speed', ['SS', 'SN', 'SF'], ['#9ecae1', '#3182bd', '#08306b']),
              ('load', ['SN', 'LS', 'LU'], ['#3182bd', '#31a354', '#de2d26']),
              ('config', ['K1', 'K2', 'K3'], ['#756bb1', '#e6550d', '#636363'])]
    titles = {'speed': 'Speed', 'load': 'Loading', 'config': 'Limb configuration'}
    lab = {t[0]: t[3] for t in TRIALS}
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.8), sharex=True, sharey='row')
    for j, (g, tags, cols) in enumerate(groups):
        for tag, col in zip(tags, cols):
            pr = cycle_profiles(data[tag], B, cycles_fn, tag)
            for i, k in enumerate(('ta', 'gas')):
                M = pr[k]
                if len(M) == 0:
                    continue
                mu, sd = M.mean(0), M.std(0); x = np.linspace(0, 100, 101)
                axs[i, j].plot(x, mu, c=col, lw=1.4, label=f"{lab[tag].replace(' (1 kg, pulled)', '').replace(' (1 kg)', '')} ($n={len(M)}$)")
                axs[i, j].fill_between(x, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
        axs[0, j].set_title(f'({"abc"[j]}) {titles[g]}', loc='left', fontsize=9)
        axs[0, j].legend(fontsize=6.3, frameon=False, loc='upper right')
        axs[1, j].set_xlabel('Movement cycle (%)\n0 = TA burst onset')
    axs[0, 0].set_ylabel('TA (% ref.)'); axs[1, 0].set_ylabel('GAS (% ref.)')
    fig.tight_layout(h_pad=0.6, w_pad=0.6); F.save_to(fig, P.PA_REPORT_FIG, 'figA_cycle_profiles.png')


def fig_metrics(rows):
    keys = [('period', 'Cycle period (s)'), ('ta_peak', 'TA (DF agonist) burst peak\n(% ref.)'),
            ('gas_pf_peak', 'GAS (PF agonist) burst peak\n(% ref.)'), ('gas_co_mean', 'GAS (antagonist) mean\nduring TA burst (% ref.)'),
            ('ta_during_pf', 'TA (antagonist) mean\nduring PF burst (% ref.)'), ('cci', 'Co-contraction index,\nwhole cycle (%)')]
    order = [t[0] for t in TRIALS]; names = ['N', 'S', 'F', 'K1', 'K2', 'K3', 'LS', 'LU']
    gcol = {'speed': '#3182bd', 'config': '#756bb1', 'load': '#de2d26'}
    grp = {t[0]: t[2] for t in TRIALS}
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6))
    rng = np.random.default_rng(2)
    for ax, (k, lab), l in zip(axs.ravel(), keys, 'abcdef'):
        for i, tag in enumerate(order):
            v = np.array([r[k] for r in rows if r['tag'] == tag], float); v = v[~np.isnan(v)]
            if not len(v):
                continue
            ax.plot(i + rng.uniform(-0.15, 0.15, len(v)), v, 'o', ms=2.5, mfc=gcol[grp[tag]], mec='none', alpha=0.5)
            ax.errorbar(i, v.mean(), yerr=v.std(), fmt='_', ms=12, mew=1.8, c='k', capsize=3, lw=1)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(names, fontsize=7)
        for x in (2.5, 5.5):
            ax.axvline(x, c='0.7', lw=0.6)
        ax.set_ylabel(lab, fontsize=7.5); ax.grid(axis='x', visible=False)
        ax.set_title(f'({l})', loc='left', fontsize=9)
    fig.tight_layout(h_pad=1.0, w_pad=0.8); F.save_to(fig, P.PA_REPORT_FIG, 'figA_condition_metrics.png')


if __name__ == '__main__':
    data, B, ref, rows, cycles_fn = analyse()
    summarise(rows, ref, data)
    fig_method(data, B); fig_overview(data, B); fig_profiles(data, B, cycles_fn); fig_metrics(rows)
