"""
HRI Class Project 2 - Part B: figures for the revised report (04_report/).

Re-uses the processing of s2_emg_processing_alignment.py (filters, artefact masks, gait events, alignment,
normalisation) and the device model of s4_spring_shoe_model.py. Nothing in the analysis is changed:
the script rebuilds the strides exactly as s3_phase_analysis.py does, prints the headline
numbers so they can be checked against results/emg/*.txt, and draws the figures in one style.

Run from the 02_code folder (after s2_emg_processing_alignment.py has written alignment.csv):
    python s7_report_figures.py
Outputs: 04_report/figures/*.png (only the figures used in the revised report).
"""
import os, csv, shutil
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Polygon, FancyArrowPatch, Circle
import s2_emg_processing_alignment as E
import s4_spring_shoe_model as M

import config_paths as PTH
BASE = E.BASE
OUT = PTH.REPORT_FIG; os.makedirs(OUT, exist_ok=True)
OLD = PTH.OLD_REPORT_FIG

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.5,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03})
CU, CC = '#1f77b4', '#d62728'          # unconstrained / constrained
CF, CB = '#ff7f0e', '#2ca02c'          # front / back spring
COND = {'U1': 'U', 'U2': 'U', 'C1': 'C', 'C2': 'C'}
COL = {'U': CU, 'C': CC}
GCX = np.linspace(0, 100, 101)
GC_LABEL = 'Gait cycle (%)   [0 % = heel strike; dotted = mean toe-off]'


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print('wrote', name)


def save_to(fig, folder, name):
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, name)); plt.close(fig); print('wrote', name)


# =====================================================================================  data
def build_strides():
    """Strides fully inside the EMG window, resampled to 0-100 % GC (same logic as
    s3_phase_analysis.py)."""
    al = {r['trial']: float(r['D_refined_s'])
          for r in csv.DictReader(open(os.path.join(PTH.RES_EMG, 'alignment.csv')))}
    strides, raw = {}, {}
    for tag, (vid, ef, *_r) in E.TRIALS.items():
        t_e, ta, gas = E.load_emg(ef)
        raw[tag] = np.genfromtxt(os.path.join(E.EMG_DIR, ef), delimiter=',', names=True)
        t_v, theta, pitch, fps = E.load_kin(tag); D = al[tag]
        hs, to = E.gait_events(t_v, theta, pitch)
        omega = np.gradient(theta, t_v)
        alpha = np.gradient(omega, t_v)                   # deg/s^2, from the 6 Hz-filtered angle
        rows = []
        for a, b in zip(hs[:-1], hs[1:]):
            if t_v[a] < D or t_v[b] > D + t_e[-1]:
                continue
            tt = np.linspace(t_v[a], t_v[b], 101)
            to_in = to[(to > a) & (to < b)]
            to_pct = 100 * (t_v[to_in[0]] - t_v[a]) / (t_v[b] - t_v[a]) if len(to_in) else np.nan
            rows.append(dict(tag=tag, t0=t_v[a], t1=t_v[b], T=t_v[b] - t_v[a], to_pct=to_pct,
                             th=np.interp(tt, t_v, theta), om=np.interp(tt, t_v, omega),
                             al=np.interp(tt, t_v, alpha),
                             ta=np.interp(tt - D, t_e, ta), gas=np.interp(tt - D, t_e, gas)))
        strides[tag] = rows
    nta = np.nanmean([np.nanmax(s['ta']) for t in ('U1', 'U2') for s in strides[t]])
    ngas = np.nanmean([np.nanmax(s['gas']) for t in ('U1', 'U2') for s in strides[t]])
    for tag in strides:
        for s in strides[tag]:
            s['ta_n'] = 100 * s['ta'] / nta; s['gas_n'] = 100 * s['gas'] / ngas
    print(f'normalisation: TA {1000 * nta:.1f} mV, GAS {1000 * ngas:.1f} mV')
    return strides, raw, al


def stride_metrics(s):
    th, om, T, to = s['th'], s['om'], s['T'], s['to_pct']
    to_i = int(round(to))
    st_min = 15 + int(np.argmin(th[15:61])); po = 50 + int(np.argmax(th[50:86]))
    sw_min = to_i + int(np.argmin(th[to_i:96]))

    def clean_mean(x, a, b):
        x = x[a:b]; v = ~np.isnan(x)
        return x[v].mean() if v.mean() >= 0.5 else np.nan
    return dict(T=T, stance=T * to / 100, swing=T * (1 - to / 100), to=to,
                df_exc=th[0] - th[st_min], po_exc=th[po] - th[15 + int(np.argmin(th[15:po + 1]))],
                po_peak=th[po], df_peak=th[st_min], sw_min=th[sw_min], sw_ret=th[po] - th[sw_min],
                om_pf=om[40:86].max(), om_sw=-om[to_i:96].min(), om_load=-om[0:20].min(),
                gas_po=clean_mean(s['gas_n'], 50, to_i + 1), ta_sw=clean_mean(s['ta_n'], to_i, 101))


def check_numbers(strides):
    met = {t: [stride_metrics(s) for s in strides[t]] for t in strides}
    print('\nheadline check (mean +/- SD, n):')
    for k in ('T', 'stance', 'swing', 'to', 'po_peak', 'df_peak', 'df_exc', 'sw_min', 'sw_ret',
              'om_load', 'om_pf', 'om_sw', 'gas_po', 'ta_sw'):
        out = []
        for tags in (('U1', 'U2'), ('C1', 'C2')):
            v = np.array([m[k] for t in tags for m in met[t]], float); v = v[~np.isnan(v)]
            out.append(f'{v.mean():7.2f} +/- {v.std():5.2f} (n={len(v)})')
        print(f'  {k:8s}', ' | '.join(out))
    return met


# =====================================================================================  device figures
def zigzag(p0, p1, n=7, amp=7.0):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0; L = np.linalg.norm(d); u = d / L; nrm = np.array([-u[1], u[0]])
    s = np.linspace(0, 1, 2 * n + 3)
    off = np.r_[0, [amp * (1 if i % 2 else -1) for i in range(2 * n + 1)], 0]
    pts = p0 + np.outer(s, d) + np.outer(off, nrm)
    return pts[:, 0], pts[:, 1]


def draw_link(ax, A, P, col, lw=1.6, ls='-', spring_len=55, alpha=1.0):
    """Wire from the foot point P to the spring, spring next to the band point A."""
    A, P = np.asarray(A), np.asarray(P); u = (P - A) / np.linalg.norm(P - A)
    S = A + spring_len * u
    x, y = zigzag(A, S); ax.plot(x, y, c=col, lw=1.1, ls=ls, alpha=alpha)
    ax.plot([S[0], P[0]], [S[1], P[1]], c=col, lw=lw, ls=ls, alpha=alpha)


def fig_device_schematic():
    mm = 1000
    Af, Ab = M.A_f * mm, M.A_b * mm; Pf, Pb = M.P_f0 * mm, M.P_b0 * mm
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    # shank
    ax.add_patch(Polygon([[-38, 0], [-52, 300], [-48, 430], [42, 430], [36, 300], [26, 0]],
                         closed=True, fc='0.92', ec='0.6', lw=0.8, zorder=0))
    ax.text(-3, 395, 'shank', ha='center', fontsize=8, color='0.35')
    # band
    ax.plot([Ab[0] - 6, Af[0] + 6], [Af[1], Af[1]], c='k', lw=5, solid_capstyle='butt', zorder=3)
    ax.text(Af[0] + 14, Af[1] + 10, 'shank band\n(below knee)', fontsize=8, va='bottom')
    # foot / projections, neutral and 15 deg PF (dashed)
    R = M.rot(np.radians(-15))
    Pf15, Pb15 = R @ Pf, R @ Pb
    ax.plot([Pb[0], Pf[0]], [Pb[1], Pf[1]], c='k', lw=3.2, solid_capstyle='round', zorder=3)
    ax.plot([Pb15[0], Pf15[0]], [Pb15[1], Pf15[1]], c='0.45', lw=2, ls='--', zorder=2)
    draw_link(ax, Af, Pf, CF); draw_link(ax, Ab, Pb, CB)
    draw_link(ax, Af, Pf15, CF, lw=1.2, ls='--', alpha=0.6)
    # points
    for P, lab, dx, dy, ha in ((Af, '$A_f$', 6, -24, 'left'), (Ab, '$A_b$', -6, -24, 'right'),
                               (Pf, '$P_f$', 4, 8, 'left'), (Pb, '$P_b$', -4, 8, 'right')):
        ax.plot(*P, 'o', ms=4.5, mfc='white', mec='k', zorder=5)
        ax.text(P[0] + dx, P[1] + dy, lab, ha=ha, fontsize=10)
    ax.plot(0, 0, 'o', ms=8, c='k', zorder=6)
    ax.annotate('ankle axis $O$\n(lateral malleolus)', xy=(-4, 4), xytext=(-48, 92), fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='-', lw=0.6, color='0.3'))
    # link labels
    ax.text(140, 170, 'front link S1\nwire + spring\n$L_1=340$ mm', color=CF, fontsize=8, ha='left')
    ax.text(-122, 200, 'back link S2\nwire + spring\n$L_2=330$ mm', color=CB, fontsize=8, ha='right')
    ax.annotate('springs\n$k=754.6$ N/m\n(drawn enlarged)', xy=(Af[0] + 12, Af[1] - 30), xytext=(150, 252), fontsize=8,
                arrowprops=dict(arrowstyle='-', lw=0.6, color='0.3'))
    # dimensions
    def dim(p0, p1, txt, off, rot=0, fs=8):
        ax.annotate('', xy=p1, xytext=p0, arrowprops=dict(arrowstyle='<->', lw=0.7, color='0.25'))
        ax.text((p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1], txt, ha='center', va='center',
                fontsize=fs, rotation=rot, color='0.15')
    dim((0, -88), (Pf[0], -88), '$L_3=170$ mm', (0, -13)); dim((Pb[0], -88), (0, -88), '$L_4=130$ mm', (0, -13))
    for x in (Pb[0], 0, Pf[0]):
        ax.plot([x, x], [-8, -96], c='0.6', lw=0.5)
    dim((Ab[0], Af[1] + 28), (Af[0], Af[1] + 28), '$W=122$ mm', (0, 11))
    dim((-240, 0), (-240, Af[1]), '$D=322.5$ mm (reconstructed)', (-12, 0), rot=90)
    ax.plot([-250, Ab[0]], [Af[1], Af[1]], c='0.6', lw=0.5); ax.plot([-250, Pb[0]], [0, 0], c='0.6', lw=0.5)
    # angle convention arrows (counter-clockwise = toe up = dorsiflexion)
    ax.add_patch(FancyArrowPatch((92, -5), (88, 30), connectionstyle='arc3,rad=0.3', arrowstyle='-|>',
                                 mutation_scale=9, lw=1, color='0.15'))
    ax.text(97, 26, r'DF, $\theta>0$', fontsize=8)
    ax.add_patch(FancyArrowPatch((92, 5), (88, -32), connectionstyle='arc3,rad=-0.3', arrowstyle='-|>',
                                 mutation_scale=9, lw=1, color='0.15'))
    ax.text(97, -40, r'PF, $\theta<0$', fontsize=8, va='center')
    ax.annotate('', xy=(-215, 475), xytext=(-215, 425), arrowprops=dict(arrowstyle='-|>', lw=0.8))
    ax.annotate('', xy=(-165, 425), xytext=(-215, 425), arrowprops=dict(arrowstyle='-|>', lw=0.8))
    ax.text(-210, 477, '$y$ (along shank)', fontsize=7.5, va='bottom'); ax.text(-160, 425, '$x$ (forward)', fontsize=7.5, va='center')
    ax.set_xlim(-275, 285); ax.set_ylim(-125, 490); ax.set_aspect('equal')
    fig.savefig(os.path.join(PTH.RES_FIG, 'fig_device_schematic.png')); plt.close(fig)   # superseded in the report by fig_device_illustration.png


def fig_device_model():
    th = np.radians(np.linspace(-25, 25, 1001))
    vals = [M.device_torque(t) for t in th]
    tau = np.array([v[0] for v in vals]); Ff = np.array([v[1][0] for v in vals]); Fb = np.array([v[2][0] for v in vals])
    K = -np.gradient(tau, th); deg = np.degrees(th)
    lo, hi = np.degrees(-M.d2_0 / abs(M.device_torque(0)[2][1])), np.degrees(M.d1_0 / M.device_torque(0)[1][1])
    ops = {'push-off': 90 - 94.82, 'late-stance DF': 90 - 79.78}   # measured constrained mean peaks, theta = 90 - Theta
    print('\nmodel at measured angles:')
    for name, d in ops.items():
        t, f, b = M.device_torque(np.radians(d))
        print(f'  {name:15s} theta {d:+.2f} deg: torque {t:+.2f} N m, front {f[0]:.1f} N, back {b[0]:.1f} N')
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6))
    for a in axs:
        a.axvspan(lo, hi, color='0.85', lw=0); a.set_xlabel(r'Ankle angle $\theta$ (deg), $+$ = DF')
        a.set_xlim(-25, 25)
    axs[0].plot(deg, tau, c='k'); axs[0].axhline(0, c='0.4', lw=0.6)
    axs[0].set_ylabel(r'Device torque $\tau_\mathrm{dev}$ (N m), $+$ = DF')
    axs[1].plot(deg, Ff, c=CF, label='front spring S1'); axs[1].plot(deg, Fb, c=CB, label='back spring S2')
    axs[1].set_ylabel('Wire tension $F$ (N)'); axs[1].legend(loc='upper center', frameon=False)
    axs[2].plot(deg, K, c='#9467bd'); axs[2].set_ylabel(r'Stiffness $K=-d\tau/d\theta$ (N m/rad)')
    for y, txt, x in ((31.8, 'both taut 31.8', 3), (19.6, 'PF side 19.6', -24), (12.2, 'DF side 12.2', 6)):
        axs[2].axhline(y, c='0.5', ls=':', lw=0.8); axs[2].text(x, y + 0.6, txt, fontsize=7, color='0.3')
    axs[2].set_ylim(0, 35)
    for name, d in ops.items():
        t, f, b = M.device_torque(np.radians(d)); F = f[0] if d < 0 else b[0]
        c = CF if d < 0 else CB
        axs[0].plot(d, t, 'o', ms=5, mfc='white', mec=c, mew=1.4, zorder=5)
        axs[1].plot(d, F, 'o', ms=5, mfc='white', mec=c, mew=1.4, zorder=5)
    axs[0].annotate('push-off\n(measured)', xy=(ops['push-off'], 1.8), xytext=(-24, -3.6), fontsize=7,
                    arrowprops=dict(arrowstyle='-', lw=0.5))
    axs[0].annotate('late-stance DF\n(measured)', xy=(ops['late-stance DF'], -2.3), xytext=(3, 3.0), fontsize=7,
                    arrowprops=dict(arrowstyle='-', lw=0.5))
    for a, l in zip(axs, 'abc'):
        a.set_title(f'({l})', loc='left', fontsize=9)
    fig.tight_layout(w_pad=1.2)
    save(fig, 'fig_device_model.png')


def fig_electrodes():
    fig, ax = plt.subplots(figsize=(3.6, 4.6))
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    leg = [[-42, -6], [-62, 150], [-74, 270], [-66, 360], [-55, 440], [44, 440], [40, 360], [33, 200], [26, -6]]
    ax.add_patch(Polygon(leg, closed=True, fc='#f3e3d3', ec='0.45', lw=0.8))
    shoe = [[-150, -14], [-150, -45], [195, -45], [195, -24], [120, -12], [30, -8], [-50, -8], [-90, -12]]
    ax.add_patch(Polygon(shoe, closed=True, fc='0.3', ec='k', lw=0.6))
    ax.plot([-130, 170], [0, 0], c='k', lw=3)
    ax.text(150, -62, 'shoe projections\n(ankle-axis level)', fontsize=7, ha='center', va='top', color='0.3')
    Af, Ab = M.A_f * 1000, M.A_b * 1000
    ax.plot([Ab[0] - 15, Af[0] + 15], [Af[1], Af[1]], c='k', lw=5, solid_capstyle='butt')
    draw_link(ax, Af, (170, 0), CF, spring_len=50); draw_link(ax, Ab, (-130, 0), CB, spring_len=50)
    # electrodes (indicative positions)
    for y in (262, 242):
        ax.add_patch(Circle((32, y), 7, fc='#4f81bd', ec='k', lw=0.6, zorder=5))
    for y in (240, 220):                       # GAS pair below the back spring, beside the back wire
        ax.add_patch(Circle((-66, y), 7, fc='#c0504d', ec='k', lw=0.6, zorder=5))
    ax.add_patch(Circle((0, 16), 7, fc='0.85', ec='k', lw=0.6, zorder=5))
    ax.annotate('TA pair\n(2 cm apart)', xy=(38, 252), xytext=(105, 185), fontsize=7.5,
                arrowprops=dict(arrowstyle='-', lw=0.6))
    ax.annotate('GAS pair\n(2 cm apart)', xy=(-72, 230), xytext=(-215, 175), fontsize=7.5,
                arrowprops=dict(arrowstyle='-', lw=0.6))
    ax.annotate('back spring', xy=(-68, 300), xytext=(-215, 285), fontsize=7.5, color=CB,
                arrowprops=dict(arrowstyle='-', lw=0.6, color=CB))
    ax.annotate('ground\n(ankle bone)', xy=(6, 18), xytext=(45, 60), fontsize=7.5,
                arrowprops=dict(arrowstyle='-', lw=0.6))
    ax.text(110, 330, 'front wire', color=CF, fontsize=7.5); ax.text(-215, 120, 'back wire\n(down the calf)', color=CB, fontsize=7.5)
    ax.text(-5, 452, 'posterior  $\\leftarrow$   $\\rightarrow$  anterior', ha='center', fontsize=7.5, color='0.3')
    ax.set_xlim(-225, 215); ax.set_ylim(-95, 470); ax.set_aspect('equal')
    fig.savefig(os.path.join(PTH.RES_FIG, 'fig_electrode_wire_routing.png')); plt.close(fig)   # superseded by fig_electrode_illustration.png


# =====================================================================================  data figures
def phase_guides(ax, to_u, to_c, ymax=None):
    for x in (15, 50):
        ax.axvline(x, c='0.55', lw=0.6, ls='--')
    ax.axvline(to_u, c=CU, ls=':', lw=1); ax.axvline(to_c, c=CC, ls=':', lw=1)


def fig_gait_cycle(strides):
    fig, axes = plt.subplots(3, 1, figsize=(6.3, 7.0), sharex=True)
    tos = {}
    for cond, tags, col, lab in (('U', ('U1', 'U2'), CU, 'Unconstrained'), ('C', ('C1', 'C2'), CC, 'Constrained')):
        S = [s for t in tags for s in strides[t]]
        tos[cond] = np.nanmean([s['to_pct'] for s in S])
        for ax, key in zip(axes, ('th', 'ta_n', 'gas_n')):
            Mx = np.array([s[key] for s in S]); mu, sd = np.nanmean(Mx, 0), np.nanstd(Mx, 0)
            few = (~np.isnan(Mx)).mean(0) < 0.4
            mu[few] = np.nan; sd[few] = np.nan
            ax.plot(GCX, mu, c=col, label=f'{lab} ($n={len(S)}$ strides)')
            ax.fill_between(GCX, mu - sd, mu + sd, color=col, alpha=0.18, lw=0)
    for ax in axes:
        phase_guides(ax, tos['U'], tos['C'])
    axes[0].axhline(90, c='k', lw=0.6)
    axes[0].set_ylabel('Shank–sole angle $\\Theta$ (deg)\n$<90^\\circ$ DF, $>90^\\circ$ PF')
    axes[1].set_ylabel('TA envelope\n(% of unconstr. ref. peak)')
    axes[2].set_ylabel('GAS envelope\n(% of unconstr. ref. peak)')
    axes[0].legend(loc='lower left', frameon=False, ncol=2)
    for x, txt in ((7.5, 'Loading'), (32.5, 'Mid/terminal stance'), (59.5, 'Push-off'), (85, 'Swing')):
        axes[0].text(x, 109.5, txt, ha='center', fontsize=7.5, color='0.3')
    axes[0].set_ylim(76, 112)
    for a, l in zip(axes, 'abc'):
        a.set_title(f'({l})', loc='left', fontsize=9)
    axes[2].set_xlabel(GC_LABEL); axes[2].set_xlim(0, 100)
    fig.tight_layout(h_pad=0.6)
    save(fig, 'fig_gait_cycle_theta_TA_GAS.png')


def fig_velocity(strides):
    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    tos = {}
    for cond, tags, col, lab in (('U', ('U1', 'U2'), CU, 'Unconstrained'), ('C', ('C1', 'C2'), CC, 'Constrained')):
        Mx = np.array([s['om'] for t in tags for s in strides[t]])
        mu, sd = Mx.mean(0), Mx.std(0)
        ax.plot(GCX, mu, c=col, label=f'{lab} ($n={len(Mx)}$)')
        ax.fill_between(GCX, mu - sd, mu + sd, color=col, alpha=0.18, lw=0)
        tos[cond] = np.nanmean([s['to_pct'] for t in tags for s in strides[t]])
    phase_guides(ax, tos['U'], tos['C'])
    ax.axhline(0, c='k', lw=0.6)
    ax.set_ylabel('$d\\Theta/dt$ (deg/s)\n$+$ towards PF, $-$ towards DF')
    ax.set_xlabel(GC_LABEL); ax.set_xlim(0, 100); ax.legend(frameon=False, loc='upper left')
    ax.annotate('push-off PF', xy=(60, 140), xytext=(30, 120), fontsize=7.5, arrowprops=dict(arrowstyle='-', lw=0.5))
    ax.annotate('swing DF', xy=(74, -95), xytext=(82, -125), fontsize=7.5, arrowprops=dict(arrowstyle='-', lw=0.5))
    ax.annotate('loading DF', xy=(8, -75), xytext=(16, -118), fontsize=7.5, arrowprops=dict(arrowstyle='-', lw=0.5))
    ax.set_ylim(-135, 185)
    fig.tight_layout()
    save(fig, 'fig_angular_velocity.png')


def fig_single_stride(strides, raw, al):
    env_cache = {}
    for tag in ('U2', 'C1'):
        d = raw[tag]
        for ch, name in (('Tibialis_Raw_V', 'TA'), ('Gastro_Raw_V', 'GAS')):
            m, _ = E.saturation_mask(d[ch]); m = m | E.lowfreq_mask(d[ch])
            env_cache[(tag, name)] = (E.emg_envelope(d[ch]), m)

    def pick(tag):
        best = None
        for s in strides[tag]:
            c = (~np.isnan(s['ta']) & ~np.isnan(s['gas'])).mean()
            if best is None or c > best[0] + 1e-9:
                best = (c, s)
        return best[1]
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 6.4), sharex=True)
    for col, tag in enumerate(('U2', 'C1')):
        s = pick(tag); D = al[tag]; d = raw[tag]; c = COL[COND[tag]]
        te = d['Time_s']; m = (te >= s['t0'] - D) & (te <= s['t1'] - D)
        gc = 100 * (te[m] - (s['t0'] - D)) / s['T']
        a = axes[0, col]
        a.plot(GCX, s['th'], c=c); a.axhline(90, c='k', lw=0.6); a.axvline(s['to_pct'], c='k', ls=':', lw=0.9)
        a.set_title(('(a) U2, unconstrained' if col == 0 else '(b) C1, constrained') +
                    f'  (stride at video $t$ = {s["t0"]:.1f} s)', loc='left', fontsize=9)
        a.set_ylabel('$\\Theta$ (deg)')
        for row, (ch, name) in enumerate((('Tibialis_Raw_V', 'TA'), ('Gastro_Raw_V', 'GAS')), start=1):
            env, mk = env_cache[(tag, name)]
            ax = axes[row, col]
            ax.plot(gc, d[ch][m], c='0.55', lw=0.35)
            if mk[m].any():
                ax.fill_between(gc, 0, 3.3, where=mk[m], color=CC, alpha=0.13, lw=0)
            ax.set_ylim(0, 3.3); ax.axvline(s['to_pct'], c='k', ls=':', lw=0.9)
            ax.set_ylabel(f'{name} raw (V)')
            ax2 = ax.twinx(); ax2.grid(False); ax2.spines['right'].set_visible(True)
            ax2.plot(gc, np.where(mk[m], np.nan, env[m] * 1000), c=c, lw=1.6)
            ax2.set_ylim(0, 450); ax2.set_ylabel(f'{name} envelope (mV)')
        axes[2, col].set_xlabel('Gait cycle (%)'); axes[2, col].set_xlim(0, 100)
    fig.tight_layout(w_pad=2.2, h_pad=0.5)
    save(fig, 'fig_representative_stride.png')


def fig_summary(met):
    keys = [('stance', 'Stance duration (s)', 1), ('po_exc', 'Push-off PF excursion (deg)', 1),
            ('df_exc', 'Stance DF excursion (deg)', 1), ('om_pf', 'Peak push-off PF velocity (deg/s)', 1),
            ('gas_po', 'GAS, push-off phase\n(% of unconstr. ref. peak)', 1),
            ('ta_sw', 'TA, swing phase\n(% of unconstr. ref. peak)', 1)]
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.6))
    rng = np.random.default_rng(1)
    pos = {'U1': 0, 'U2': 1, 'C1': 2.4, 'C2': 3.4}
    for ax, (k, lab, _), l in zip(axs.ravel(), keys, 'abcdef'):
        for cond, tags in (('U', ('U1', 'U2')), ('C', ('C1', 'C2'))):
            allv = []
            for t in tags:
                v = np.array([m[k] for m in met[t]], float); v = v[~np.isnan(v)]; allv += list(v)
                ax.plot(pos[t] + rng.uniform(-0.18, 0.18, len(v)), v, 'o', ms=3, mfc=COL[cond], mec='none', alpha=0.55)
            allv = np.array(allv); xc = np.mean([pos[t] for t in tags])
            ax.errorbar(xc, allv.mean(), yerr=allv.std(), fmt='_', ms=22, mew=2, c='k', capsize=4, lw=1.2, zorder=6)
            ax.text(xc, 1.0, f'$n={len(allv)}$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=7)
        ax.set_xticks([0, 1, 2.4, 3.4]); ax.set_xticklabels(['U1', 'U2', 'C1', 'C2'])
        ax.set_xlim(-0.6, 4.0); ax.set_ylabel(lab, fontsize=8); ax.grid(axis='x', visible=False)
        lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, hi_ + 0.14 * (hi_ - lo_))
        ax.set_title(f'({l})', loc='left', fontsize=9, pad=12)
    fig.tight_layout(h_pad=1.4, w_pad=1.0)
    save(fig, 'fig_key_outcomes.png')


def fig_phase_emg(strides):
    """Phase-averaged TA, GAS and CCI with and without the constraint (same rule as
    s3_phase_analysis.py: a stride counts if >= 50 % of the phase is artefact-free)."""
    phases = ['Load.', 'Mid/term.\nstance', 'Push-\noff', 'Swing']

    def win(to, k):
        to_i = int(round(to))
        return [(0, 16), (15, 51), (50, to_i + 1), (to_i, 101)][k]

    def val(s, q, a, b):
        ta, gas = s['ta_n'][a:b], s['gas_n'][a:b]
        if q == 'CCI':
            both = ~np.isnan(ta) & ~np.isnan(gas)
            return 200 * np.minimum(ta, gas)[both].sum() / (ta + gas)[both].sum() if both.mean() >= 0.5 else np.nan
        x = ta if q == 'TA' else gas; v = ~np.isnan(x)
        return x[v].mean() if v.mean() >= 0.5 else np.nan
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.9))
    for ax, q, lab in zip(axs, ('TA', 'GAS', 'CCI'),
                          ('TA (% of unconstr. ref. peak)', 'GAS (% of unconstr. ref. peak)', 'Co-contraction index (%)')):
        for j, (cond, tags) in enumerate((('U', ('U1', 'U2')), ('C', ('C1', 'C2')))):
            for k in range(4):
                v = np.array([val(s, q, *win(s['to_pct'], k)) for t in tags for s in strides[t]], float)
                v = v[~np.isnan(v)]; x = k + (j - 0.5) * 0.38
                if len(v) == 0:
                    ax.text(x, 2, 'n=0', ha='center', va='bottom', fontsize=6, rotation=90); continue
                weak = cond == 'C' and len(v) < 5
                ax.bar(x, v.mean(), 0.38, yerr=v.std() if len(v) > 1 else 0, capsize=2, error_kw=dict(lw=0.8),
                       color='white' if weak else COL[cond], edgecolor=COL[cond], hatch='///' if weak else None, lw=0.8)
                ax.text(x, 1.5, f'n={len(v)}', ha='center', va='bottom', fontsize=6, rotation=90,
                        color=COL[cond] if weak else 'white')
        ax.set_xticks(range(4)); ax.set_xticklabels(phases, fontsize=7); ax.set_ylabel(lab, fontsize=8)
        ax.grid(axis='x', visible=False); ax.set_ylim(0, 100 if q != 'CCI' else 100)
    from matplotlib.patches import Patch
    axs[0].legend(handles=[Patch(fc=CU, label='Unconstrained'), Patch(fc=CC, label='Constrained'),
                           Patch(fc='white', ec=CC, hatch='///', label='Constrained, $n<5$')],
                  fontsize=6.5, frameon=False, loc='upper right')
    for a, l in zip(axs, 'abc'):
        a.set_title(f'({l})', loc='left', fontsize=9)
    fig.tight_layout(w_pad=1.0)
    save(fig, 'fig_phase_emg.png')


# =====================================================================================  photographs
def title_photo():
    img = mpimg.imread(os.path.join(OLD, 'fig03_annotated_video_frames_U2_vs_C1.png'))
    crop = img[300:760, 860:1320]                    # constrained frame: band, springs, wires, shoe
    plt.imsave(os.path.join(OUT, 'title_spring_shoe_photo.png'), crop)
    print('wrote title_spring_shoe_photo.png')


def copy_existing():
    for src, dst in (('fig03_annotated_video_frames_U2_vs_C1.png', 'fig_tracking_frames_U2_C1.png'),
                     ('fig07_emg_artefact_clean_vs_clipped.png', 'fig_emg_artefact.png'),
                     ('figA_U1_emg_recording_system_plot.png', 'figA_U1_recording_system.png'),
                     ('figA_U2_emg_recording_system_plot.png', 'figA_U2_recording_system.png'),
                     ('figA_C1_emg_recording_system_plot.png', 'figA_C1_recording_system.png'),
                     ('figA_C2_emg_recording_system_plot.png', 'figA_C2_recording_system.png')):
        shutil.copyfile(os.path.join(OLD, src), os.path.join(OUT, dst)); print('copied', dst)


if __name__ == '__main__':
    fig_device_schematic(); fig_device_model(); fig_electrodes()
    strides, raw, al = build_strides()
    met = check_numbers(strides)
    fig_gait_cycle(strides); fig_velocity(strides); fig_single_stride(strides, raw, al); fig_summary(met)
    fig_phase_emg(strides)
    title_photo(); copy_existing()
