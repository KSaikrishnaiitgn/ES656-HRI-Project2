"""
HRI Class Project 2 - Part B: reachable workspace and manipulability of the knee-ankle-foot chain,
with and without the spring-shoe constraint (brief step 1).

Model: planar two-link chain in the sagittal plane with its base at the knee (thigh frame, thigh
vertical):
    link 1 = shank, knee -> ankle,      l_s = 0.246 H     (Winter 2009 segment proportion)
    link 2 = foot,  ankle -> forefoot,  l_f = 0.15 m      (ankle to metatarsal heads, estimate)
    joints q = (q_k, theta): knee flexion (+ = flexion) and ankle angle from neutral (+ = DF).
    End point = forefoot (metatarsal heads).
Check on l_s: thigh 0.245H + shank 0.246H + ankle height 0.039H = 0.885 m, against the
measured leg length of 0.890 m.

Conventional (static) workspace = set of forefoot positions the subject can reach and hold:
  * without the device: knee 0-135 deg flexion, ankle 20 deg DF to 50 deg PF (ASSUMED typical
    anatomical ranges; the subject's range of motion was not measured);
  * with the device: the knee range is unchanged (the device spans only the ankle). The ankle
    range is limited by a MODEL CRITERION: an ankle posture is counted as reachable if holding it
    against the device torque needs no more than a share s of the subject's maximum voluntary
    torque (MVC) in that direction:
        DF side:  |tau_dev(theta)| <= s * MVC_DF,  MVC_DF = 41.9 N m
        PF side:  |tau_dev(theta)| <= s * MVC_PF,  MVC_PF = 131.0 N m
    MVC values = healthy men aged 20-29, right ankle (Moraux et al. 2013, BMC Musculoskelet
    Disord 14:104). Nominal s = 5 % (a light holding effort); s is swept as a sensitivity.
    A second, purely mechanical criterion (the spring reaching an assumed maximum extension)
    is also swept.
Walking workspace: knee 0-60 deg (typical walking range, assumed; the knee was not tracked) x the
  ankle range MEASURED in each condition (mean per-stride min and max of theta).

Manipulability (both depend on theta only, not on the knee angle):
  * kinematic (Yoshikawa)  w   = sqrt(det(J J^T)) = |det J| = l_s l_f cos(theta)
  * dynamic                w_d = |det(J M^-1)|    = |det J| / det M(theta)
    M = joint-space inertia of shank + foot (+ shoe) about the knee (Winter segment
    parameters; shoe 0.99 kg, J_shoe about the ankle 0.0094 kg m^2). Band mass neglected.

Run from 02_code:  python s5_workspace_manipulability.py
Outputs: 04_report/figures/fig_workspace.png, fig_manipulability.png,
         03_results/analysis/s5_workspace_manipulability.txt
"""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import config_paths as P
import s4_spring_shoe_model as M
import s7_report_figures as F          # shared style, colours and stride builder

os.makedirs(P.RES_ANALYSIS, exist_ok=True)
H, MB = M.H_body, M.M_body
L_S, L_F = 0.246 * H, 0.15
KNEE_ANAT, ANK_ANAT = (0.0, 135.0), (-50.0, 20.0)      # deg, assumed typical ranges
KNEE_WALK = (0.0, 60.0)                                 # deg, assumed typical walking range
MVC_DF, MVC_PF = 41.9, 131.0                            # N m, Moraux et al. 2013 (men 20-29, right)
SHARE_NOM = 0.05                                        # nominal holding effort, share of MVC


def tau_dev_deg(d):
    return M.device_torque(np.radians(d))[0]


def stretch_mm(d):
    _, f, b = M.device_torque(np.radians(d))
    return f[3] * 1000, b[3] * 1000


def rom_by_effort(share):
    """Ankle ROM (theta, deg) reachable when holding the device torque costs <= share of MVC."""
    df = next((d for d in np.arange(0, ANK_ANAT[1] + 1e-9, 0.05) if -tau_dev_deg(d) > share * MVC_DF), None)
    pf = next((d for d in np.arange(0, ANK_ANAT[0] - 1e-9, -0.05) if tau_dev_deg(d) > share * MVC_PF), None)
    return (ANK_ANAT[0] if pf is None else pf, ANK_ANAT[1] if df is None else df)


def rom_by_extension(ext_mm):
    """Ankle ROM if each spring could stretch at most ext_mm (link becomes a hard stop)."""
    df = next((d for d in np.arange(0, ANK_ANAT[1] + 1e-9, 0.05) if stretch_mm(d)[1] > ext_mm), None)
    pf = next((d for d in np.arange(0, ANK_ANAT[0] - 1e-9, -0.05) if stretch_mm(d)[0] > ext_mm), None)
    return (ANK_ANAT[0] if pf is None else pf, ANK_ANAT[1] if df is None else df)

# ---- segment inertial parameters (Winter 2009), all in the segment's own frame -------------
m_sh = 0.0465 * MB; c_sh = 0.433 * L_S; I_sh = m_sh * (0.302 * L_S) ** 2          # shank about COM
m_ft = M.m_foot; c_ft = np.array([M.x_foot, M.y_foot]); I_ft = M.J_foot - m_ft * c_ft @ c_ft
m_so = M.m_shoe; c_so = np.array([M.x_shoe, M.y_shoe]); I_so = M.J_shoe - m_so * c_so @ c_so


def rot2(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def fk(qk, th):
    """Ankle and forefoot positions (thigh frame, knee at origin, x forward, y up); angles in rad."""
    ank = L_S * np.array([-np.sin(qk), -np.cos(qk)])
    phi = th - qk                                          # foot direction (0 = horizontal forward)
    return ank, ank + L_F * np.array([np.cos(phi), np.sin(phi)])


def jacobian(qk, th):
    phi = th - qk
    dk = L_S * np.array([-np.cos(qk), np.sin(qk)]) + L_F * np.array([np.sin(phi), -np.cos(phi)])
    dt = L_F * np.array([-np.sin(phi), np.cos(phi)])
    return np.column_stack([dk, dt])


def mass_matrix(qk, th, shoe):
    """Joint-space inertia about the knee from COM Jacobians (finite differences)."""
    def coms(q):
        k, t = q
        ank, _ = fk(k, t); R = rot2(t - k)
        out = [(m_sh, I_sh, L_S * np.array([-np.sin(k), -np.cos(k)]) * (c_sh / L_S), np.array([-1.0, 0.0])),
               (m_ft, I_ft, ank + R @ c_ft, np.array([-1.0, 1.0]))]
        if shoe:
            out.append((m_so, I_so, ank + R @ c_so, np.array([-1.0, 1.0])))
        return out
    q0 = np.array([qk, th]); eps = 1e-6
    base = coms(q0)
    Mq = np.zeros((2, 2))
    for i, (m, I, p, jw) in enumerate(base):
        Jv = np.zeros((2, 2))
        for j in range(2):
            dq = np.zeros(2); dq[j] = eps
            Jv[:, j] = (coms(q0 + dq)[i][2] - coms(q0 - dq)[i][2]) / (2 * eps)
        Mq += m * Jv.T @ Jv + I * np.outer(jw, jw)
    return Mq


def manip(th, shoe, qk=0.3):
    J = jacobian(qk, th); Mq = mass_matrix(qk, th, shoe)
    return abs(np.linalg.det(J)), abs(np.linalg.det(J @ np.linalg.inv(Mq)))


def measured_ranges(strides):
    """Ankle range used in walking: mean per-stride min and max of Theta, converted to theta."""
    out = {}
    for c, tags in (('U', ('U1', 'U2')), ('C', ('C1', 'C2'))):
        th = [90 - s['th'] for t in tags for s in strides[t]]
        out[c] = (np.mean([x.min() for x in th]), np.mean([x.max() for x in th]))
    return out


def region(knee, ank, n=60):
    """Boundary of the forefoot workspace for rectangular joint limits (deg)."""
    k = np.radians(np.linspace(*knee, n)); a = np.radians(np.linspace(*ank, n))
    edge = ([(ki, a[0]) for ki in k] + [(k[-1], ai) for ai in a] +
            [(ki, a[-1]) for ki in k[::-1]] + [(k[0], ai) for ai in a[::-1]])
    return np.array([fk(*e)[1] for e in edge])


def draw_leg(ax, qk_deg, th_deg, **kw):
    ank, toe = fk(np.radians(qk_deg), np.radians(th_deg))
    ax.plot([0, ank[0], toe[0]], [0, ank[1], toe[1]], '-o', ms=3, **kw)


def area_of(r):
    x, y = r[:, 0], r[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def fig_workspace(rng, lines):
    rom5 = rom_by_effort(SHARE_NOM); rom10 = rom_by_effort(0.10)
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 6.6))

    # (a) conventional workspace, without and with the device (effort criterion)
    ax = axs[0, 0]
    an = region(KNEE_ANAT, ANK_ANAT); rc = region(KNEE_ANAT, rom5)
    ax.fill(*an.T, color=F.CU, alpha=0.18, lw=0)
    ax.plot(*np.r_[an, an[:1]].T, c=F.CU, lw=1.2, label='without device')
    ax.fill(*rc.T, color=F.CC, alpha=0.30, lw=0)
    ax.plot(*np.r_[rc, rc[:1]].T, c=F.CC, lw=1.2, label=f'with device ({100 * SHARE_NOM:.0f} % MVC criterion)')
    draw_leg(ax, 0, 0, c='k', lw=1.4); draw_leg(ax, 0, ANK_ANAT[0], c='0.45', lw=0.8, ls=':')
    draw_leg(ax, 0, ANK_ANAT[1], c='0.45', lw=0.8, ls=':')
    ax.plot([0, 0], [0, 0.12], c='k', lw=2.5); ax.text(0.012, 0.09, 'thigh', fontsize=7)
    ax.plot(0, 0, 'ko', ms=5); ax.text(0.015, -0.03, 'knee', fontsize=7.5)
    ax.set_aspect('equal'); ax.set_xlabel('$x$, forward (m)'); ax.set_ylabel('$y$, up (m)')
    ax.set_xlim(-0.62, 0.30); ax.set_ylim(-0.66, 0.16)
    ax.legend(loc='lower left', fontsize=6.8, frameon=False)
    ax.set_title('(a) conventional forefoot workspace', loc='left', fontsize=9)

    # (b) ankle ROM vs allowed holding effort
    ax = axs[0, 1]
    sh = np.linspace(0.005, 0.15, 59); lim = np.array([rom_by_effort(s) for s in sh])
    ax.fill_between(100 * sh, lim[:, 0], lim[:, 1], color=F.CC, alpha=0.25, lw=0, label='reachable with device')
    ax.plot(100 * sh, lim[:, 0], c=F.CC); ax.plot(100 * sh, lim[:, 1], c=F.CC)
    for y in ANK_ANAT:
        ax.axhline(y, c=F.CU, ls='--', lw=1)
    ax.text(14.8, ANK_ANAT[1] + 1.5, 'without device', color=F.CU, fontsize=7, ha='right')
    ax.axvline(100 * SHARE_NOM, c='k', ls=':', lw=0.9); ax.text(100 * SHARE_NOM + 0.3, -46, 'nominal', fontsize=7)
    ax.axhspan(*rng['C'], color='0.5', alpha=0.15, lw=0); ax.text(0.8, rng['C'][1] + 0.8, 'used in walking (C)', fontsize=6.5, color='0.3')
    ax.set_xlabel('Allowed holding effort $s$ (% of MVC)'); ax.set_ylabel(r'Ankle range limit $\theta$ (deg), $+$ = DF')
    ax.set_xlim(0, 15); ax.set_ylim(-55, 26)
    ax.set_title('(b) ankle range vs effort criterion', loc='left', fontsize=9)

    # (c) ankle ROM vs assumed maximum spring extension
    ax = axs[1, 0]
    ex = np.linspace(5, 120, 116); lim = np.array([rom_by_extension(e) for e in ex])
    ax.fill_between(ex, lim[:, 0], lim[:, 1], color='#9467bd', alpha=0.22, lw=0)
    ax.plot(ex, lim[:, 0], c='#9467bd'); ax.plot(ex, lim[:, 1], c='#9467bd')
    for y in ANK_ANAT:
        ax.axhline(y, c=F.CU, ls='--', lw=1)
    walk_ext = max(stretch_mm(rng['C'][0])[0], stretch_mm(rng['C'][1])[1])
    ax.axvspan(0, walk_ext, color='0.5', alpha=0.15, lw=0)
    ax.text(walk_ext / 2, -48, 'reached in\nwalking', fontsize=6.5, ha='center', color='0.3')
    ax.set_xlabel('Assumed maximum spring extension (mm)'); ax.set_ylabel(r'Ankle range limit $\theta$ (deg)')
    ax.set_xlim(0, 120); ax.set_ylim(-55, 26)
    ax.set_title('(c) ankle range vs spring extension limit', loc='left', fontsize=9)

    # (d) walking workspace with the measured ankle ranges
    ax = axs[1, 1]
    ax.fill(*an.T, color='0.9', lw=0, label='without device (conventional)')
    for c, col, lab in (('U', F.CU, 'walking, unconstrained'), ('C', F.CC, 'walking, constrained')):
        r = region(KNEE_WALK, rng[c])
        ax.fill(*r.T, color=col, alpha=0.25, lw=0); ax.plot(*np.r_[r, r[:1]].T, c=col, lw=1.2, label=lab)
    draw_leg(ax, 0, 0, c='k', lw=1.2); draw_leg(ax, 60, 0, c='0.5', lw=0.9, ls='--')
    ax.plot(0, 0, 'ko', ms=4)
    ax.set_aspect('equal'); ax.set_xlabel('$x$, forward (m)'); ax.set_ylabel('$y$, up (m)')
    ax.set_xlim(-0.62, 0.30); ax.set_ylim(-0.66, 0.16)
    ax.legend(loc='lower left', fontsize=6.8, frameon=False)
    ax.set_title('(d) workspace used in walking (measured ankle range)', loc='left', fontsize=9)
    fig.tight_layout(h_pad=1.4, w_pad=1.2)
    F.save(fig, 'fig_workspace.png')

    # numbers
    lines.append(f'MVC: DF {MVC_DF} N m, PF {MVC_PF} N m (Moraux et al. 2013)')
    lines.append(f'conventional workspace without device: knee {KNEE_ANAT}, ankle {ANK_ANAT}, area {area_of(an) * 1e4:.0f} cm^2')
    for s, rom in ((SHARE_NOM, rom5), (0.10, rom10)):
        r = region(KNEE_ANAT, rom)
        lines.append(f'with device, {100 * s:.0f} % MVC: ankle {rom[0]:+.1f} to {rom[1]:+.1f} deg (span {rom[1] - rom[0]:.1f} of '
                     f'{ANK_ANAT[1] - ANK_ANAT[0]:.0f} deg), area {area_of(r) * 1e4:.0f} cm^2 ({100 * area_of(r) / area_of(an):.0f} % of barefoot)')
    for s in (0.02, 0.03, 0.05, 0.075, 0.10):
        lo, hi = rom_by_effort(s); lines.append(f'  effort {100 * s:4.1f} % MVC: PF limit {lo:+.1f}, DF limit {hi:+.1f}')
    for e in (25, 30, 50, 100):
        lo, hi = rom_by_extension(e); lines.append(f'  spring extension limit {e} mm: PF limit {lo:+.1f}, DF limit {hi:+.1f}')
    lines.append(f'  max spring extension reached in walking (model at measured C range) = {walk_ext:.1f} mm')
    for c in ('U', 'C'):
        r = region(KNEE_WALK, rng[c])
        lines.append(f'walking workspace {c}: ankle theta {rng[c][0]:+.1f} to {rng[c][1]:+.1f} deg '
                     f'(span {rng[c][1] - rng[c][0]:.1f} deg), forefoot area {area_of(r) * 1e4:.0f} cm^2')
    for d in (-50, -24, -20, -7.3, 0, 9.2, 10.2, 20):
        t, f, b = M.device_torque(np.radians(d))
        lines.append(f'  theta {d:+6.1f} deg: tau_dev {t:+6.2f} N m, front stretch {max(f[3], 0) * 1000:6.1f} mm, '
                     f'back stretch {max(b[3], 0) * 1000:6.1f} mm')
    return rom5


def ellipse(ax, centre, A, scale, **kw):
    U, s, _ = np.linalg.svd(A)
    ang = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    ax.add_patch(Ellipse(centre, 2 * s[0] * scale, 2 * s[1] * scale, angle=ang, fill=False, **kw))


def fig_manipulability(strides, rng, rom5, lines):
    th = np.radians(np.linspace(-50, 20, 141))
    wU = np.array([manip(t, False) for t in th]); wC = np.array([manip(t, True) for t in th])
    w0 = L_S * L_F; wd0 = manip(0, False)[1]
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6))
    ax = axs[0, 0]
    ax.axvspan(*rom5, color=F.CC, alpha=0.10, lw=0)
    ax.plot(np.degrees(th), wU[:, 0] / w0, c='k', label='kinematic $w$ (same with and without)')
    ax.plot(np.degrees(th), wU[:, 1] / wd0, c=F.CU, label='dynamic $w_d$, without device')
    ax.plot(np.degrees(th), wC[:, 1] / wd0, c=F.CC, label='dynamic $w_d$, with device (shoe)')
    for c, col, y in (('U', F.CU, 0.08), ('C', F.CC, 0.03)):
        ax.plot(rng[c], [y, y], c=col, lw=4, solid_capstyle='butt')
    ax.text(-49, 0.13, 'bars: ankle range used in walking', fontsize=6.5, color='0.3')
    ax.set_xlabel(r'Ankle angle $\theta$ (deg), $+$ = DF'); ax.set_ylabel('Manipulability\n(relative to barefoot at neutral)')
    ax.set_ylim(0, 1.15); ax.set_xlim(-50, 20); ax.legend(fontsize=6.5, frameon=False, loc='center left')
    ax.set_title('(a) vs ankle angle (red band: range with device)', loc='left', fontsize=9)
    # mean kinematic manipulability over each conventional workspace (uniform in theta)
    for name, rom in (('without device', ANK_ANAT), ('with device 5 % MVC', rom5)):
        t = np.radians(np.linspace(*rom, 400))
        lines.append(f'mean kinematic w over conventional workspace {name}: {np.cos(t).mean():.3f} of max, '
                     f'minimum {np.cos(t).min():.3f}')

    ax = axs[0, 1]
    for c, tags, col, shoe in (('U', ('U1', 'U2'), F.CU, False), ('C', ('C1', 'C2'), F.CC, True)):
        W = np.array([[manip(np.radians(90 - x), shoe)[1] for x in s['th']] for t in tags for s in strides[t]])
        W /= manip(0, False)[1]
        mu, sd = W.mean(0), W.std(0)
        ax.plot(F.GCX, mu, c=col, label=('unconstrained' if c == 'U' else 'constrained') + ', dynamic $w_d$')
        ax.fill_between(F.GCX, mu - sd, mu + sd, color=col, alpha=0.18, lw=0)
        K = np.array([np.cos(np.radians(90 - s['th'])) for t in tags for s in strides[t]])
        ax.plot(F.GCX, K.mean(0), c=col, ls='--', lw=1, label=('unconstrained' if c == 'U' else 'constrained') + ', kinematic $w$')
        lines.append(f'{c}: kinematic w over the stride {K.min():.3f}-{K.max():.3f} of max (mean {K.mean():.3f}); '
                     f'dynamic w_d mean {mu.mean():.3f} of barefoot-neutral (range {mu.min():.3f}-{mu.max():.3f})')
    ax.set_xlabel('Gait cycle (%)'); ax.set_ylabel('Manipulability (relative)'); ax.set_ylim(0, 1.1); ax.set_xlim(0, 100)
    ax.legend(fontsize=6.5, frameon=False, loc='lower left', ncol=1)
    ax.set_title('(b) along the measured trajectories', loc='left', fontsize=9)

    poses = {'U': [('peak DF', 87.2), ('push-off', 105.5), ('swing', 91.6)],
             'C': [('peak DF', 79.8), ('push-off', 94.8), ('swing', 90.5)]}
    for ax, kind, title in ((axs[1, 0], 'v', '(c) velocity ellipses (shank frame)'),
                            (axs[1, 1], 'd', '(d) dynamic ellipses (shank frame)')):
        for j, (c, col, shoe) in enumerate((('U', F.CU, False), ('C', F.CC, True))):
            for i, (name, Th) in enumerate(poses[c]):
                t = np.radians(90 - Th); xo = 0.45 * i
                ank, toe = fk(0.0, t); ank = ank + [xo, 0]; toe = toe + [xo, 0]
                ax.plot([xo, ank[0], toe[0]], [0, ank[1], toe[1]], c=col, lw=1, alpha=0.8)
                A = jacobian(0.0, t) if kind == 'v' else jacobian(0.0, t) @ np.linalg.inv(mass_matrix(0.0, t, shoe))
                ellipse(ax, toe, A, 0.13 if kind == 'v' else 0.006, ec=col, lw=1.4)
                if j == 0:
                    ax.text(xo + 0.05, 0.06, name, fontsize=7.5, ha='center')
        ax.set_aspect('equal'); ax.set_xlim(-0.08, 1.12); ax.set_ylim(-0.55, 0.12)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(title, loc='left', fontsize=9)
    fig.tight_layout(h_pad=1.2)
    F.save(fig, 'fig_manipulability.png')
    r = manip(0, True)[1] / manip(0, False)[1]
    lines.append(f'dynamic manipulability with shoe / barefoot at neutral = {r:.3f} ({100 * (r - 1):+.1f} %)')
    for d in (-15, 0, 10):
        mu, mc = manip(np.radians(d), False), manip(np.radians(d), True)
        lines.append(f'  theta {d:+d}: w = {mu[0] / w0:.3f} of max; w_d(shoe)/w_d(barefoot) = {mc[1] / mu[1]:.3f}')
    dm = [manip(0, True, qk)[1] / manip(0, False, qk)[1] for qk in np.radians([0, 30, 60])]
    lines.append('  knee-independence check of the ratio at knee 0/30/60 deg: ' + ', '.join(f'{x:.4f}' for x in dm))
    MU, MC = mass_matrix(0.3, 0, False), mass_matrix(0.3, 0, True)
    lines.append(f'  M (barefoot) = {np.round(MU, 4).tolist()}, M (shoe) = {np.round(MC, 4).tolist()} kg m^2 at knee 17 deg')


if __name__ == '__main__':
    strides, _, _ = F.build_strides()
    rng = measured_ranges(strides)
    lines = [f'segment lengths: shank {L_S * 1000:.0f} mm (0.246 H), foot (ankle-metatarsal) {L_F * 1000:.0f} mm; '
             f'kinematic manipulability max l_s*l_f = {L_S * L_F * 1e4:.1f} cm^2']
    rom5 = fig_workspace(rng, lines); fig_manipulability(strides, rng, rom5, lines)
    open(os.path.join(P.RES_ANALYSIS, 's5_workspace_manipulability.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))