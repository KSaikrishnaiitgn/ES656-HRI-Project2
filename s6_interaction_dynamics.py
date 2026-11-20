"""
HRI Class Project 2 - Part B: how the imposed dynamics act on the limb during walking (brief step 2).

The device model (s4_spring_shoe_model.py) is evaluated stride by stride along the MEASURED ankle
trajectory (theta = 90 deg - Theta) of the constrained trials:
  * wire tensions F_f, F_b and device torque tau_dev           (spring / stiffness term)
  * inertial torque of the shoe, J_shoe * theta_ddot           (inertia term; shank frame,
                                                                  shank acceleration neglected)
  * force applied by the wires to the shank band, split into the component along the shank
    (axial) and across it (antero-posterior)                     (forces at the point of interaction)
For comparison, the same spring model is evaluated along the UNCONSTRAINED trajectory, i.e. the
torque the device WOULD have applied had the subject kept the barefoot ankle pattern
(a model counterfactual, not a measurement). Damping is not modelled (not measured).

All values are model predictions at measured angles; no force was measured. The constrained
absolute angles may contain a constant shoe offset (see report, Section 7.4).

Run from 02_code:  python s6_interaction_dynamics.py
Outputs: 04_report/figures/fig_interaction_gait_cycle.png, 03_results/analysis/s6_interaction_dynamics.txt
"""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config_paths as P
import s4_spring_shoe_model as M
import s7_report_figures as F

os.makedirs(P.RES_ANALYSIS, exist_ok=True)


def device_state(th):
    """Wire tensions, device torque and force on the band (shank frame) at ankle angle th (rad)."""
    tau, f, b = M.device_torque(th)
    Fband = np.zeros(2)
    for (Fi, *_), A, P0 in ((f, M.A_f, M.P_f0), (b, M.A_b, M.P_b0)):
        Pp = M.rot(th) @ P0; u = (A - Pp) / np.linalg.norm(A - Pp)
        Fband += -Fi * u                          # the wire pulls the band towards the foot
    return f[0], b[0], tau, Fband


def per_condition(strides, tags):
    out = {k: [] for k in ('Ff', 'Fb', 'tau', 'tin', 'ax', 'ap')}
    for t in tags:
        for s in strides[t]:
            th = np.radians(90 - s['th']); thdd = -np.radians(s['al'])
            st = np.array([np.r_[device_state(x)[:3], device_state(x)[3]] for x in th])
            out['Ff'].append(st[:, 0]); out['Fb'].append(st[:, 1]); out['tau'].append(st[:, 2])
            out['ap'].append(st[:, 3]); out['ax'].append(st[:, 4])
            out['tin'].append(M.J_shoe * thdd)
    return {k: np.array(v) for k, v in out.items()}


def band(ax, y, col, label, ls='-', lw=1.5, alpha=0.18):
    mu, sd = y.mean(0), y.std(0)
    ax.plot(F.GCX, mu, c=col, ls=ls, lw=lw, label=label)
    if alpha:
        ax.fill_between(F.GCX, mu - sd, mu + sd, color=col, alpha=alpha, lw=0)


def main():
    strides, _, _ = F.build_strides()
    C = per_condition(strides, ('C1', 'C2')); U = per_condition(strides, ('U1', 'U2'))
    to_c = np.nanmean([s['to_pct'] for t in ('C1', 'C2') for s in strides[t]])
    to_u = np.nanmean([s['to_pct'] for t in ('U1', 'U2') for s in strides[t]])
    fig, axs = plt.subplots(3, 1, figsize=(6.3, 7.0), sharex=True)
    band(axs[0], C['Ff'], F.CF, 'front wire S1 (constrained trajectory)')
    band(axs[0], C['Fb'], F.CB, 'back wire S2 (constrained trajectory)')
    band(axs[0], U['Ff'], F.CF, 'front wire, barefoot trajectory (counterfactual)', ls='--', lw=1, alpha=0)
    axs[0].set_ylabel('Wire tension (N)')
    band(axs[1], C['tau'], 'k', r'spring torque $\tau_\mathrm{dev}$ (constrained trajectory)')
    band(axs[1], U['tau'], '0.5', r'$\tau_\mathrm{dev}$ on barefoot trajectory (counterfactual)', ls='--', lw=1, alpha=0)
    band(axs[1], C['tin'], '#9467bd', r'shoe inertial torque $J_\mathrm{shoe}\ddot\theta$')
    axs[1].axhline(0, c='0.4', lw=0.6); axs[1].set_ylabel('Torque on foot (N m)\n$+$ = DF')
    band(axs[2], C['ax'], '#8c564b', 'along the shank ($-$ = pulls band towards ankle)')
    band(axs[2], C['ap'], '#17becf', 'across the shank ($+$ = forward)')
    axs[2].axhline(0, c='0.4', lw=0.6); axs[2].set_ylabel('Force on shank band (N)')
    for ax, l in zip(axs, 'abc'):
        F.phase_guides(ax, to_u, to_c); ax.legend(fontsize=7, frameon=False, loc='upper left')
        ax.set_title(f'({l})', loc='left', fontsize=9)
    axs[0].set_ylim(0, 48); axs[1].set_ylim(-3.5, 7.0); axs[2].set_ylim(-26, 16)
    axs[2].set_xlabel(F.GC_LABEL); axs[2].set_xlim(0, 100)
    fig.tight_layout(h_pad=0.6)
    F.save(fig, 'fig_interaction_gait_cycle.png')

    def pk(a, f=np.max):
        v = f(a, axis=1); return f"{v.mean():.2f} +/- {v.std():.2f}"
    L = ['Model predictions along measured trajectories (per-stride peaks, mean +/- SD over strides)',
         f"constrained (n={len(C['tau'])}): front tension peak {pk(C['Ff'])} N; back tension peak {pk(C['Fb'])} N",
         f"  device torque max (DF) {pk(C['tau'])} N m, min (PF) {pk(C['tau'], np.min)} N m",
         f"  shoe inertial torque max {pk(C['tin'])} N m, min {pk(C['tin'], np.min)} N m",
         f"  RMS over stride: spring torque {np.sqrt((C['tau'] ** 2).mean()):.2f} N m, shoe inertial torque {np.sqrt((C['tin'] ** 2).mean()):.2f} N m",
         f"  band force along shank min {pk(C['ax'], np.min)} N; across shank max {pk(C['ap'])} N, min {pk(C['ap'], np.min)} N",
         f"counterfactual on barefoot trajectory (n={len(U['tau'])}): front tension peak {pk(U['Ff'])} N, device torque max {pk(U['tau'])} N m",
         f"  ratio of mean peak device torque, constrained / barefoot trajectory = {C['tau'].max(1).mean() / U['tau'].max(1).mean():.2f}",
         f"mean peak angular acceleration |theta_dd|: constrained {np.abs(C['tin']).max(1).mean() / M.J_shoe:.0f} rad/s^2"]
    for ph, (a, b) in (('loading 0-15', (0, 16)), ('mid/terminal stance 15-50', (15, 51)), ('push-off 50-70', (50, 71)), ('swing 72-100', (72, 101))):
        L.append(f"  phase {ph}: mean tau_dev {C['tau'][:, a:b].mean():+.2f} N m, mean |shoe inertial| {np.abs(C['tin'][:, a:b]).mean():.2f} N m")
    open(os.path.join(P.RES_ANALYSIS, 's6_interaction_dynamics.txt'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
