"""
HRI Class Project 2 - Part B
Model of the external constraint: shoe with front/back projections connected to a
shank band (just below the knee) by inextensible wires in series with extension springs.

Sagittal-plane model. Origin O = ankle joint axis (lateral malleolus).
x = forward (toe direction), y = up along the shank. Shank is held fixed (shank frame).
theta = ankle angle of the foot relative to neutral (90 deg):
    theta > 0 -> dorsiflexion (toe up), theta < 0 -> plantarflexion.

Measured values (from the sketch):
    k        = 754.6 N/m (both springs)
    L1 = 340 mm  front link length (band -> front projection), neutral posture
    L2 = 330 mm  back  link length (band -> back projection),  neutral posture
    L3 = 170 mm  ankle -> front projection (horizontal)
    L4 = 130 mm  ankle -> back projection  (horizontal)
    W  = 122 mm  horizontal distance between the two band attachment points (shin <-> calf)
    S1 free 30.05 mm, attached 31.90 mm  -> front pretension stretch 1.85 mm
    S2 free 30.01 mm, attached 32.95 mm  -> back  pretension stretch 2.94 mm

ASSUMPTIONS (to be confirmed):
    h  = vertical offset of the projections below the ankle axis (sketch suggests ~0)
    Springs have zero initial tension (force = k * stretch beyond free length).
    Wire goes slack (zero force) when the spring returns to its free length.
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------- parameters (SI units) ----------------
k = 754.6                     # N/m
L1, L2 = 0.340, 0.330         # m, link lengths at neutral
L3, L4 = 0.170, 0.130         # m, projection distances from ankle
W = 0.122                     # m, band attachment spacing
d1_0 = (31.90 - 30.05) / 1000 # m, front spring pretension stretch
d2_0 = (32.95 - 30.01) / 1000 # m, back spring pretension stretch
h = 0.0                       # m, projection height below ankle axis (confirmed: same level)

# ---------------- subject / segment parameters ----------------
g = 9.81
M_body = 60.05                # kg (Vicon subject file)
H_body = 1.670                # m
m_foot = 0.0145 * M_body      # Winter (2009) foot mass fraction
l_foot = 0.15                 # m, ankle -> 2nd metatarsal head (ESTIMATE, measure with tape)
x_foot = 0.50 * l_foot        # foot COM forward of ankle (Winter: 0.5 of segment)
y_foot = -0.04                # m, foot COM below ankle axis (estimate)
J_foot = m_foot * (0.690 * l_foot)**2   # Winter radius of gyration about proximal end
m_shoe = 0.99                 # kg, shoe + projections + attachments (measured)
# shoe modelled as a uniform bar from the back tip (-L4) to the front tip (+L3)
x_shoe = (L3 - L4) / 2
y_shoe = -0.04
J_shoe = m_shoe * (L3**2 - L3 * L4 + L4**2) / 3 + m_shoe * y_shoe**2
J_total = J_foot + J_shoe
walk_speed = 3.0 / 3.6        # m/s, right leg fitted with the device

# ---------------- geometry reconstruction ----------------
# Unknowns: a = x of front band point, D = vertical drop band -> projections.
# Back band point x = a - W.
# (L3 - a)^2 + D^2 = L1^2 ;  (-L4 - (a - W))^2 + D^2 = L2^2
c = W - L4                                    # back: (c - a)^2
a = (L3**2 - c**2 - (L1**2 - L2**2)) / (2 * (L3 - c))
D = np.sqrt(L1**2 - (L3 - a)**2)
H = D - h                                     # band height above ankle axis

A_f = np.array([a, H])                        # front band point (shank frame)
A_b = np.array([a - W, H])                    # back band point
P_f0 = np.array([L3, -h])                     # front projection (foot frame)
P_b0 = np.array([-L4, -h])                    # back projection


def rot(th):
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def link(th, A, P0, L0, d0):
    """Return spring force (N), moment arm (m, signed), torque on foot (N m)."""
    P = rot(th) @ P0
    d = A - P
    ell = np.linalg.norm(d)
    stretch = d0 + (ell - L0)
    F = k * stretch if stretch > 0 else 0.0   # wire slack -> no force
    u = d / ell
    r = P[0] * u[1] - P[1] * u[0]             # signed moment arm (z-component of P x u)
    return F, r, F * r, stretch


def device_torque(th):
    Ff, rf, tf, sf = link(th, A_f, P_f0, L1, d1_0)
    Fb, rb, tb, sb = link(th, A_b, P_b0, L2, d2_0)
    return tf + tb, (Ff, rf, tf, sf), (Fb, rb, tb, sb)


if __name__ == "__main__":
    print(f"Reconstructed band points: front x = {a*1000:.1f} mm, back x = {(a-W)*1000:.1f} mm")
    print(f"Band height above projections D = {D*1000:.1f} mm (above ankle H = {H*1000:.1f} mm, h = {h*1000:.0f} mm)")
    print(f"Front wire tilt from vertical = {np.degrees(np.arctan2(L3-a, D)):.1f} deg, "
          f"back = {np.degrees(np.arctan2(a-W+L4, D)):.1f} deg")

    _, (Ff, rf, tf, _), (Fb, rb, tb, _) = device_torque(0.0)
    print(f"\nNeutral: front T = {Ff:.2f} N, r_f = {rf*1000:.1f} mm; back T = {Fb:.2f} N, r_b = {rb*1000:.1f} mm")
    print(f"Net torque at neutral = {tf+tb:.4f} N m  (+ = dorsiflexion)")
    K_both = k * (rf**2 + rb**2)
    print(f"Small-angle stiffness: both taut {K_both:.2f} N m/rad; "
          f"dorsiflexion (back only) {k*rb**2:.2f}; plantarflexion (front only) {k*rf**2:.2f} N m/rad")
    th_eq = (tf + tb) / K_both
    print(f"Device-alone equilibrium angle = {np.degrees(th_eq):.2f} deg")
    print(f"Front wire slack beyond +{np.degrees(d1_0/rf):.2f} deg dorsiflexion; "
          f"back wire slack beyond {np.degrees(-d2_0/abs(rb)):.2f} deg plantarflexion")

    print("\n angle | torque  | F_front stretch | F_back stretch")
    for deg in [-20, -15, -10, -5, -2, 0, 2, 5, 10, 15, 20]:
        t, f, b = device_torque(np.radians(deg))
        print(f"{deg:+5d}  | {t:+6.2f}  | {f[0]:5.1f} N {f[3]*1000:5.1f}mm | {b[0]:5.1f} N {b[3]*1000:5.1f}mm")

    # ---------------- swing phase (foot off ground, shank ~vertical, muscles relaxed) -------------
    def gravity_torque(th, with_shoe=True):
        tau = 0.0
        parts = [(m_foot, x_foot, y_foot)] + ([(m_shoe, x_shoe, y_shoe)] if with_shoe else [])
        for m, x, y in parts:
            xr = x * np.cos(th) - y * np.sin(th)   # horizontal COM position after rotation
            tau += -m * g * xr                     # weight forward of ankle -> plantarflexing
        return tau

    def brentq(f, lo, hi, n=80):              # simple bisection (avoids a scipy dependency)
        for _ in range(n):
            mid = 0.5 * (lo + hi)
            if np.sign(f(mid)) == np.sign(f(lo)):
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    print(f"\nFoot: m = {m_foot:.3f} kg, J = {J_foot:.4f} kg m^2;  Shoe+device: m = {m_shoe} kg, J = {J_shoe:.4f}")
    print(f"Total J about ankle = {J_total:.4f} kg m^2")
    print(f"Gravity torque at neutral: foot only {gravity_torque(0, False):.3f}, foot+shoe {gravity_torque(0):.3f} N m")
    th_sw = brentq(lambda t: device_torque(t)[0] + gravity_torque(t), np.radians(-40), np.radians(10))
    print(f"Swing equilibrium (relaxed muscles, device on) = {np.degrees(th_sw):.2f} deg")
    print("  -> without device the foot would hang in plantarflexion until passive tissue/TA holds it")
    for name, K in [("both springs", k*(rf**2+rb**2)), ("plantarflexion side", k*rf**2), ("dorsiflexion side", k*rb**2)]:
        print(f"Natural frequency foot+shoe on springs ({name}): {np.sqrt(K/J_total)/(2*np.pi):.1f} Hz")

    # torque-angle curve
    ths = np.radians(np.linspace(-25, 25, 501))
    tau = np.array([device_torque(t)[0] for t in ths])
    Kd = np.gradient(tau, ths)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(np.degrees(ths), tau, lw=2)
    ax[0].axhline(0, c='k', lw=0.5); ax[0].axvline(0, c='k', lw=0.5)
    ax[0].set_xlabel('Ankle angle (deg)  [+ dorsiflexion]')
    ax[0].set_ylabel('Device torque on foot (N m)  [+ dorsiflexing]')
    ax[0].set_title('Restoring torque of shoe-spring constraint')
    ax[1].plot(np.degrees(ths), -Kd, lw=2, c='tab:orange')
    ax[1].set_xlabel('Ankle angle (deg)'); ax[1].set_ylabel('Rotational stiffness (N m/rad)')
    ax[1].set_title('Local stiffness  K = -dτ/dθ')
    for a_ in ax: a_.grid(alpha=0.3)
    plt.tight_layout()
    import os, config_paths as P; os.makedirs(P.RES_FIG, exist_ok=True)
    plt.savefig(os.path.join(P.RES_FIG, 'spring_shoe_torque_angle.png'), dpi=150)
    print("\nSaved plot.")
