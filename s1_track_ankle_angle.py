"""
HRI Class Project 2 - Part B: video-based ankle angle from coloured stickers.

Markers (right leg, camera on the subject's right side, subject faces image-right):
    pink   (magenta) stickers : shank  -> upper and lower
    yellow (fluorescent)      : foot   -> heel and toe

Angles reported:
    theta (video, CSV)    = included angle at the ankle between the shank (lower->upper pink sticker)
                            and the sole: 90 deg = neutral, < 90 dorsiflexion, > 90 plantarflexion
    ankle_angle_r (.mot)  = OpenSim convention = 90 - theta:
                            0 deg = neutral, + dorsiflexion, - plantarflexion
The angle is drawn at the ankle (lower pink sticker, near the lateral malleolus) between the shank
segment and a dashed line parallel to the sole.

Neutral calibration: the line through the two foot stickers is not exactly parallel to the sole.
During foot-flat (both foot stickers at their lowest image position) the sole lies on the
treadmill belt, which is horizontal in the image. The median tilt of the foot-sticker line in
those frames is taken as the sticker-line-to-sole offset and removed.

Outputs (in out_dir):
    <tag>_annotated.mp4   video with lines through the markers and the live ankle angle
    <tag>_ankle.csv       frame, time, raw marker positions, raw and filtered angle
    <tag>_ankle_r.mot     OpenSim motion file (time, ankle_angle_r in degrees, filtered)

Usage:
    python s1_track_ankle_angle.py VIDEO OUT_DIR TAG LABEL SEED_FRAME  pinkUp_x,y pinkLow_x,y yelHeel_x,y yelToe_x,y
"""
import sys, os, csv
import cv2
import numpy as np

# ---------------------------------------------------------------- colour thresholds (OpenCV HSV)
# Sampled from the videos: pink H~160-164, S~120-130; yellow stickers H~36-38, S~120-160;
# green screen H~40-41, S~185 (hence the saturation ceiling for yellow).
PINK = dict(lo=(150, 60, 90), hi=(172, 255, 255))   # upper bound excludes the red band clip (H~175+)
YELLOW = dict(lo=(28, 70, 100), hi=(39, 178, 255))
AREA_MIN, AREA_MAX = 60, 6000     # px, accepted blob area at full resolution
SEARCH_R = 110                    # px, search radius around the predicted marker position
MAX_JUMP = 140                    # px, reject candidates further than this from the prediction


def blobs(mask):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if AREA_MIN <= a <= AREA_MAX:
            out.append((cent[i][0], cent[i][1], a))
    return out


def colour_mask(hsv, rng):
    m = cv2.inRange(hsv, np.array(rng['lo']), np.array(rng['hi']))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def track_pair(hsv, rng, pred, ref_pair=None, anchor=None, ref_anchor=None):
    """Find the two markers of one colour near their predicted positions (2x2 array).
    ref_pair   : expected distance between the two markers (rejects implausible pairs)
    anchor     : position of a reliably tracked neighbouring marker (lower pink sticker)
    ref_anchor : expected distance of each marker to the anchor (rejects wires/background)"""
    H, W = hsv.shape[:2]
    x0 = int(max(0, pred[:, 0].min() - SEARCH_R)); x1 = int(min(W, pred[:, 0].max() + SEARCH_R))
    y0 = int(max(0, pred[:, 1].min() - SEARCH_R)); y1 = int(min(H, pred[:, 1].max() + SEARCH_R))
    cand = [(x + x0, y + y0, a) for x, y, a in blobs(colour_mask(hsv[y0:y1, x0:x1], rng))]
    found = np.full((2, 2), np.nan)
    if not cand:
        return found
    c = np.array([[x, y] for x, y, _ in cand])
    d = np.linalg.norm(c[None, :, :] - pred[:, None, :], axis=2)   # 2 x Ncand
    d[d > MAX_JUMP] = np.inf
    if anchor is not None and ref_anchor is not None:
        da = np.linalg.norm(c - anchor, axis=1)
        for k in range(2):                         # marker k must sit at a plausible distance
            bad = (da < 0.6 * ref_anchor[k]) | (da > 1.5 * ref_anchor[k])
            d[k, bad] = np.inf
    best, best_cost = (None, None), 2 * MAX_JUMP   # cost of declaring both missing
    opts = list(range(len(cand))) + [None]
    for i in opts:
        for j in opts:
            if i is not None and i == j:
                continue
            ci = d[0, i] if i is not None else MAX_JUMP
            cj = d[1, j] if j is not None else MAX_JUMP
            cost = ci + cj
            if i is not None and j is not None and ref_pair:
                sep = np.linalg.norm(c[i] - c[j])
                if not (0.6 * ref_pair < sep < 1.5 * ref_pair):
                    continue
                cost += 0.5 * abs(sep - ref_pair)
            if cost < best_cost:
                best_cost, best = cost, (i, j)
    for k, idx in enumerate(best):
        if idx is not None and np.isfinite(d[k, idx]):
            found[k] = c[idx]
    return found


def run_tracking(path, seed_frame, seeds):
    """Stream the video once. Frames before the seed are buffered (keep the seed early) and
    tracked backwards afterwards; frames from the seed on are tracked forwards on the fly."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows = []                                    # per-frame (4,2) marker positions
    early = []                                   # HSV frames before the seed

    def step(hsv, state):
        last, prev = state
        pred = last + 0.7 * (last - prev)        # damped constant-velocity prediction
        ref_pink = np.linalg.norm(last[0] - last[1])
        pk = track_pair(hsv, PINK, pred[0:2], ref_pair=ref_pink)
        low = pk[1] if not np.isnan(pk[1]).any() else pred[1]
        # foot markers: move rigidly with the lower shank sticker (ankle rotation per frame is small)
        pred_y = last[2:4] + (low - last[1])
        ref_ht = np.linalg.norm(last[2] - last[3])
        ref_anchor = np.linalg.norm(last[2:4] - last[1], axis=1)
        yl = track_pair(hsv, YELLOW, pred_y, ref_pair=ref_ht, anchor=low, ref_anchor=ref_anchor)
        meas = np.vstack([pk, yl])
        new = np.where(np.isnan(meas), np.vstack([pred[0:2], pred_y]), meas)
        return meas, (new, last)

    state = (np.array(seeds, float), np.array(seeds, float))
    f = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        if f < seed_frame:
            early.append(hsv); rows.append(None)
        else:
            meas, state = step(hsv, state)
            rows.append(meas)
        f += 1
    cap.release()
    state = (np.array(seeds, float), np.array(seeds, float))
    for f in range(seed_frame - 1, -1, -1):
        rows[f], state = step(early[f], state)
    return np.array(rows), fps


def fill_gaps(x, max_gap=8):
    """Linear interpolation over NaN runs up to max_gap frames."""
    x = x.copy(); idx = np.arange(len(x)); good = ~np.isnan(x)
    if good.sum() < 2:
        return x
    interp = np.interp(idx, idx[good], x[good])
    run = 0
    for i in range(len(x)):
        if np.isnan(x[i]):
            run += 1
        else:
            if 0 < run <= max_gap:
                x[i - run:i] = interp[i - run:i]
            run = 0
    return x


def butter_lowpass_filtfilt(x, fc, fs):
    """Zero-phase 2nd-order Butterworth low-pass (bilinear transform), applied to valid runs."""
    wc = np.tan(np.pi * fc / fs)
    k1 = np.sqrt(2) * wc; k2 = wc ** 2
    a0 = 1 + k1 + k2
    b = np.array([k2, 2 * k2, k2]) / a0
    a = np.array([1, 2 * (k2 - 1) / a0, (1 - k1 + k2) / a0])

    def lfilt(s):
        y = np.zeros_like(s); y[:2] = s[:2]
        for i in range(2, len(s)):
            y[i] = b[0]*s[i] + b[1]*s[i-1] + b[2]*s[i-2] - a[1]*y[i-1] - a[2]*y[i-2]
        return y

    out = np.full_like(x, np.nan)
    good = ~np.isnan(x)
    edges = np.flatnonzero(np.diff(np.r_[0, good.astype(int), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        seg = x[s:e]
        if len(seg) < 12:
            out[s:e] = seg; continue
        pad = min(30, len(seg) - 1)
        ext = np.r_[2*seg[0] - seg[pad:0:-1], seg, 2*seg[-1] - seg[-2:-pad-2:-1]]
        y = lfilt(lfilt(ext)[::-1])[::-1]
        out[s:e] = y[pad:pad + len(seg)]
    return out


def compute_angles(P):
    # flip image y so that +y is up
    up, low, heel, toe = (P[:, i] * [1, -1] for i in range(4))
    shank = up - low
    foot = toe - heel
    shank_tilt = np.degrees(np.arctan2(shank[:, 0], shank[:, 1]))   # forward lean from vertical
    foot_tilt = np.degrees(np.arctan2(foot[:, 1], foot[:, 0]))      # toe-up tilt from horizontal
    # included angle between shank (up) and foot (forward); 90 deg -> neutral
    cosang = np.sum(shank * foot, 1) / (np.linalg.norm(shank, axis=1) * np.linalg.norm(foot, axis=1))
    raw = 90 - np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    return raw, foot_tilt, shank_tilt


def foot_flat_offset(P, foot_tilt):
    hy, ty = P[:, 2, 1], P[:, 3, 1]       # image y grows downward -> lowest = largest y
    ok = ~np.isnan(hy) & ~np.isnan(ty) & ~np.isnan(foot_tilt)
    flat = ok & (hy > np.nanpercentile(hy, 90) - 6) & (ty > np.nanpercentile(ty, 90) - 6)
    return float(np.median(foot_tilt[flat])), int(flat.sum())


def sole_direction(P_i, offset):
    """Unit vector (image coords) of the sole: foot-sticker line rotated toe-up by -offset."""
    f = (P_i[3] - P_i[2]) * [1, -1]                      # to math frame (y up)
    a = np.radians(-offset)
    f = np.array([f[0]*np.cos(a) - f[1]*np.sin(a), f[0]*np.sin(a) + f[1]*np.cos(a)])
    f = f / (np.linalg.norm(f) + 1e-9)
    return f * [1, -1]                                   # back to image frame


def blend_rect(img, p0, p1, colour=(25, 25, 25), alpha=0.6):
    over = img.copy()
    cv2.rectangle(over, p0, p1, colour, -1)
    cv2.addWeighted(over, alpha, img, 1 - alpha, 0, img)


def write_outputs(path, out_dir, tag, label, P, fps, angle_raw, angle_f, offset, make_video=True):
    """angle_* : OpenSim convention (0 = neutral, + dorsiflexion).
    theta     : included shank-sole angle at the ankle = 90 - ankle angle (90 deg = neutral,
                < 90 dorsiflexion, > 90 plantarflexion)."""
    n = len(P); t = np.arange(n) / fps
    theta_raw, theta_f = 90 - angle_raw, 90 - angle_f
    with open(os.path.join(out_dir, f"{tag}_ankle.csv"), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['frame', 'time_s', 'pinkUp_x', 'pinkUp_y', 'pinkLow_x', 'pinkLow_y',
                    'heel_x', 'heel_y', 'toe_x', 'toe_y',
                    'theta_shank_sole_raw_deg', 'theta_shank_sole_filt_deg',
                    'ankle_raw_deg', 'ankle_filt_deg'])
        for i in range(n):
            w.writerow([i, f"{t[i]:.4f}"] + [f"{v:.1f}" for v in P[i].ravel()]
                       + [f"{theta_raw[i]:.2f}", f"{theta_f[i]:.2f}",
                          f"{angle_raw[i]:.2f}", f"{angle_f[i]:.2f}"])

    good = ~np.isnan(angle_f)
    with open(os.path.join(out_dir, f"{tag}_ankle_r.mot"), 'w') as fh:
        fh.write(f"{tag}_ankle_r\nversion=1\nnRows={good.sum()}\nnColumns=2\ninDegrees=yes\nendheader\n")
        fh.write("time\tankle_angle_r\n")
        for i in np.flatnonzero(good):
            fh.write(f"{t[i]:.5f}\t{angle_f[i]:.4f}\n")

    if not make_video:
        return
    # ---------------- annotated video (angle only; see s8_render_annotated_videos.py for angle + EMG) -------
    cap = cv2.VideoCapture(path)
    W_out, H_out = 720, 1280
    sc = W_out / int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vw = cv2.VideoWriter(os.path.join(out_dir, f"{tag}_annotated.mp4"),
                         cv2.VideoWriter_fourcc(*'mp4v'), fps, (W_out, H_out))
    PINK_C, YEL_C, WHITE = (190, 120, 255), (0, 220, 255), (255, 255, 255)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    win = int(4 * fps)                                         # 4 s rolling plot
    TH_LO, TH_HI = 70.0, 115.0                                 # fixed theta axis (same for both videos)
    run_min, run_max = np.inf, -np.inf

    def ip(v):
        return tuple(int(round(x)) for x in v)

    for i in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (W_out, H_out))
        p = P[i] * sc
        th = theta_f[i]
        if not np.isnan(th):
            run_min, run_max = min(run_min, th), max(run_max, th)

        if not np.isnan(p).any():
            up, ank, heel, toe = p
            # segments between markers
            cv2.line(fr, ip(ank), ip(up), PINK_C, 5, cv2.LINE_AA)
            # solid foot line = the sole: through the midpoint of the two foot stickers, but
            # oriented along the calibrated sole direction (sticker line corrected by the offset),
            # so it is parallel to the dashed line used for the ankle angle
            sd = sole_direction(P[i], offset)
            mid_ft = 0.5 * (heel + toe); half = 0.5 * np.linalg.norm(toe - heel)
            cv2.line(fr, ip(mid_ft - sd * half), ip(mid_ft + sd * half), YEL_C, 5, cv2.LINE_AA)
            # dashed sole line through the ankle vertex (parallel to the sole)
            L = 0.8 * np.linalg.norm(toe - heel)
            for s0 in np.arange(0, L, 14):
                cv2.line(fr, ip(ank + sd * s0), ip(ank + sd * min(s0 + 8, L)), YEL_C, 2, cv2.LINE_AA)
            # arc at the ankle between the shank and the sole
            v_s = up - ank
            a_s = np.degrees(np.arctan2(v_s[1], v_s[0]))
            a_f = np.degrees(np.arctan2(sd[1], sd[0]))
            diff = (a_f - a_s + 180) % 360 - 180
            start, end = (a_s, a_s + diff) if diff >= 0 else (a_s + diff, a_s)
            r = 38
            cv2.ellipse(fr, ip(ank), (r, r), 0, start, end, WHITE, 3, cv2.LINE_AA)
            mid = np.radians((start + end) / 2)
            lab_pos = ank + (r + 26) * np.array([np.cos(mid), np.sin(mid)]) + [-14, 8]
            if not np.isnan(th):
                cv2.putText(fr, f"{th:.0f}", ip(lab_pos), FONT, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(fr, f"{th:.0f}", ip(lab_pos), FONT, 0.75, WHITE, 2, cv2.LINE_AA)
            for k, col in [(0, PINK_C), (1, PINK_C), (2, YEL_C), (3, YEL_C)]:
                cv2.circle(fr, ip(p[k]), 10, col, -1, cv2.LINE_AA)
                cv2.circle(fr, ip(p[k]), 10, WHITE, 2, cv2.LINE_AA)

        # ---- info panel (top-left, translucent)
        blend_rect(fr, (10, 10), (470, 200))
        cv2.putText(fr, f"ANKLE (sagittal) - {label}", (22, 38), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
        if not np.isnan(th):
            dev = 90 - th                                      # = ankle angle, + DF
            kind = 'DF' if dev >= 0 else 'PF'
            cv2.putText(fr, f"theta  {th:5.1f} deg", (22, 80), FONT, 1.0, WHITE, 2, cv2.LINE_AA)
            cv2.putText(fr, f"deviation from 90:  {abs(dev):4.1f} deg {kind}", (22, 115), FONT, 0.65,
                        YEL_C, 2, cv2.LINE_AA)
        else:
            cv2.putText(fr, "marker lost", (22, 80), FONT, 1.0, WHITE, 2, cv2.LINE_AA)
        if np.isfinite(run_min):
            cv2.putText(fr, f"min / max so far:  {run_min:5.1f} / {run_max:5.1f} deg", (22, 148), FONT,
                        0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(fr, f"t {t[i]:5.2f} s   frame {i}", (22, 180), FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # ---- rolling theta plot (bottom, translucent)
        x0g, x1g, y0g, y1g = 20, W_out - 20, H_out - 250, H_out - 20
        blend_rect(fr, (x0g, y0g), (x1g, y1g))
        gy0, gy1 = y0g + 40, y1g - 15                          # plotting area

        def ymap(v):
            return int(gy0 + (TH_HI - v) / (TH_HI - TH_LO) * (gy1 - gy0))

        cv2.putText(fr, "theta (shank-sole) [deg], last 4 s   (<90 DF / >90 PF)", (x0g + 12, y0g + 25),
                    FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        for tick in range(int(TH_LO), int(TH_HI) + 1, 10):
            yy = ymap(tick)
            col = (230, 230, 230) if tick == 90 else (110, 110, 110)
            cv2.line(fr, (x0g + 45, yy), (x1g - 10, yy), col, 1 if tick != 90 else 2, cv2.LINE_AA)
            cv2.putText(fr, str(tick), (x0g + 8, yy + 5), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        seg = theta_f[max(0, i - win):i + 1]
        pts = [(int(x0g + 45 + (x1g - x0g - 55) * (k + win - len(seg)) / win), ymap(np.clip(v, TH_LO, TH_HI)))
               for k, v in enumerate(seg) if not np.isnan(v)]
        if len(pts) > 1:
            cv2.polylines(fr, [np.array(pts, np.int32)], False, YEL_C, 3, cv2.LINE_AA)
        vw.write(fr)
    cap.release(); vw.release()


if __name__ == "__main__":
    path, out_dir, tag, label, seed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    seeds = [tuple(map(float, s.split(','))) for s in sys.argv[6:10]]
    P, fps = run_tracking(path, seed, seeds)
    for i in range(4):
        for j in range(2):
            P[:, i, j] = fill_gaps(P[:, i, j])
    raw, foot_tilt, _ = compute_angles(P)
    offset, nflat = foot_flat_offset(P, foot_tilt)
    # sticker line tilted toe-down by |offset| relative to the sole -> true sole is |offset| more
    # toe-up -> shank-sole angle smaller -> more dorsiflexion: subtract the (negative) offset
    raw_cal = raw - offset
    # spike rejection: samples deviating > 8 deg from a 7-frame running median are treated as
    # mis-detections (e.g. a sticker briefly confused with another object) and interpolated
    pad = np.pad(raw_cal, 3, mode='edge')
    med = np.array([np.nanmedian(pad[i:i + 7]) for i in range(len(raw_cal))])
    spikes = np.abs(raw_cal - med) > 8
    raw_cal = raw_cal.copy(); raw_cal[spikes] = np.nan
    raw_cal = fill_gaps(raw_cal, max_gap=8)
    print(f"  spike frames removed: {int(spikes.sum())}")
    filt = butter_lowpass_filtfilt(raw_cal, 6.0, fps)
    found = ~np.isnan(P).any(axis=(1, 2))
    print(f"{tag}: {len(P)} frames at {fps:.2f} fps, all 4 markers found in {found.mean()*100:.1f}% of frames")
    print(f"  foot-sticker line offset vs sole = {offset:+.2f} deg (from {nflat} foot-flat frames)")
    print(f"  ankle angle (filtered): min {np.nanmin(filt):+.1f}, max {np.nanmax(filt):+.1f}, "
          f"range {np.nanmax(filt)-np.nanmin(filt):.1f} deg")
    print(f"  theta shank-sole (filtered): min {np.nanmin(90-filt):.1f}, max {np.nanmax(90-filt):.1f} deg")
    make_video = not (len(sys.argv) > 10 and sys.argv[10] == 'novideo')
    write_outputs(path, out_dir, tag, label, P, fps, raw_cal, filt, offset, make_video)
    with open(os.path.join(out_dir, f"{tag}_calibration.txt"), 'w') as fh:
        fh.write(f"fps={fps:.4f}\nsticker_to_sole_offset_deg={offset:.3f}\nfoot_flat_frames={nflat}\n")
    print("  outputs written")
