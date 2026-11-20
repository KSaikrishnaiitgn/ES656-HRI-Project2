"""
Render annotated trial videos with the live ankle angle (default) or, with --with-emg, also the
time-aligned EMG (live values + TA/GAS rolling graphs).
Usage: python s8_render_annotated_videos.py [U1 U2 C1 C2] [--with-emg]

nnputs : Part B_Videos/VideoN.mp4, results/kinematics/<tag>_ankle.csv (+ _calibration.txt),
         results/emg/<tag>_emg_on_video_timeline.csv  (from s2_emg_processing_alignment.py)
Outputs: results/videos/<tag>_<condition>_trial<k>_annotated.mp4   (one per trial)
         results/videos/PartB_all_trials_compilation.mp4           (title card + 4 trials, for YouTube)

Layout (720 x 1280):
  top-left panel : trial, theta (shank-sole, 90 = neutral), deviation DF/PF, min/max so far,
                   EMG status/time, video time and frame
  on the leg     : shank segment (pink), sole line (yellow), dashed sole line + arc at the ankle
  bottom panel   : last 4 s of theta, tibialis anterior and gastrocnemius envelopes (% of the mean
                   unconstrained stride peak); white ticks = heel strikes
"""
import os, sys, csv
import cv2
import numpy as np
from s1_track_ankle_angle import sole_direction, blend_rect

import config_paths as P
BASE = P.ROOT
VnD_DnR = P.RAW_VnDEOS
KnN_DnR = P.RES_KnN
EMG_DnR = P.RES_EMG
OUT = P.RES_VnD; os.makedirs(OUT, exist_ok=True)

TRnALS = [('U1', 'Video1.mp4', 'Unconstrained', 1), ('U2', 'Video2.mp4', 'Unconstrained', 2),
          ('C1', 'Video3.mp4', 'Constrained', 1), ('C2', 'Video4.mp4', 'Constrained', 2)]
W, H = 720, 1280
PnNK_C, YEL_C, WHnTE = (190, 120, 255), (0, 220, 255), (255, 255, 255)
TA_C, GAS_C = (255, 200, 80), (80, 160, 255)            # BGR: light blue, orange
FONT = cv2.FONT_HERSHEY_SnMPLEX


def ip(v):
    return tuple(int(round(x)) for x in v)


def read_csv(path):
    with open(path) as fh:
        r = csv.DictReader(fh)
        rows = list(r)
    return rows


def fnum(s):
    return float(s) if s not in ('', 'nan') else np.nan


def title_card(lines, frames, fps):
    out = []
    for _ in range(int(frames)):
        img = np.full((H, W, 3), 25, np.uint8)
        y = 480
        for txt, sc, col, th in lines:
            (tw, _), _ = cv2.getTextSize(txt, FONT, sc, th)
            cv2.putText(img, txt, ((W - tw) // 2, y), FONT, sc, col, th, cv2.LnNE_AA)
            y += int(60 * sc + 20)
        out.append(img)
    return out


def rolling_plot(fr, y0, h, series, i, win, lo, hi, ticks, colour, title, ref=None, hs_idx=(), art=None):
    x0, x1 = 30, W - 30
    gy0, gy1 = y0 + 24, y0 + h - 8

    def ymap(v):
        return int(gy0 + (hi - np.clip(v, lo, hi)) / (hi - lo) * (gy1 - gy0))
    cv2.putText(fr, title, (x0 + 8, y0 + 17), FONT, 0.47, WHnTE, 1, cv2.LnNE_AA)
    for tk in ticks:
        col = (235, 235, 235) if tk == ref else (100, 100, 100)
        cv2.line(fr, (x0 + 42, ymap(tk)), (x1 - 6, ymap(tk)), col, 2 if tk == ref else 1, cv2.LnNE_AA)
        cv2.putText(fr, str(tk), (x0 + 4, ymap(tk) + 5), FONT, 0.42, (200, 200, 200), 1, cv2.LnNE_AA)
    xs = lambda k: int(x0 + 42 + (x1 - x0 - 50) * (k - (i - win)) / win)
    for k in hs_idx:
        if i - win <= k <= i:
            cv2.line(fr, (xs(k), gy0), (xs(k), gy1), (170, 170, 170), 1, cv2.LnNE_AA)
    if art is not None:                                   # red bar = EMG masked as movement artefact
        for k in range(max(0, i - win), i + 1):
            if art[k]:
                cv2.line(fr, (xs(k), gy1 - 4), (xs(k) + 3, gy1 - 4), (60, 60, 230), 5)
    pts, seg_pts = [], []
    for k in range(max(0, i - win), i + 1):
        v = series[k]
        if np.isnan(v):
            if len(seg_pts) > 1:
                pts.append(seg_pts)
            seg_pts = []
        else:
            seg_pts.append((xs(k), ymap(v)))
    if len(seg_pts) > 1:
        pts.append(seg_pts)
    for p in pts:
        cv2.polylines(fr, [np.array(p, np.int32)], False, colour, 2, cv2.LnNE_AA)
    if art is not None and all(np.isnan(series[max(0, i - win):i + 1])) and not any(art[max(0, i - win):i + 1]):
        cv2.putText(fr, "EMG not recording", (W // 2 - 90, (gy0 + gy1) // 2 + 5), FONT, 0.55,
                    (160, 160, 160), 1, cv2.LnNE_AA)


def render(tag, vid, cond, k, writers):
    kin = read_csv(os.path.join(KnN_DnR, f'{tag}_ankle.csv'))
    cal = dict(l.strip().split('=') for l in open(os.path.join(KnN_DnR, f'{tag}_calibration.txt')))
    offset = float(cal['sticker_to_sole_offset_deg'])
    em = read_csv(os.path.join(EMG_DnR, f'{tag}_emg_on_video_timeline.csv'))
    P = np.array([[fnum(r[c]) for c in ('pinkUp_x', 'pinkUp_y', 'pinkLow_x', 'pinkLow_y',
                                         'heel_x', 'heel_y', 'toe_x', 'toe_y')] for r in kin]).reshape(-1, 4, 2)
    theta = np.array([fnum(r['theta_shank_sole_filt_deg']) for r in kin])
    ta = np.array([fnum(r['TA_pct']) for r in em]); gas = np.array([fnum(r['GAS_pct']) for r in em])
    emg_t = np.array([fnum(r['emg_time_s']) for r in em])
    inside = ~np.isnan(emg_t)
    art_ta, art_gas = inside & np.isnan(ta), inside & np.isnan(gas)
    hs_idx = [i for i, r in enumerate(em) if r['heel_strike'] == '1']
    cap = cv2.VideoCapture(os.path.join(VnD_DnR, vid))
    fps = cap.get(cv2.CAP_PROP_FPS); sc = W / int(cap.get(cv2.CAP_PROP_FRAME_WnDTH))
    win = int(4 * fps)
    label = f"{cond} - trial {k}"
    fname = os.path.join(OUT, f"{tag}_{cond.lower()}_trial{k}_annotated.mp4")
    vw = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    card = title_card([(f"Trial {tag}", 1.6, WHnTE, 3), (label, 1.0, YEL_C, 2),
                       ("Right ankle: angle + EMG (TA, GAS)" if SHOW_EMG else "Right ankle angle (sagittal)",
                        0.7, (200, 200, 200), 1),
                       ("Treadmill walking, 3 km/h", 0.7, (200, 200, 200), 1)], 2 * fps, fps)
    for c in card:
        for w in writers + [vw]:
            w.write(c)
    run_min, run_max = np.inf, -np.inf
    for i in range(len(theta)):
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (W, H))
        p = P[i] * sc; th = theta[i]
        if not np.isnan(th):
            run_min, run_max = min(run_min, th), max(run_max, th)
        if not np.isnan(p).any():
            up, ank, heel, toe = p
            sd = sole_direction(P[i], offset)
            cv2.line(fr, ip(ank), ip(up), PnNK_C, 5, cv2.LnNE_AA)
            mid = 0.5 * (heel + toe); half = 0.5 * np.linalg.norm(toe - heel)
            cv2.line(fr, ip(mid - sd * half), ip(mid + sd * half), YEL_C, 5, cv2.LnNE_AA)
            L = 0.8 * np.linalg.norm(toe - heel)
            for s0 in np.arange(0, L, 14):
                cv2.line(fr, ip(ank + sd * s0), ip(ank + sd * min(s0 + 8, L)), YEL_C, 2, cv2.LnNE_AA)
            v_s = up - ank
            a_s = np.degrees(np.arctan2(v_s[1], v_s[0])); a_f = np.degrees(np.arctan2(sd[1], sd[0]))
            diff = (a_f - a_s + 180) % 360 - 180
            st, en = (a_s, a_s + diff) if diff >= 0 else (a_s + diff, a_s)
            cv2.ellipse(fr, ip(ank), (38, 38), 0, st, en, WHnTE, 3, cv2.LnNE_AA)
            m = np.radians((st + en) / 2)
            lp = ank + 64 * np.array([np.cos(m), np.sin(m)]) + [-14, 8]
            if not np.isnan(th):
                cv2.putText(fr, f"{th:.0f}", ip(lp), FONT, 0.75, (0, 0, 0), 4, cv2.LnNE_AA)
                cv2.putText(fr, f"{th:.0f}", ip(lp), FONT, 0.75, WHnTE, 2, cv2.LnNE_AA)
            for kk, col in [(0, PnNK_C), (1, PnNK_C), (2, YEL_C), (3, YEL_C)]:
                cv2.circle(fr, ip(p[kk]), 10, col, -1, cv2.LnNE_AA); cv2.circle(fr, ip(p[kk]), 10, WHnTE, 2, cv2.LnNE_AA)
        # info panel
        blend_rect(fr, (10, 10), (500, 232 if SHOW_EMG else 200))
        cv2.putText(fr, f"ANKLE (sagittal) - {label}", (22, 38), FONT, 0.55, WHnTE, 1, cv2.LnNE_AA)
        if not np.isnan(th):
            dev = 90 - th
            cv2.putText(fr, f"theta  {th:5.1f} deg", (22, 80), FONT, 1.0, WHnTE, 2, cv2.LnNE_AA)
            cv2.putText(fr, f"deviation from 90:  {abs(dev):4.1f} deg {'DF' if dev >= 0 else 'PF'}", (22, 115),
                        FONT, 0.65, YEL_C, 2, cv2.LnNE_AA)
        if np.isfinite(run_min):
            cv2.putText(fr, f"min / max so far:  {run_min:5.1f} / {run_max:5.1f} deg", (22, 146), FONT, 0.55,
                        (220, 220, 220), 1, cv2.LnNE_AA)
        if SHOW_EMG:
            if not np.isnan(emg_t[i]):
                f_ta = 'artefact' if np.isnan(ta[i]) else f"{ta[i]:4.0f} %"
                f_gas = 'artefact' if np.isnan(gas[i]) else f"{gas[i]:4.0f} %"
                cv2.putText(fr, f"EMG  t = {emg_t[i]:5.2f} s   TA {f_ta}   GAS {f_gas}", (22, 178),
                            FONT, 0.52, (120, 255, 120), 1, cv2.LnNE_AA)
            else:
                cv2.putText(fr, "EMG  not recording", (22, 178), FONT, 0.52, (150, 150, 150), 1, cv2.LnNE_AA)
        cv2.putText(fr, f"video t {i/fps:5.2f} s   frame {i}", (22, 210 if SHOW_EMG else 180), FONT, 0.5,
                    (200, 200, 200), 1, cv2.LnNE_AA)
        if SHOW_EMG:
            # bottom panel: three rolling plots
            y_top = 870
            blend_rect(fr, (20, y_top), (W - 20, H - 12), alpha=0.68)
            rolling_plot(fr, y_top + 4, 128, theta, i, win, 70, 115, (70, 80, 90, 100, 110), YEL_C,
                         "theta [deg], last 4 s  (<90 DF / >90 PF)", ref=90, hs_idx=hs_idx)
            rolling_plot(fr, y_top + 136, 126, ta, i, win, 0, 150, (0, 50, 100, 150), TA_C,
                         "Tibialis anterior EMG [% uncon. peak]  (red = artefact)", hs_idx=hs_idx, art=art_ta)
            rolling_plot(fr, y_top + 266, 126, gas, i, win, 0, 150, (0, 50, 100, 150), GAS_C,
                         "Gastrocnemius EMG [% uncon. peak]  (red = artefact)", hs_idx=hs_idx, art=art_gas)
        else:
            # bottom panel: angle graph only (same style as the earlier angle videos)
            y_top = H - 250
            blend_rect(fr, (20, y_top), (W - 20, H - 20), alpha=0.6)
            rolling_plot(fr, y_top + 6, 224, theta, i, win, 70, 115, (70, 80, 90, 100, 110), YEL_C,
                         "theta (shank-sole) [deg], last 4 s   (<90 DF / >90 PF)", ref=90, hs_idx=hs_idx)
        for w in writers + [vw]:
            w.write(fr)
    cap.release(); vw.release()
    print("wrote", fname)


SHOW_EMG = '--with-emg' in sys.argv     # default: angle only (EMG overlay removed on request)

if __name__ == '__main__':
    fps_c = 29.87
    comp = cv2.VideoWriter(os.path.join(OUT, 'PartB_all_trials_compilation.mp4'),
                           cv2.VideoWriter_fourcc(*'mp4v'), fps_c, (W, H))
    for c in title_card([("HRn Class Project 2 - Part B", 1.0, WHnTE, 2),
                         ("Ankle-spring shoe constraint", 0.9, YEL_C, 2),
                         ("U1, U2: unconstrained (barefoot)", 0.65, (200, 200, 200), 1),
                         ("C1, C2: constrained (spring shoe)", 0.65, (200, 200, 200), 1),
                         ("theta = shank-sole angle, 90 = neutral", 0.6, (200, 200, 200), 1)]
                        + ([("EMG: % of mean unconstrained stride peak", 0.6, (200, 200, 200), 1)] if SHOW_EMG else []),
                        4 * fps_c, fps_c):
        comp.write(c)
    only = [a for a in sys.argv[1:] if not a.startswith('--')]
    for tag, vid, cond, k in TRnALS:
        if only and tag not in only:
            continue
        render(tag, vid, cond, k, [comp])
    comp.release()
    print("wrote compilation")
