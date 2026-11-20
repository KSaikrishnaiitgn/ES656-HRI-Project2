"""
HRI Class Project 2 - Part A: compilation of the experiment videos with title cards
(same card style as the Part B compilation in s8_render_annotated_videos.py).

Input : PartA/experimental videos/*.mp4  (1920x1080; configuration 1 was not filmed: video V2_config1 shows configuration 2 and V2_config_2 configuration 3, matched by EMG time stamps)
Output: 03_results/videos/PartA_all_trials_compilation.mp4  (1280x720, 30 fps, no audio)

Order follows the experiment protocol: (a) movement speed, (b) limb configuration,
(c) stable loading, (d) unstable loading. Run from 02_code:  python partA_compile_videos.py
"""
import os
import cv2
import numpy as np
import config_paths as P

VID_DIR = os.path.join(P.ROOT, 'PartA', 'experimental videos')
OUT = P.RES_VID; os.makedirs(OUT, exist_ok=True)
W, H, FPS = 1280, 720, 30.0
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, YEL, GREY = (255, 255, 255), (0, 220, 255), (200, 200, 200)

TRIALS = [  # tag, file, condition title, condition group
    ('A1', 'V1_slow.mp4', 'Movement speed - slow', '(a) Varying movement speed'),
    ('A2', 'V1_normal.mp4', 'Movement speed - normal', '(a) Varying movement speed'),
    ('A3', 'V1_fast.mp4', 'Movement speed - fast', '(a) Varying movement speed'),
    ('A4', 'V2_config1.mp4', 'Limb configuration 2', '(d) Different limb configurations'),
    ('A5', 'V2_config_2.mp4', 'Limb configuration 3', '(d) Different limb configurations'),
    ('A6', 'V3_stable_loading.mp4', 'Stable loading', '(b) Stable limb loading'),
    ('A7', 'V4_Unstable_loading.mp4', 'Unstable loading', '(c) Unstable limb loading'),
]


def title_card(lines, seconds):
    img = np.full((H, W, 3), 25, np.uint8)
    total = sum(int(50 * sc + 22) for _, sc, _, _ in lines)
    y = (H - total) // 2 + 40
    for txt, sc, col, th in lines:
        (tw, _), _ = cv2.getTextSize(txt, FONT, sc, th)
        cv2.putText(img, txt, ((W - tw) // 2, y), FONT, sc, col, th, cv2.LINE_AA)
        y += int(50 * sc + 22)
    return [img] * int(seconds * FPS)


def corner_label(fr, txt):
    (tw, th), _ = cv2.getTextSize(txt, FONT, 0.7, 2)
    overlay = fr.copy()
    cv2.rectangle(overlay, (12, 12), (12 + tw + 20, 12 + th + 20), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, fr, 0.45, 0, fr)
    cv2.putText(fr, txt, (22, 12 + th + 9), FONT, 0.7, WHITE, 2, cv2.LINE_AA)


def main():
    path = os.path.join(OUT, 'PartA_all_trials_compilation.mp4')
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    frames = title_card([("HRI Class Project 2 - Part A", 1.4, WHITE, 3),
                         ("Agonist and antagonist activation", 1.0, YEL, 2),
                         ("EMG: tibialis anterior (TA) and gastrocnemius (GAS), right leg", 0.7, GREY, 1),
                         ("Conditions: movement speed, limb configuration,", 0.7, GREY, 1),
                         ("stable loading, unstable loading", 0.7, GREY, 1),
                         ("Recorded 15 November 2026", 0.7, GREY, 1)], 4)
    for f in frames:
        vw.write(f)
    total = len(frames)
    for tag, fname, title, group in TRIALS:
        cap = cv2.VideoCapture(os.path.join(VID_DIR, fname))
        if not cap.isOpened():
            print('missing', fname); continue
        for f in title_card([(f"Trial {tag}", 1.8, WHITE, 3), (title, 1.1, YEL, 2), (group, 0.75, GREY, 1)], 2):
            vw.write(f)
        n = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
            corner_label(fr, f"{tag} | {title}")
            vw.write(fr); n += 1
        cap.release()
        total += n + int(2 * FPS)
        print(f'{tag} {fname}: {n} frames ({n / FPS:.1f} s)')
    vw.release()
    print(f'wrote {path} ({total / FPS:.1f} s)')


if __name__ == '__main__':
    main()
