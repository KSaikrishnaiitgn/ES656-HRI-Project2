"""Print video properties and save a few sample frames for checking marker colours."""
import cv2, os, sys

out_dir = sys.argv[1]
for path in sys.argv[2:]:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{os.path.basename(path)}: {w}x{h}, {fps:.2f} fps, {n} frames, {n/fps:.1f} s")
    tag = os.path.splitext(os.path.basename(path))[0].replace(' ', '')
    for frac in [0.1, 0.3, 0.5, 0.7]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac))
        ok, fr = cap.read()
        if ok:
            scale = 960 / max(fr.shape[:2])
            small = cv2.resize(fr, None, fx=scale, fy=scale)
            cv2.imwrite(os.path.join(out_dir, f"{tag}_f{int(frac*100)}.jpg"), small)
    cap.release()
