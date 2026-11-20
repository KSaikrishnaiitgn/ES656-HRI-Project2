"""Save full-resolution crops around the lower leg and print HSV of clicked-style sample points."""
import cv2, sys, os
import numpy as np

path, frame_idx, x0, y0, x1, y1, out = sys.argv[1], int(sys.argv[2]), *map(int, sys.argv[3:7]), sys.argv[7]
cap = cv2.VideoCapture(path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ok, fr = cap.read()
crop = fr[y0:y1, x0:x1]
cv2.imwrite(out, crop)
hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
# report HSV stats of strongly saturated pixels, binned by hue
H, S, V = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()
m = (S > 120) & (V > 120)
hist = np.bincount(H[m] // 5, minlength=36)
print("hue bins (H/5) of saturated pixels:", {i * 5: int(c) for i, c in enumerate(hist) if c > 50})
