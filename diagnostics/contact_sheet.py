"""Save a contact sheet of thumbnails (every `step` frames from `start`) with frame numbers."""
import cv2, sys
import numpy as np

path, start, step, count, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
cap = cv2.VideoCapture(path)
thumbs = []
for k in range(count):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start + k * step)
    ok, fr = cap.read()
    if not ok:
        break
    t = cv2.resize(fr, (216, 384))
    cv2.putText(t, str(start + k * step), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    thumbs.append(t)
while len(thumbs) % 6:
    thumbs.append(np.zeros_like(thumbs[0]))
rows = [np.hstack(thumbs[i:i + 6]) for i in range(0, len(thumbs), 6)]
cv2.imwrite(out, np.vstack(rows))
