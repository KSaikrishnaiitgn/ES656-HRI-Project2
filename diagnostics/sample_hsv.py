"""Print the median HSV of a 9x9 patch at given (x,y) full-resolution points of one frame."""
import cv2, sys
import numpy as np

path, idx = sys.argv[1], int(sys.argv[2])
cap = cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
ok, fr = cap.read()
hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
for pt in sys.argv[3:]:
    x, y = map(int, pt.split(','))
    patch = hsv[y-4:y+5, x-4:x+5].reshape(-1, 3)
    print(pt, np.median(patch, axis=0))
