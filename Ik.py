import numpy as np
import cv2 as cv
import argparse

# ============ PARAMÈTRES À MODIFIER POUR TESTER ============
MAX_CORNERS = 100     # nombre de points suivis (essayez 300)
QUALITY     = 0.3     # qualité minimale des coins (essayez 0.01)
MIN_DIST    = 7       # distance minimale entre deux points
BLOCK_SIZE  = 7
WIN_SIZE    = (15, 15)  # taille de la fenêtre (essayez (5, 5) ou (31, 31))
MAX_LEVEL   = 2       # niveaux de la pyramide (essayez 0)
# ===========================================================

parser = argparse.ArgumentParser(description='Lucas-Kanade Optical Flow')
parser.add_argument('video', type=str, nargs='?',
                    default='slow_traffic_small.avi',
                    help='path to video file')
args = parser.parse_args()

cap = cv.VideoCapture(args.video)
if not cap.isOpened():
    print('Unable to open video:', args.video)
    raise SystemExit

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=MAX_CORNERS,
                      qualityLevel=QUALITY,
                      minDistance=MIN_DIST,
                      blockSize=BLOCK_SIZE)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=WIN_SIZE,
                 maxLevel=MAX_LEVEL,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors (un par point)
color = np.random.randint(0, 255, (MAX_CORNERS, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
if not ret:
    print('Cannot read the first frame')
    raise SystemExit
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
if p0 is None:
    print('No corners found: lower QUALITY')
    raise SystemExit
print('Points detected:', len(p0))

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is None:
        print('All points lost')
        break
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    if len(good_new) == 0:
        print('All points lost')
        break

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)

    # affiche le nombre de points encore suivis
    cv.putText(img, f'Points: {len(good_new)}', (10, 25),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:  # Échap pour quitter
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv.destroyAllWindows()