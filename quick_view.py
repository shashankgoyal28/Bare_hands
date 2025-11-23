# this code was used to visually verify that the dataset’s labels are correct before starting training.
import cv2
from pathlib import Path

DATAROOT = Path('.')
names = ['bare_hand', 'glove_hand'] 

img_paths = list((DATAROOT/'train'/'images').glob('*.jpg'))[:8]
for p in img_paths:
    img = cv2.imread(str(p))
    h,w = img.shape[:2]
    lbl = DATAROOT/'train'/'labels'/ (p.stem + '.txt')
    if lbl.exists():
        for line in open(lbl):
            cls, xc, yc, ww, hh = map(float, line.split())
            x1 = int((xc - ww/2) * w); y1 = int((yc - hh/2) * h)
            x2 = int((xc + ww/2) * w); y2 = int((yc + hh/2) * h)
            color = (0,255,0) if int(cls)==0 else (0,0,255)
            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
            cv2.putText(img, names[int(cls)], (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,1)
    cv2.imshow('img', img); cv2.waitKey(0)
cv2.destroyAllWindows()
