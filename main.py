# this is the detection_script.py for Convinience i have named it as main.py  
import argparse, json, yaml
from pathlib import Path
from ultralytics import YOLO
import cv2
from tqdm import tqdm

# loading the date from data_yaml
def load_names(data_yaml):
    d = yaml.safe_load(open(data_yaml))
    return d.get('names', ['bare_hand', 'glove_hand'])

# Custom code for date detection and segmentation 
def draw_boxes(img, dets, names):
    for d in dets:
        x1,y1,x2,y2 = map(int, d['bbox'])
        label = f"{d['label']} {d['confidence']:.2f}"
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, label, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return img

def main(args):
    model = YOLO(args.model)
    names = load_names(args.data_yaml)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.logs); log_dir.mkdir(parents=True, exist_ok=True)

# in case there is no image found 
    imgs = sorted(Path(args.input).glob('*.jpg'))
    if len(imgs)==0:
        print("No .jpg images found in", args.input); return

    for p in tqdm(imgs):
        img = cv2.imread(str(p))
        results = model.predict(source=str(p), conf=args.confidence, imgsz=args.imgsz, device=args.device, verbose=False)
        r = results[0]
        detections=[]
        try:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls   = r.boxes.cls.cpu().numpy().astype(int)
            for i,box in enumerate(boxes):
                if confs[i] < args.confidence: continue
                x1,y1,x2,y2 = map(float, box)
                detections.append({
                    "label": names[int(cls[i])],
                    "confidence": float(round(float(confs[i]), 4)),
                    "bbox": [round(x1,1), round(y1,1), round(x2,1), round(y2,1)]
                })
        except Exception:
            pass

        out_img = draw_boxes(img.copy(), detections, names)
        cv2.imwrite(str(out_dir / p.name), out_img)
        with open(log_dir / (p.stem + '.json'), 'w') as f:
            json.dump({"filename": p.name, "detections": detections}, f, indent=2)

    print("Finished. Annotated images ->", out_dir, "Logs ->", log_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='folder with .jpg images')
    ap.add_argument('--output', default='output', help='annotated images folder')
    ap.add_argument('--logs', default='logs', help='per-image json logs')
    ap.add_argument('--model', default='yolov8n.pt', help='weights path')
    ap.add_argument('--data-yaml', default='Glove Hand and Bare Hand.v1i.yolov8/data.yaml', help='path to data.yaml')
    ap.add_argument('--confidence', type=float, default=0.4)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--device', default='0', help='gpu id or "cpu"')
    args = ap.parse_args()
    main(args)
