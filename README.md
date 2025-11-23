# Bare_hands
Glove vs Bare Hand Detection
1. Project Overview
This project implements an end-to-end object detection pipeline to classify:
->gloved_hand
->bare_hand

The system is designed for factory safety compliance, where it can be applied to camera streams or image snapshots to verify PPE adherence.

The detection pipeline uses YOLOv8 for training and inference, and outputs:

->Annotated images
->JSON logs containing bounding boxes, class labels, and confidence scores

2. Dataset

Name: Glove Hand and Bare Hand Detection
Source: Roboflow Universe (custom dataset forked and exported in YOLO format)
link : https://universe.roboflow.com/shashanks/glove-hand-and-bare-hand-zwvif-rvxh4 
The dataset contained three splits:
Glove Hand and Bare Hand Computer Vision Model
total: 671
Train: 415 images
Valid: 160 images
Test: 96 images

Class definitions
1: glove_hand
0: bare_hand

3. Model Used

The object detection model used in this project is:

YOLOv8n (Ultralytics)

Fine-tuned on the Roboflow dataset for gloved_hand and bare_hand detection

Final trained model saved at:
/Users/shashankgoyal/runs/detect/glove_train/weights/best.pt
which is offcourse outside the current working directory 

4. Preprocessing & Training
Preprocessing

Dataset downloaded in YOLO format

Verified dataset integrity using quick_view.py (visual label inspection)

Confirmed bounding box alignment and class mappings

Added .gitignore to avoid uploading large/irrelevant folders (venv/, runs/, etc.)

Training Details

Base model: yolov8n.pt

Epochs: 30 (or the number you used)

Image size: 640x640

Batch size: 4 (CPU) or 8 (GPU)

Device: CPU (Apple M2)

Augmentation: Default Ultralytics augmentations

Training command used:
yolo task=detect mode=train \
  model=yolov8n.pt \
  data="Glove Hand and Bare Hand.v1i.yolov8/data.yaml" \
  epochs=30 \
  imgsz=640 \
  batch=4 \
  device=cpu \
  name=glove_train

5. What Worked & What Didn’t
What Worked

Fine-tuning YOLOv8 achieved correct detection of gloved vs bare hands

After training, detection script generated proper bounding boxes + JSON logs

Roboflow dataset was clean and easy to integrate

The trained model performed well even on unseen images sampled(sample_images) from the dataset
(output) conatins more then 3-5 annotated images 
with complete box and category as glove_hands or bare_hands 

-> What Didn’t Work Initially 
Paths containing spaces caused some issues; resolved by quoting paths

Incorrect assumption of training output path (runs/detect/glove_train) — required locating actual weight file

6. How to Run the Detection Script
detection_script = main.py
python3 main.py \
  --input sample_images \
  --output output \
  --logs logs \
  --model model/best.pt \          
  <!-- since this is not present in the current working directory i have zipped it up in this folder only it would be more clear for your reference the path to which would be runs/detect/glove_train/weights/best.pt -->
  --data-yaml data.yaml \
  --confidence 0.3 \
  --device cpu


Argument	Description
--input	- Folder containing .jpg images to run detection on
--output	- Folder to save annotated images
--logs	- Folder to save JSON detection results
--model	- Path to your trained YOLOv8 weights (best.pt)
--data-yaml	- Path to your dataset’s data.yaml file
--confidence	- Detection confidence threshold (0–1)
--device	- cpu or GPU index (0)

7. Output Format

Each JSON file looks like:
{
  "filename": "image1.jpg",
  "detections": [
    {
      "label": "gloved_hand",
      "confidence": 0.92,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}

Annotated images contain bounding boxes around detected hands.

exact validation metrics (mAP, precision, recall) i am adding the metrics for clearer understanding for your references. 