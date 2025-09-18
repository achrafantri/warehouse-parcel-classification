import os
import cv2
import torch
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils import fuse_model

# --- Chemins ---
stage2_name = "yolov8n_finetune_box_fragile"
workdir = r"C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\runs\finetune"
best_weights = os.path.join(workdir, stage2_name, "weights", "best.pt")

video_path = r"C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\Sorting Amazon Packages.mp4"
output_dir = os.path.join(workdir, "inference_results_bytrack")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "tracked_output.mp4")

# --- Charger le modèle YOLOv8 ---
model = YOLO(best_weights)

# --- Initialiser ByteTrack ---
tracker = BYTETracker(track_thresh=0.2, track_buffer=30, match_thresh=0.8)

# --- Lecture vidéo ---
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Détection YOLO ---
    results = model.predict(frame_rgb, conf=0.2, imgsz=640, verbose=False)[0]

    dets = []
    if results.boxes is not None and len(results.boxes) > 0:
        for i in range(len(results.boxes)):
            box = results.boxes[i].xyxy.cpu().numpy()[0]  # [x1, y1, x2, y2]
            score = float(results.boxes[i].conf.cpu().numpy())
            dets.append([box[0], box[1], box[2], box[3], score])

        dets = torch.tensor(dets, dtype=torch.float32)

    # --- Tracking ByteTrack ---
    outputs = tracker.update(dets, [height, width], frame)

    # --- Dessin des tracks ---
    if outputs is not None:
        for output in outputs:
            x1, y1, x2, y2, track_id = output[:5]
            x1, y1, x2, y2, track_id = map(int, [x1, y1, x2, y2, track_id])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("ByteTrack", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Vidéo tracée sauvegardée dans {output_path}")
