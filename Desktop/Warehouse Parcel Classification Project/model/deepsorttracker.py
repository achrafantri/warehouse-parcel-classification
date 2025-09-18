import os
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch

# --- Chemins ---
stage2_name = "yolov8n_finetune_box_fragile"
workdir = r"C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\runs\finetune"
best_weights = os.path.join(workdir, stage2_name, "weights", "best.pt")

video_path = r"C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\Parcel sorter _ sorting machine.mp4"
output_dir = os.path.join(workdir, "inference_results_deepsort")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "tracked_output.mp4")

# --- Charger le modèle YOLOv8 ---
print(f"==> Chargement du modèle fine-tuné depuis {best_weights}")
model = YOLO(best_weights)

# --- Initialiser DeepSORT ---
tracker = DeepSort(max_age=30)  # max_age = nombre de frames avant de supprimer un track

# --- Lecture vidéo ---
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # --- Convertir en RGB pour YOLO ---
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Détection YOLO ---
    results = model.predict(frame_rgb, conf=0.2, imgsz=640, verbose=False)[0]

    # --- Préparer les détections pour DeepSORT ---
    detections = []
    if results.boxes is not None and len(results.boxes) > 0:
        for i in range(len(results.boxes)):
            box = results.boxes[i].xyxy.cpu().numpy()  # [[x1, y1, x2, y2]]
            x1, y1, x2, y2 = box[0]
            score = float(results.boxes[i].conf.cpu().numpy())
            cls = int(results.boxes[i].cls.cpu().numpy())
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], score, cls))

    print(f"Frame {frame_count}: {len(detections)} détections")  # Debug

    # --- Tracking DeepSORT ---
    tracks = tracker.update_tracks(detections, frame=frame)

    # --- Dessin des tracks ---
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_id = track.track_id

        # Chercher la classe correspondante
        cls_name = ""
        for det in detections:
            dx, dy, dw, dh = det[0]
            if abs(dx - x1) < 5 and abs(dy - y1) < 5:
                cls_name = str(det[2])
                break

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{track_id} Class:{cls_name}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- Sauvegarde et affichage ---
    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Vidéo tracée sauvegardée dans {output_path}")
