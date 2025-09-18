import cv2
import torch
import numpy as np
import csv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ======================
# PARAMETRES
# ======================
video_input = r"C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\Sorting Amazon Packages.mp4"
video_output = "output_tracking.mp4"
csv_output = "tracking_3d_video.csv"

box_model_path = r"C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\runs\train\yolov8n-obb-warehouse\weights\best.pt"
fragile_model_path = r"C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\runs\finetune\yolov8n_finetune_box_fragile\weights\best.pt"

# Calibration approximative : 200 px ~ 40 cm
scale_px = 200
scale_cm = 40

# ======================
# CHARGEMENT DES MODELES
# ======================
print("📦 Chargement des modèles YOLO...")
model_box = YOLO(box_model_path)
model_fragile = YOLO(fragile_model_path)

print("🔍 Chargement du modèle MiDaS (profondeur)...")
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).dpt_transform
midas.eval()

print("🛰️ Initialisation de DeepSORT...")
tracker = DeepSort(max_age=30)

# ======================
# FONCTION PROFONDEUR
# ======================
def get_depth_map(frame):
    if frame is None or frame.size == 0:
        raise ValueError("Frame vide")
    if frame.shape[2] == 3:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = np.stack([frame]*3, axis=-1)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    input_batch = midas_transforms(img_rgb).unsqueeze(0)
    with torch.no_grad():
        depth = midas(input_batch)
        if depth.ndim == 3:
            depth = depth.unsqueeze(0)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
    depth_np = depth.cpu().numpy()
    return depth_np

# ======================
# VIDEO
# ======================
cap = cv2.VideoCapture(video_input)
if not cap.isOpened():
    raise RuntimeError(f"❌ Impossible d'ouvrir la vidéo {video_input}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(video_output, fourcc, fps, (width, height))

frame_id = 0

csv_file = open(csv_output, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["id","frame","x","y","z","L","W","H","fragile"])

print("🚀 Début du tracking 3D sur vidéo...")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    frame_id += 1

    # 1. Détection colis
    results_box = model_box(frame, verbose=False)
    dets = []

    # 2. Profondeur avec protection
    try:
        depth_map = get_depth_map(frame)
    except Exception as e:
        print(f"⚠️ Frame {frame_id} profondeur erreur: {e}")
        depth_map = None

    for box in results_box[0].boxes.xyxy:
        x1,y1,x2,y2 = map(int, box.tolist())
        conf = float(box[4])
        cls = int(box[5])

        # Vérifier fragile
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        results_fragile = model_fragile.predict(crop, imgsz=128, conf=0.25, verbose=False)
        fragile_flag = 1 if len(results_fragile[0].boxes) > 0 else 0

        # Profondeur moyenne
        mean_depth = 0
        if depth_map is not None:
            depth_crop = depth_map[y1:y2, x1:x2]
            mean_depth = np.median(depth_crop)

        # Dimensions approximatives
        pixel_h, pixel_w = y2 - y1, x2 - x1
        real_h = (pixel_h / scale_px) * scale_cm
        real_w = (pixel_w / scale_px) * scale_cm
        real_l = real_w

        dets.append((
            [x1,y1,x2-x1,y2-y1],
            conf,
            cls,
            {"depth":mean_depth,"fragile":fragile_flag,"dims":(real_l,real_w,real_h)}
        ))

    # 3. Tracking DeepSORT
    tracks = tracker.update_tracks(dets, frame=frame)

    # Dessin et CSV
    for t in tracks:
        if not t.is_confirmed():
            continue
        track_id = t.track_id
        x1,y1,x2,y2 = map(int, t.to_ltrb())
        data = t.get_det_supplement()
        if data is not None:
            z = data["depth"]
            L,W,H = data["dims"]
            fragile = data["fragile"]
            writer.writerow([track_id, frame_id, x1, y1, z, L, W, H, fragile])
            color = (0,0,255) if fragile else (0,255,0)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"ID {track_id} F:{fragile}",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    # Assurer type correct pour vidéo
    frame_to_write = frame.copy()
    if frame_to_write.dtype != np.uint8:
        frame_to_write = (frame_to_write*255).astype(np.uint8)
    if len(frame_to_write.shape)==2:
        frame_to_write = cv2.cvtColor(frame_to_write, cv2.COLOR_GRAY2BGR)
    out_video.write(frame_to_write)

cap.release()
out_video.release()
csv_file.close()
print(f"✅ Tracking terminé. CSV: {csv_output}, Vidéo: {video_output}")
