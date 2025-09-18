import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
from tqdm import tqdm

# === CONFIGURATION ===
sam_checkpoint = "sam_vit_b_01ec64.pth"  # chemin local du modèle SAM
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

input_images_path = "./wpd/test/images"
output_images_path = "./wpd_seg/test/images"
output_labels_path = "./wpd_seg/test/labels"

os.makedirs(output_images_path, exist_ok=True)
os.makedirs(output_labels_path, exist_ok=True)

# === LOAD SAM ===
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# === CONVERT SAM TO YOLOv8 SEG ===
def mask_to_yolo_format(mask):
    # Trouver le contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yolo_seg_lines = []
    for contour in contours:
        if len(contour) < 6:
            continue
        contour = contour.squeeze()
        normalized = contour.astype(np.float32)
        normalized[:, 0] /= mask.shape[1]  # x
        normalized[:, 1] /= mask.shape[0]  # y
        flattened = normalized.flatten()
        coords = " ".join([f"{x:.6f}" for x in flattened])
        line = f"0 {coords}"
        yolo_seg_lines.append(line)
    return yolo_seg_lines

# === TRAITEMENT ===
image_files = [f for f in os.listdir(input_images_path) if f.endswith(('.jpg', '.png'))]

for filename in tqdm(image_files):
    image_path = os.path.join(input_images_path, filename)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image_rgb)
    boxes = [[0, 0, image.shape[1], image.shape[0]]]  # full image
    transformed_boxes = predictor.transform.apply_boxes_torch(
        torch.tensor(boxes, device=device), image.shape[:2]
    )

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=True
    )

    masks_np = masks[0].cpu().numpy()

    all_lines = []
    for i in range(masks_np.shape[0]):
        mask = masks_np[i].astype(np.uint8) * 255
        lines = mask_to_yolo_format(mask)
        all_lines.extend(lines)

    # Sauvegarde des fichiers
    out_img_path = os.path.join(output_images_path, filename)
    out_lbl_path = os.path.join(output_labels_path, filename.replace('.jpg', '.txt').replace('.png', '.txt'))

    cv2.imwrite(out_img_path, image)
    with open(out_lbl_path, 'w') as f:
        for line in all_lines:
            f.write(line + "\n")
