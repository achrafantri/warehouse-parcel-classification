import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import argparse

# ======================
# Constantes et fonctions
# ======================

# Paramètres simplifiés de la caméra pour la projection 3D
FOCAL_LENGTH_PIXELS = 1000  
CAMERA_MATRIX = np.array([
    [FOCAL_LENGTH_PIXELS, 0, 320],
    [0, FOCAL_LENGTH_PIXELS, 240],
    [0, 0, 1]
], dtype=np.float32)

def load_midas(device):
    """
    Charge le modèle de profondeur MiDaS.
    """
    print("🔍 Chargement du modèle MiDaS (profondeur)...")
    try:
        midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).dpt_transform
        midas.to(device).eval()
        return midas, midas_transforms
    except Exception as e:
        print(f"Erreur de chargement de MiDaS : {e}")
        return None, None

def get_depth_map(frame, midas, midas_transform, device):
    """
    Génère la carte de profondeur d'une image en utilisant MiDaS.
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = midas_transform(img_rgb).to(device)
    if input_batch.dim() == 5:
        input_batch = input_batch.squeeze(1)

    with torch.no_grad():
        depth = midas(input_batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    return depth.cpu().numpy()

def project_3d_box(image, x_center, y_center, rotation, mean_depth_cm, color):
    """
    Projete et dessine une boîte 3D simplifiée sur l'image en utilisant le centre et la rotation de la boîte de détection.
    """
    # Dimensions du cuboïde 3D à projeter (arbitraires pour l'affichage)
    longueur_m, largeur_m, hauteur_m = 0.4, 0.3, 0.2
    
    # Définir les 8 points du cube 3D centrés à l'origine
    points_3d_cube = np.array([
        [-largeur_m / 2, -hauteur_m / 2, longueur_m / 2],
        [largeur_m / 2, -hauteur_m / 2, longueur_m / 2],
        [largeur_m / 2, hauteur_m / 2, longueur_m / 2],
        [-largeur_m / 2, hauteur_m / 2, longueur_m / 2],
        [-largeur_m / 2, -hauteur_m / 2, -longueur_m / 2],
        [largeur_m / 2, -hauteur_m / 2, -longueur_m / 2],
        [largeur_m / 2, hauteur_m / 2, -longueur_m / 2],
        [-largeur_m / 2, hauteur_m / 2, -longueur_m / 2]
    ], dtype=np.float32)

    # Calculer la position 3D (tvec) à partir des coordonnées 2D et de la profondeur
    fx = CAMERA_MATRIX[0, 0]
    fy = CAMERA_MATRIX[1, 1]
    cx = CAMERA_MATRIX[0, 2]
    cy = CAMERA_MATRIX[1, 2]
    
    tx = (x_center - cx) * (mean_depth_cm / 100) / fx
    ty = (y_center - cy) * (mean_depth_cm / 100) / fy
    tz = mean_depth_cm / 100

    # Utiliser la rotation OBB pour l'orientation de la boîte 3D
    rvec = np.array([0, 0, rotation], dtype=np.float32).reshape(3, 1)
    tvec = np.array([tx, ty, tz], dtype=np.float32).reshape(3, 1)

    # Projeter les points 3D sur l'image
    points_2d, _ = cv2.projectPoints(points_3d_cube, rvec, tvec, CAMERA_MATRIX, None)
    points_2d = np.int32(points_2d).reshape(-1, 2)
    
    # Dessiner les lignes de la boîte 3D
    cv2.polylines(image, [points_2d[0:4]], True, color, 2)
    cv2.polylines(image, [points_2d[4:8]], True, color, 2)
    cv2.line(image, tuple(points_2d[0]), tuple(points_2d[4]), color, 2)
    cv2.line(image, tuple(points_2d[1]), tuple(points_2d[5]), color, 2)
    cv2.line(image, tuple(points_2d[2]), tuple(points_2d[6]), color, 2)
    cv2.line(image, tuple(points_2d[3]), tuple(points_2d[7]), color, 2)


def main():
    parser = argparse.ArgumentParser(description="Détection 3D des colis sans calcul de volume.")
    parser.add_argument("--image_path", type=str, required=True, help="Chemin vers le fichier image à analyser.")
    args = parser.parse_args()
    
    # Chemins
    model_yolo_path = r"C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\runs\finetune\yolov8n_finetune_box_fragile\weights\best.pt"
    output_image_path = "output_image_3d.png"

    if not os.path.exists(model_yolo_path):
        print(f"Erreur : Modèle YOLO introuvable à {model_yolo_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Utilisation du périphérique :", device)

    # Chargement des modèles
    print("📦 Chargement du modèle YOLO...")
    model_yolo = YOLO(model_yolo_path)
    model_midas, transform_midas = load_midas(device)
    if not model_midas:
        return

    # Traitement de l'image
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Erreur : Impossible de lire l'image à {args.image_path}")
        return

    results = model_yolo.predict(image, imgsz=640, conf=0.3, device=device, verbose=False)
    processed_image = image.copy()
    depth_map = get_depth_map(image, model_midas, transform_midas, device)

    if results and len(results) > 0:
        for result in results:
            if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
                obb_boxes = result.obb.xywhr.cpu().numpy()
                class_ids = result.obb.cls.cpu().numpy()
                names = result.names

                for i in range(len(obb_boxes)):
                    x_center, y_center, _, _, rotation = obb_boxes[i]
                    class_id = int(class_ids[i])
                    class_name = names[class_id]
                    
                    # Profondeur à partir de MiDaS
                    depth_crop = depth_map[int(y_center)-20:int(y_center)+20, int(x_center)-20:int(x_center)+20]
                    mean_depth_val = np.median(depth_crop) if depth_crop.size > 0 else 0
                    mean_depth_cm = mean_depth_val * 100 
                    
                    est_fragile = (class_name == "fragile")
                    color = (0, 0, 255) if est_fragile else (0, 255, 0)
                    
                    # Dessiner la boîte 3D
                    project_3d_box(processed_image, int(x_center), int(y_center), rotation, mean_depth_cm, color)

    # Sauvegarde de l'image de sortie
    cv2.imwrite(output_image_path, processed_image)
    print(f"✅ Image de sortie sauvegardée sous : {output_image_path}")

if __name__ == "__main__":
    main()
