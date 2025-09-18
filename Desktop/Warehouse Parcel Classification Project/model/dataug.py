import os
import cv2
import glob
import numpy as np
import random
import shutil
import yaml

# ======================
# PARAMÈTRES
# ======================
dataset_dir = r"C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\warehouse parcel detection.v11-dim.yolov8-obb"
augmented_dir = r"C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\dataset_augmented"
n_aug_per_image = 5  # nombre d'augmentations par image

# Créer les dossiers augmentés
for split in ["train","valid","test"]:
    os.makedirs(os.path.join(augmented_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(augmented_dir, split, "labels"), exist_ok=True)

# ======================
# FONCTIONS D'AUGMENTATION
# ======================
def random_flip(image, boxes):
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        if len(boxes) > 0:
            boxes[:,0] = 1 - boxes[:,0]  # flip horizontal
    if random.random() < 0.5:
        image = cv2.flip(image, 0)
        if len(boxes) > 0:
            boxes[:,1] = 1 - boxes[:,1]  # flip vertical
    return image, boxes

def random_brightness(image):
    factor = 0.7 + random.random()*0.6
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,2] = np.clip(hsv[:,:,2]*factor,0,255)
    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return image

def random_rotate(image, boxes):
    h, w = image.shape[:2]
    angle = random.uniform(-15,15)
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w,h), borderValue=(114,114,114))
    if len(boxes) > 0:
        for i in range(len(boxes)):
            x_c, y_c = boxes[i,0]*w, boxes[i,1]*h
            coords = np.dot(M[:,:2], [x_c, y_c]) + M[:,2]
            boxes[i,0], boxes[i,1] = coords[0]/w, coords[1]/h
    return image, boxes

def random_scale(image, boxes):
    h, w = image.shape[:2]
    scale = 0.9 + random.random()*0.2  # scale ±10%
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if len(boxes) > 0:
        boxes[:,0:4] *= scale
    return image, boxes

# ======================
# TRAITEMENT
# ======================
for split in ["train","valid","test"]:
    img_dir = os.path.join(dataset_dir, split, "images")
    lbl_dir = os.path.join(dataset_dir, split, "labels")
    out_img_dir = os.path.join(augmented_dir, split, "images")
    out_lbl_dir = os.path.join(augmented_dir, split, "labels")

    image_files = glob.glob(os.path.join(img_dir,"*.jpg"))
    print(f"Traitement {split}: {len(image_files)} images")

    for img_path in image_files:
        filename = os.path.basename(img_path).split(".")[0]
        label_path = os.path.join(lbl_dir, filename+".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Impossible de lire l'image : {img_path}")
            continue

        h, w = img.shape[:2]

        # Lire labels
        boxes = []
        if os.path.exists(label_path):
            with open(label_path,"r") as f:
                for line in f.readlines():
                    parts = line.strip().replace(',',' ').split()
                    if len(parts) != 5:
                        print(f"⚠️ Ligne ignorée dans {label_path}: {line.strip()}")
                        continue
                    cls, x, y, bw, bh = map(float, parts)
                    boxes.append([x, y, bw, bh, int(cls)])
            boxes = np.array(boxes)
        else:
            boxes = np.zeros((0,5))

        if len(boxes) == 0:
            print(f"⚠️ Aucun objet trouvé dans {label_path}")

        # Sauvegarder image originale et label
        shutil.copy(img_path, os.path.join(out_img_dir, filename+".jpg"))
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(out_lbl_dir, filename+".txt"))
        else:
            open(os.path.join(out_lbl_dir, filename+".txt"), "w").close()

        # Générer augmentations
        for n in range(n_aug_per_image):
            aug_img = img.copy()
            aug_boxes = boxes.copy()

            aug_img = random_brightness(aug_img)
            aug_img, aug_boxes = random_flip(aug_img, aug_boxes)
            aug_img, aug_boxes = random_rotate(aug_img, aug_boxes)
            aug_img, aug_boxes = random_scale(aug_img, aug_boxes)

            aug_filename = f"{filename}_aug{n}.jpg"
            cv2.imwrite(os.path.join(out_img_dir, aug_filename), aug_img)

            label_lines = []
            for b in aug_boxes:
                cls = int(b[4])
                x, y, bw, bh = b[0], b[1], b[2], b[3]
                label_lines.append(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")

            # Sauvegarder le fichier .txt augmenté
            label_file_path = os.path.join(out_lbl_dir, f"{filename}_aug{n}.txt")
            with open(label_file_path,"w") as f:
                f.write("\n".join(label_lines))

# ======================
# MISE À JOUR DU YAML
# ======================
yaml_path = os.path.join(dataset_dir,"data.yaml")
if os.path.exists(yaml_path):
    with open(yaml_path,"r") as f:
        data = yaml.safe_load(f)
    data_aug = data.copy()
    data_aug['train'] = os.path.join(augmented_dir,"train","images")
    data_aug['valid'] = os.path.join(augmented_dir,"valid","images")  # clé valid
    if 'test' in data_aug:
        data_aug['test'] = os.path.join(augmented_dir,"test","images")
    with open(os.path.join(augmented_dir,"data.yaml"),"w") as f:
        yaml.dump(data_aug, f)

print("✅ Data augmentation terminée. Nouveau dataset :", augmented_dir)
