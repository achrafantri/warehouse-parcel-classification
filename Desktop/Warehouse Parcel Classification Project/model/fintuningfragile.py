import os
from ultralytics import YOLO

# Chemins
stage1_weights = r"C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\runs\train\yolov8n-obb-warehouse\weights\best.pt"
data_stage2 = r"C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\new\data_box_fragile.yaml"

# Dossier de sortie
workdir = r"C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\runs\finetune2"

# Config fine-tuning
epochs_stage2 = 30
imgsz = 640
batch_stage2 = 8
lr_stage2 = 1e-4
stage2_name = "yolov8n_finetune_box_fragile"

def finetune():
    print(f"==> Fine-tuning du modèle (box + fragile) à partir de {stage1_weights}")
    model = YOLO(stage1_weights)
    
    model.train(
        data=data_stage2,
        epochs=epochs_stage2,
        imgsz=imgsz,
        batch=batch_stage2,
        lr0=lr_stage2,
        name=stage2_name,
        project=workdir,
        augment=False,   # éviter trop d'augmentations pendant fine-tune
        save=True,
        exist_ok=True
    )
    
    best_weights = os.path.join(workdir, stage2_name, "weights", "best.pt")
    print("✅ Fine-tuning terminé. Poids sauvegardés dans:", best_weights)

if __name__ == "__main__":
    finetune()
