import os
from ultralytics import YOLO

# Poids fine-tunés (stage 2)
stage2_name = "yolov8n_finetune_box_fragile"
workdir = r"C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\runs\finetune"
best_weights = os.path.join(workdir, stage2_name, "weights", "best.pt")

# Dossier contenant les images de test
test_dir = r"C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\testf.jpg"

def test_model():
    print(f"==> Chargement du modèle fine-tuné depuis {best_weights}")
    model = YOLO(best_weights)

    # Validation (si ton yaml de dataset contient une clé "val")
    
    results = model.val()
    print("📊 Résultats validation :", results)

    # Inférence sur le dossier test
    print(f"\n==> Inférence sur le dossier {test_dir}")
    preds = model.predict(
        source=test_dir,
        save=True,
        project=workdir,
        name="inference_results2",
        exist_ok=True
    )

    print("✅ Résultats enregistrés dans :",
          os.path.join(workdir, "inference_results"))

if __name__ == "__main__":
    test_model()
