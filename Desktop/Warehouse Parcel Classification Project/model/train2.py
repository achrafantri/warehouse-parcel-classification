# 🚀 Script d'entraînement YOLOv8-Seg autonome (pas besoin de notebook)
# Utilise Ultralytics YOLOv8 pour segmentation d'objets

import os
import sys
import warnings
warnings.filterwarnings('ignore')

def main():
    print("📌 Démarrage de l'entraînement YOLOv8 Segmentation")
    print(f"Python: {sys.version}")
    print(f"Exécutable: {sys.executable}")
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"🖥️ GPU disponible: {torch.cuda.is_available()}")
        
        from ultralytics import YOLO
        print("✅ Ultralytics importé")

        # Chemins personnalisés
        data_path = r'C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\wpd\data.yaml'
        model_path = r'yolov8n-seg.pt'  # Téléchargé automatiquement si non présent
        
        if not os.path.exists(data_path):
            print(f"❌ data.yaml introuvable: {data_path}")
            return
        
        print(f"📂 Configuration trouvée: {data_path}")
        
        # Chargement du modèle segmentation
        print("📦 Chargement du modèle YOLOv8-Seg...")
        model = YOLO(model_path)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🧠 Device utilisé: {device}")

        print("🚀 Lancement de l'entraînement...")
        results = model.train(
            data=data_path,
            epochs=50,
            imgsz=640,
            batch=2,
            device=device,
            project='runs/train',
            name='yolov8n-seg-warehouse',
            exist_ok=True,
            verbose=True,
            patience=10,
            save=True,
            save_period=10,
            workers=0,
            amp=False,
            cache=False
        )

        print("✅ Entraînement terminé avec succès !")
        print("📁 Résultats sauvegardés dans: runs/train/yolov8n-seg-warehouse")

        # Affichage métriques segmentation
        if hasattr(results, 'seg'):
            print(f"🎯 mAP50 (seg): {results.seg.map50:.3f}")
            print(f"🎯 mAP50-95 (seg): {results.seg.map:.3f}")
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("Installez les dépendances avec:")
        print("pip install ultralytics torch torchvision numpy opencv-python pillow pyyaml")

    except Exception as e:
        print(f"❌ Erreur pendant l'entraînement: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
