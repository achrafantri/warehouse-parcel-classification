# 🚀 Script d'entraînement YOLOv8 autonome (sans notebook)
# Ce script fonctionne directement avec Python sans environnement virtuel

import os
import sys
import warnings
warnings.filterwarnings('ignore')

def main():
    print(" Démarrage de l'entraînement YOLOv8 OBB")
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    
    try:
        # Import des modules
        import numpy as np
        print(f" NumPy {np.__version__}")
        
        import torch
        print(f" PyTorch {torch.__version__}")
        print(f"GPU disponible: {torch.cuda.is_available()}")
        
        from ultralytics import YOLO
        print(" Ultralytics importé")
        
        # Configuration des chemins
        data_path = r'C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\warehouse parcel detection.v11-dim.yolov8-obb\data.yaml'
        model_path = r'C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\yolo8n-obb.pt'
        
        # Vérification des fichiers
        if not os.path.exists(data_path):
            print(f" Fichier data.yaml non trouvé: {data_path}")
            return
        
        if not os.path.exists(model_path):
            print(f" Modèle non trouvé: {model_path}")
            print("Téléchargement du modèle...")
            model_path = 'yolov8n-obb.pt'  # Ultralytics le téléchargera automatiquement
        
        print(f" Configuration trouvée: {data_path}")
        
        # Chargement du modèle
        print("📦 Chargement du modèle YOLOv8-OBB...")
        model = YOLO(model_path)
        
        # Configuration d'entraînement
        print(" Démarrage de l'entraînement...")
        
        # Utiliser un device approprié
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device utilisé: {device}")
        
        # Entraînement avec paramètres optimisés pour éviter les conflits
        results = model.train(
            data=data_path,
            epochs=50 ,         # Nombre d'epochs complet
            imgsz=640,           # Taille d'image standard
            batch=2,             # Batch petit pour éviter les problèmes
            device=device,
            project='runs/train2',
            exist_ok=True,
            verbose=True,
            patience=10,         # Early stopping
            save=True,
            save_period=10,      # Sauvegarde tous les 10 epochs
            workers=0,           # DÉSACTIVER les workers pour éviter multiprocessing
            amp=False,           # Désactiver AMP pour éviter les problèmes
            cache=False          # Désactiver cache pour éviter les problèmes
        )
        
        print(" Entraînement terminé avec succès !")
        print(f"Résultats sauvegardés dans: runs/train/yolov8n-obb-warehouse2")
        
        # Afficher quelques métriques
        if hasattr(results, 'box'):
            print(f"mAP50: {results.box.map50:.3f}")
            print(f"mAP50-95: {results.box.map:.3f}")
        
    except ImportError as e:
        print(f" Erreur d'import: {e}")
        print(" Essayez d'installer les dépendances avec:")
        print("pip install ultralytics torch torchvision numpy opencv-python pillow pyyaml")
        
    except Exception as e:
        print(f" Erreur pendant l'entraînement: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
