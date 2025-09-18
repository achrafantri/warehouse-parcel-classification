# 🧪 Script de validation YOLOv8-OBB sur le jeu de données de test
# Évalue les performances du modèle sur l'ensemble de données de validation/test

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
from ultralytics import YOLO

def main():
    print("📌 Démarrage de la validation du modèle YOLOv8-OBB")

    try:
        # 📂 Chemins des fichiers
        # Le chemin de votre modèle entraîné
        model_path = r'C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\runs\train\yolov8n-obb-warehouse\weights\best.pt'
        
        # Le chemin de votre fichier de configuration de données (data.yaml)
        # Ce fichier contient les chemins vers les ensembles de données de validation et de test
        data_path = r'C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\wpd\data.yaml'

        if not os.path.exists(model_path):
            print(f"❌ Modèle entraîné introuvable: {model_path}")
            return
            
        if not os.path.exists(data_path):
            print(f"❌ Fichier de configuration data.yaml introuvable: {data_path}")
            return
            
        print(f"📦 Chargement du modèle depuis: {model_path}")
        model = YOLO(model_path)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🧠 Device utilisé: {device}")
        
        # 🚀 Lancement de la validation
        # La méthode 'val()' évalue le modèle sur le jeu de données spécifié dans data.yaml.
        # Pour OBB, il utilise les métriques spécifiques (mAP, Precision, Recall)
        print(f"🚀 Lancement de la validation sur le jeu de données spécifié dans {data_path}...")

        results = model.val(
            data=data_path,
            imgsz=640,
            conf=0.25,        # Seuil de confiance
            split="test",     # Utiliser le jeu de données de test défini dans data.yaml
            project='runs/predict',
            name='yolov8n-obb-test',
            save_json=True    # Sauvegarder les résultats dans un fichier JSON
        )

        print("✅ Validation terminée avec succès !")
        
        # Afficher les métriques clés des résultats
        if results:
            print("\n📊 Métriques de performance (OBB) :")
            print(f"🎯 mAP50: {results.obb.map50:.3f}")
            print(f"🎯 mAP50-95: {results.obb.map:.3f}")
            print(f"📈 Précision moyenne: {results.obb.precision:.3f}")
            print(f"📈 Rappel moyen: {results.obb.recall:.3f}")
            print("\n📁 Les résultats détaillés se trouvent dans le dossier 'runs/predict/yolov8n-obb-test'")
            
    except Exception as e:
        print(f"❌ Erreur pendant la validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
