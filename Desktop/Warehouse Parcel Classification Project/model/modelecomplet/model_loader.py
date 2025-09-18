import torch
from ultralytics import YOLO
from tkinter import messagebox

def load_models(model_path, device):
    """
    Charge le modèle YOLO et le modèle MiDaS.

    Args:
        model_path (str): Chemin vers le fichier .pt du modèle YOLO.
        device (str): Le périphérique à utiliser ("cuda" ou "cpu").

    Returns:
        tuple: (modèle YOLO, modèle MiDaS, transformations MiDaS) ou (None, None, None) en cas d'erreur.
    """
    # Chargement du modèle YOLO
    print("📦 Chargement du modèle YOLO...")
    try:
        model_yolo = YOLO(model_path)
    except Exception as e:
        messagebox.showerror("Erreur de chargement", f"Impossible de charger le modèle YOLO : {e}")
        return None, None, None

    # Chargement du modèle MiDaS
    print("🔍 Chargement du modèle MiDaS (profondeur)...")
    try:
        midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).dpt_transform
        midas.to(device).eval()
        return model_yolo, midas, midas_transforms
    except Exception as e:
        messagebox.showerror("Erreur de chargement", f"Impossible de charger MiDaS : {e}")
        return model_yolo, None, None
