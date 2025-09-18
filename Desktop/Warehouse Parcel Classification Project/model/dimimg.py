import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from tkinter import Tk, Label, messagebox
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk


# ======================
# Charger MiDaS
# ======================
def load_midas(device):
    print("🔍 Chargement du modèle MiDaS (profondeur)...")
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).dpt_transform
    midas.to(device).eval()
    return midas, midas_transforms


def get_depth_map(frame, midas, midas_transform, device):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = midas_transform(img_rgb).to(device)

    # Corriger la dimension en trop
    if input_batch.dim() == 5:  # (B, T, C, H, W)
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


# ======================
# Détection YOLO + MiDaS
# ======================
def run_detection(image_path, model, label_widget, midas, midas_transform, device):
    try:
        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror("Erreur", "Impossible de lire l'image.")
            return

        results = model.predict(image, imgsz=640, conf=0.3, device=device, verbose=False)
        processed_image = image.copy()

        depth_map = get_depth_map(image, midas, midas_transform, device)

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                names = result.names

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    class_id = int(class_ids[i])
                    confidence = confidences[i]

                    # Profondeur moyenne dans la boîte
                    depth_crop = depth_map[y1:y2, x1:x2]
                    mean_depth = float(np.median(depth_crop)) if depth_crop.size > 0 else 0

                    label_text = f"{names[class_id]} {confidence:.2f} D:{mean_depth:.1f}"

                    cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(processed_image, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                print("⚠️ Aucune détection pour cette image.")

        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(processed_image_rgb)
        tk_image = ImageTk.PhotoImage(pil_image)
        label_widget.config(image=tk_image)
        label_widget.image = tk_image

    except Exception as e:
        messagebox.showerror("Erreur de détection", f"{e}")


# ======================
# Interface principale
# ======================
def main():
    root = Tk()
    root.title("YOLOv8 + MiDaS Détection & Profondeur")

    model_path = r"C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\runs\train\yolov11n-warehouse\weights\best.pt"
    if not os.path.exists(model_path):
        messagebox.showerror("Erreur", f"Modèle introuvable : {model_path}")
        root.destroy()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("📦 Chargement du modèle YOLO...")
    model = YOLO(model_path)

    midas, midas_transform = load_midas(device)

    image_label = Label(root, text="Cliquez pour sélectionner une image", bg="lightgray", width=80, height=40)
    image_label.pack(padx=20, pady=20, expand=True, fill="both")

    def on_click_action(event):
        file_path = askopenfilename(filetypes=[("Fichiers images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            run_detection(file_path, model, image_label, midas, midas_transform, device)

    image_label.bind("<Button-1>", on_click_action)
    root.mainloop()


if __name__ == "__main__":
    main()
