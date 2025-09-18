import os
import torch
import cv2
from ultralytics import YOLO
from tkinter import Tk, Label, messagebox
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk

def run_detection(image_path, model, label_widget):
    """Exécute la détection et met à jour l'interface avec le résultat."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror("Erreur", "Impossible de lire l'image.")
            return

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        results = model.predict(image_path, imgsz=640, conf=0.3, device=device, verbose=False)
        processed_image = image.copy()

        for result in results:
            # Vérifier si des boxes existent
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                names = result.names

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    class_id = int(class_ids[i])
                    confidence = confidences[i]
                    label_text = f"{names[class_id]}: {confidence:.2f}"

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

def main():
    root = Tk()
    root.title("YOLOv8 Détection")

    model_path = r'C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\runs\train\yolov8n-obb-warehouse\weights\best.pt'
    if not os.path.exists(model_path):
        messagebox.showerror("Erreur", f"Modèle introuvable : {model_path}")
        root.destroy()
        return

    model = YOLO(model_path)

    image_label = Label(root, text="Cliquez pour sélectionner une image", bg="lightgray", width=80, height=40)
    image_label.pack(padx=20, pady=20, expand=True, fill="both")

    def on_click_action(event):
        file_path = askopenfilename(filetypes=[("Fichiers images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            run_detection(file_path, model, image_label)

    image_label.bind("<Button-1>", on_click_action)
    root.mainloop()

if __name__ == "__main__":
    main()
