import os
import torch
from tkinter import Tk, Label, messagebox, Text, Scrollbar, END
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import cv2

from model_loader import load_models
from processing import process_image_and_parcels

def main():
    """
    Fonction principale pour l'interface utilisateur et l'orchestration du pipeline.
    """
    root = Tk()
    root.title("YOLOv8 OBB Pipeline Logistique")

    # Mise à jour du chemin du modèle YOLO
    # Assurez-vous que le chemin est correct pour votre machine
    model_path = r"C:\Users\user\Desktop\Warehouse Parcel Classification Project\model\runs\finetune\yolov8n_finetune_box_fragile\weights\best.pt"
    if not os.path.exists(model_path):
        messagebox.showerror("Erreur", f"Modèle introuvable : {model_path}\nVeuillez vérifier le chemin.")
        root.destroy()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Charger les modèles via le module model_loader
    model_yolo, model_midas, transform_midas = load_models(model_path, device)
    if not model_yolo or not model_midas:
        root.destroy()
        return

    # Configuration de l'interface graphique
    frame = Label(root)
    frame.pack(padx=10, pady=10)

    image_label = Label(frame, text="Cliquez pour sélectionner une image", bg="lightgray", width=80, height=40)
    image_label.pack(side="left", padx=10)
    
    # Créer un widget Text pour afficher les résultats
    result_text = Text(frame, height=25, width=50)
    scrollbar = Scrollbar(frame, command=result_text.yview)
    result_text.configure(yscrollcommand=scrollbar.set)
    
    result_text.pack(side="right", padx=10, fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # L'étiquette pour la recommandation de véhicule a été supprimée.
    
    def on_click_action(event):
        """
        Gère l'action de clic pour sélectionner une image et lancer le traitement.
        """
        file_path = askopenfilename(filetypes=[("Fichiers images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            # Traiter l'image et obtenir les résultats
            processed_image_rgb, parcels_data, vehicle_recommendation = process_image_and_parcels(file_path, model_yolo, model_midas, transform_midas, device)

            if processed_image_rgb is not None:
                # Afficher l'image traitée dans l'interface
                pil_image = Image.fromarray(cv2.cvtColor(processed_image_rgb, cv2.COLOR_BGR2RGB))
                tk_image = ImageTk.PhotoImage(pil_image)
                image_label.config(image=tk_image)
                image_label.image = tk_image
                
                # Afficher les résultats de l'analyse dans le widget Text
                result_text.delete(1.0, END) # Clear previous results
                if parcels_data:
                    result_text.insert(END, "Résultats de l'analyse des colis:\n\n")
                    for i, parcel in enumerate(parcels_data):
                        result_text.insert(END, f"Colis {i+1}:\n")
                        result_text.insert(END, f"  - Classe: {parcel['class']}\n")
                        result_text.insert(END, f"  - Poids estimé: {parcel['poids']:.1f} kg\n")
                        result_text.insert(END, f"  - Dimensions estimées: L:{parcel['dimensions'][0]:.1f} x l:{parcel['dimensions'][1]:.1f} x H:{parcel['dimensions'][2]:.1f} cm\n")
                        result_text.insert(END, f"  - Fragilité: {'Oui' if parcel['est_fragile'] else 'Non'}\n\n")
                else:
                    result_text.insert(END, "Aucun colis détecté.")
                
    image_label.bind("<Button-1>", on_click_action)
    root.mainloop()

if __name__ == "__main__":
    main()
