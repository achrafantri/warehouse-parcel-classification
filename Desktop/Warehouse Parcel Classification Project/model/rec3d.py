import numpy as np
import open3d as o3d
import random

def create_box_point_cloud(center, size, num_points=1000):
    """
    Crée un nuage de points simple représentant une boîte.
    Ceci simule l'acquisition de données par une caméra de profondeur.
    """
    x_min, y_min, z_min = center - size / 2
    x_max, y_max, z_max = center + size / 2
    
    points = []
    for _ in range(num_points):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        z = random.uniform(z_min, z_max)
        points.append([x, y, z])
    
    return np.array(points)

def detect_3d_bounding_box(point_cloud):
    """
    Détecte la boîte englobante 3D orientée pour un nuage de points.
    """
    if not point_cloud:
        print("Erreur: Le nuage de points est vide.")
        return None
        
    # Utiliser le nuage de points pour créer un objet Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    # Calculer la boîte englobante orientée (Oriented Bounding Box)
    # C'est l'étape clé de la "détection" dans ce script simple.
    # Pour un cas réel, cette étape serait un modèle d'IA sophistiqué.
    oriented_box = pcd.get_oriented_bounding_box()
    
    return oriented_box

def create_coordinate_frame(size=1.0):
    """Crée un repère de coordonnées (axes X, Y, Z) pour la visualisation."""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

def main():
    print("🚀 Début de la détection d'objets 3D")
    
    # --- Étape 1 : Simulation de l'acquisition des données (Nuage de points) ---
    # Dans un scénario réel, ces données proviendraient d'un capteur LiDAR ou d'une caméra de profondeur.
    
    # Définir les paramètres du "colis" simulé
    colis_center = np.array([0, 0, 1.5])  # Position (x, y, z)
    colis_size = np.array([0.5, 0.3, 0.4]) # Dimensions (L, l, H)
    
    # Générer le nuage de points pour le colis
    colis_points = create_box_point_cloud(colis_center, colis_size)
    
    # Générer un "plan" de sol pour le contexte
    plane_points = np.random.rand(5000, 3) * np.array([2.0, 2.0, 0.1]) - np.array([1.0, 1.0, 0.05])
    
    # Combiner les points pour le nuage de points global
    total_points = np.vstack((colis_points, plane_points))
    
    # --- Étape 2 : Détection de la boîte englobante 3D ---
    print("🧠 Traitement du nuage de points pour détecter la boîte englobante...")
    detected_box = detect_3d_bounding_box(colis_points)
    
    # Vérifier que la détection a réussi
    if detected_box is None:
        return
        
    print("✅ Détection de la boîte englobante réussie.")
    
    # --- Étape 3 : Affichage 3D ---
    print("🖼️ Visualisation des résultats...")
    
    # Créer le nuage de points pour la visualisation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(total_points)
    
    # Définir la couleur de la boîte englobante (par exemple, le jaune comme dans votre image)
    detected_box.color = (1.0, 0.8, 0.0) # Jaune
    
    # Afficher le nuage de points, le repère de coordonnées et la boîte détectée
    o3d.visualization.draw_geometries([pcd, create_coordinate_frame(), detected_box])
    
    print("🏁 Fin du script.")

if __name__ == "__main__":
    main()