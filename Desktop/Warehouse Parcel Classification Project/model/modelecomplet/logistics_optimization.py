# ======================
# Constantes et fonctions pour l'optimisation logistique
# ======================
import matplotlib.pyplot as plt
import numpy as np

# Dimensions moyennes des colis pour l'estimation
MEAN_DIMENSIONS = {
    "colis": (40, 30, 20),
}
# Seuils de capacité des véhicules (en kg et en mètres cubes)
VEHICLE_CAPACITIES = {
    "camionnette": {"poids_max": 500, "volume_max": 5},
    "camion_léger": {"poids_max": 2000, "volume_max": 20},
    "poids_lourd": {"poids_max": 10000, "volume_max": 100},
}

def estimate_dimensions_and_volume(bbox, mean_depth_cm):
    """
    Estime les dimensions 3D et le volume d'un colis.
    Cette fonction se base sur la profondeur estimée et des dimensions moyennes.
    """
    x1, y1, x2, y2 = bbox
    largeur_pixels = x2 - x1
    hauteur_pixels = y2 - y1

    # Utilisation d'une simplification pour la conversion
    longueur_cm = MEAN_DIMENSIONS["colis"][0]
    largeur_cm = largeur_pixels * 0.1 # Exemple de conversion
    hauteur_cm = hauteur_pixels * 0.1 # Exemple de conversion

    return (longueur_cm, largeur_cm, hauteur_cm)


def recommend_vehicle(parcels_data):
    """
    Recommande un véhicule en fonction des dimensions et du poids totaux
    et affiche les résultats sous forme de graphique.
    """
    total_volume_m3 = 0
    total_weight_kg = 0
    has_fragile = False

    for parcel in parcels_data:
        longueur, largeur, hauteur = parcel["dimensions"]
        volume_m3 = (longueur * largeur * hauteur) / 1000000
        total_volume_m3 += volume_m3
        total_weight_kg += parcel["poids"]
        if parcel["est_fragile"]:
            has_fragile = True

    print("\n--- Analyse pour la recommandation de véhicule ---")
    print(f"Volume total des colis : {total_volume_m3:.2f} m³")
    print(f"Poids total des colis : {total_weight_kg:.2f} kg")
    print(f"Présence de colis fragiles : {'Oui' if has_fragile else 'Non'}")

    # Préparation des données pour le graphique
    vehicles = list(VEHICLE_CAPACITIES.keys())
    weight_capacities = [VEHICLE_CAPACITIES[v]["poids_max"] for v in vehicles]
    volume_capacities = [VEHICLE_CAPACITIES[v]["volume_max"] for v in vehicles]

    # Création du graphique à barres
    x = np.arange(len(vehicles))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, weight_capacities, width, label='Poids max (kg)', color='b')
    rects2 = ax.bar(x + width/2, volume_capacities, width, label='Volume max (m³)', color='g')

    # Ajout des données totales
    ax.bar(x, [total_weight_kg] * len(vehicles), width=width, label='Poids total des colis', color='lightblue', alpha=0.7)
    ax.bar(x, [total_volume_m3] * len(vehicles), width=width, label='Volume total des colis', color='lightgreen', alpha=0.7)

    # Étiquettes et titre
    ax.set_ylabel('Valeur')
    ax.set_title('Capacité des véhicules vs. Poids et Volume totaux des colis')
    ax.set_xticks(x)
    ax.set_xticklabels(vehicles)
    ax.legend()
    ax.grid(axis='y', linestyle='--')

    fig.tight_layout()
    plt.show()

    for vehicle, capacity in VEHICLE_CAPACITIES.items():
        if (total_volume_m3 <= capacity["volume_max"] and
            total_weight_kg <= capacity["poids_max"]):
            print(f"✅ Recommandation : {vehicle} (suffisant pour le volume et le poids)")
            return vehicle
    
    print("❌ Aucun véhicule ne peut transporter tous les colis. Répartition nécessaire.")
    return None
