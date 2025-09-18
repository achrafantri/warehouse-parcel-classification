import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R
import json
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

class CameraCalibration:
    """Gestion de la calibration de la caméra pour les projections 3D"""
    
    def __init__(self, intrinsic_matrix: np.ndarray, distortion_coeffs: np.ndarray = None):
        self.K = intrinsic_matrix
        self.dist_coeffs = distortion_coeffs if distortion_coeffs is not None else np.zeros(5)
        
    @classmethod
    def from_params(cls, fx: float, fy: float, cx: float, cy: float):
        """Créer la calibration à partir des paramètres intrinsèques"""
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        return cls(K)
    
    @classmethod
    def default_hd_camera(cls):
        """Calibration par défaut pour une caméra HD 1280x720"""
        fx = fy = 800  # Focale approximative
        cx, cy = 640, 360  # Centre de l'image
        return cls.from_params(fx, fy, cx, cy)
    
    def project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """Projeter des points 3D vers l'image 2D"""
        points_2d, _ = cv2.projectPoints(
            points_3d, 
            np.zeros(3), 
            np.zeros(3), 
            self.K, 
            self.dist_coeffs
        )
        return points_2d.reshape(-1, 2)

class BoundingBox3D:
    """Représentation d'une boîte englobante 3D orientée"""
    
    def __init__(self, center: np.ndarray, dimensions: np.ndarray, 
                 rotation_angle: float = 0, confidence: float = 0.0, 
                 class_name: str = "box", is_fragile: bool = False):
        self.center = np.array(center)  # [x, y, z] en cm
        self.dimensions = np.array(dimensions)  # [length, width, height] en cm
        self.rotation_angle = rotation_angle  # Rotation en radians autour de l'axe Z
        self.confidence = confidence
        self.class_name = class_name
        self.is_fragile = is_fragile
        
        # Matrice de rotation 3D
        self.rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
            [np.sin(rotation_angle), np.cos(rotation_angle), 0],
            [0, 0, 1]
        ])
        
    def get_corners(self) -> np.ndarray:
        """Obtenir les 8 coins de la boîte 3D orientée"""
        l, w, h = self.dimensions / 2
        
        # Coins dans le repère local
        corners_local = np.array([
            [-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],  # face inférieure
            [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h]       # face supérieure
        ])
        
        # Application de la rotation et translation
        corners_world = (self.rotation_matrix @ corners_local.T).T + self.center
        return corners_world
    
    def get_volume(self) -> float:
        """Calculer le volume du colis en cm³"""
        return np.prod(self.dimensions)
    
    def estimate_weight(self, density: float = 0.2) -> float:
        """Estimer le poids basé sur la densité moyenne (kg par 1000 cm³)"""
        volume_dm3 = self.get_volume() / 1000  # conversion cm³ -> dm³
        return volume_dm3 * density  # retour en kg
    
    def get_info_dict(self) -> Dict:
        """Retourner les informations sous forme de dictionnaire"""
        return {
            'center': self.center.tolist(),
            'dimensions': {
                'length': round(self.dimensions[0], 1),
                'width': round(self.dimensions[1], 1),
                'height': round(self.dimensions[2], 1)
            },
            'volume_cm3': round(self.get_volume(), 0),
            'estimated_weight_kg': round(self.estimate_weight(), 2),
            'rotation_angle_deg': round(np.degrees(self.rotation_angle), 1),
            'confidence': round(self.confidence, 3),
            'class_name': self.class_name,
            'is_fragile': self.is_fragile
        }

class YOLOv8OBBParcelDetector:
    """Détecteur spécialisé pour colis avec YOLOv8-OBB (classes: box, fragile)"""
    
    def __init__(self, model_path: str, camera_calibration: CameraCalibration = None):
        # Charger le modèle YOLOv8-OBB fine-tuné
        self.model = YOLO(model_path)
        self.camera = camera_calibration or CameraCalibration.default_hd_camera()
        
        # Classes attendues (selon votre mémoire)
        self.classes = {0: 'box', 1: 'fragile'}
        
        # Paramètres pour l'estimation 3D
        self.ground_height = 0  # Hauteur du sol en cm
        self.average_parcel_density = 0.2  # kg par 1000 cm³
        
    def detect_obb(self, image: np.ndarray) -> List[Dict]:
        """Détection avec boîtes orientées (OBB)"""
        # Exécuter l'inférence YOLOv8-OBB
        results = self.model(image, verbose=False)
        detections = []
        
        for result in results:
            if hasattr(result, 'obb') and result.obb is not None:
                # Extraction des boîtes orientées
                for i, obb in enumerate(result.obb):
                    # Coordonnées des boîtes orientées [x_center, y_center, width, height, rotation]
                    xywhr = obb.xywhr[0].cpu().numpy()  # x, y, w, h, rotation
                    confidence = obb.conf[0].cpu().numpy()
                    class_id = int(obb.cls[0].cpu().numpy())
                    class_name = self.classes.get(class_id, 'unknown')
                    
                    # Coins de la boîte orientée en 2D
                    corners_2d = self._get_obb_corners(xywhr)
                    
                    detection = {
                        'xywhr': xywhr,
                        'corners_2d': corners_2d,
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'is_fragile': class_name == 'fragile'
                    }
                    detections.append(detection)
            
            # Fallback: détection classique si OBB non disponible
            elif result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.classes.get(class_id, 'unknown')
                    
                    # Conversion en format OBB
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    xywhr = np.array([center_x, center_y, width, height, 0])
                    
                    corners_2d = np.array([
                        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                    ])
                    
                    detection = {
                        'xywhr': xywhr,
                        'corners_2d': corners_2d,
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'is_fragile': class_name == 'fragile'
                    }
                    detections.append(detection)
        
        return detections
    
    def _get_obb_corners(self, xywhr: np.ndarray) -> np.ndarray:
        """Calculer les coins d'une boîte orientée 2D"""
        x, y, w, h, rotation = xywhr
        
        # Coins dans le repère local
        corners_local = np.array([
            [-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]
        ])
        
        # Matrice de rotation 2D
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        rotation_matrix = np.array([
            [cos_r, -sin_r],
            [sin_r, cos_r]
        ])
        
        # Application rotation + translation
        corners_world = (rotation_matrix @ corners_local.T).T + np.array([x, y])
        return corners_world
    
    def estimate_3d_from_obb(self, detection: Dict, image_shape: Tuple[int, int]) -> BoundingBox3D:
        """Estimation 3D à partir d'une détection OBB 2D"""
        xywhr = detection['xywhr']
        x_2d, y_2d, width_2d, height_2d, rotation_2d = xywhr
        
        # Estimation de la profondeur basée sur la taille apparente
        # Plus l'objet est grand à l'écran, plus il est proche
        reference_width = 200  # Largeur de référence en pixels pour un colis à 2m
        reference_distance = 200  # Distance de référence en cm
        
        estimated_depth = (reference_width / max(width_2d, height_2d)) * reference_distance
        estimated_depth = np.clip(estimated_depth, 50, 500)  # Limiter entre 50cm et 5m
        
        # Conversion pixel vers cm en utilisant la géométrie projective
        # Facteur de conversion basé sur la distance estimée
        pixel_to_cm = estimated_depth / self.camera.K[0, 0]
        
        # Dimensions réelles estimées
        real_width = width_2d * pixel_to_cm
        real_height = height_2d * pixel_to_cm
        # Estimation de la profondeur (3ème dimension) basée sur les proportions
        real_depth = real_width * 0.7  # Ratio approximatif profondeur/largeur
        
        # Position 3D du centre
        center_x = (x_2d - self.camera.K[0, 2]) * pixel_to_cm
        center_y = estimated_depth
        center_z = -(y_2d - self.camera.K[1, 2]) * pixel_to_cm + self.ground_height
        
        # Ajustement pour les colis fragiles (souvent plus grands)
        if detection['is_fragile']:
            real_depth *= 1.2
            real_height *= 1.1
        
        return BoundingBox3D(
            center=np.array([center_x, center_y, center_z]),
            dimensions=np.array([real_width, real_depth, real_height]),
            rotation_angle=rotation_2d,
            confidence=detection['confidence'],
            class_name=detection['class_name'],
            is_fragile=detection['is_fragile']
        )
    
    def process_image(self, image: np.ndarray) -> List[BoundingBox3D]:
        """Traitement complet : détection OBB 2D -> estimation 3D"""
        # Détection 2D avec boîtes orientées
        detections_2d = self.detect_obb(image)
        
        # Conversion en 3D
        boxes_3d = []
        for detection in detections_2d:
            box_3d = self.estimate_3d_from_obb(detection, image.shape[:2])
            boxes_3d.append(box_3d)
        
        return boxes_3d
    
    def visualize_results(self, image: np.ndarray, boxes_3d: List[BoundingBox3D], 
                         save_path: str = None) -> np.ndarray:
        """Visualisation complète : image annotée + graphique 3D"""
        # Créer une figure avec deux sous-graphiques
        fig = plt.figure(figsize=(16, 8))
        
        # 1. Image avec boîtes 2D projetées
        ax1 = plt.subplot(1, 2, 1)
        annotated_image = self.draw_3d_boxes_on_image(image, boxes_3d)
        ax1.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Détection 2D avec Projections 3D')
        ax1.axis('off')
        
        # 2. Visualisation 3D
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        self._plot_3d_boxes(ax2, boxes_3d)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return annotated_image
    
    def _plot_3d_boxes(self, ax, boxes_3d: List[BoundingBox3D]):
        """Tracé des boîtes 3D"""
        colors = {'box': 'blue', 'fragile': 'red'}
        
        for i, box in enumerate(boxes_3d):
            corners = box.get_corners()
            color = colors.get(box.class_name, 'gray')
            
            # Dessiner les arêtes de la boîte
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # face inférieure
                [4, 5], [5, 6], [6, 7], [7, 4],  # face supérieure
                [0, 4], [1, 5], [2, 6], [3, 7]   # arêtes verticales
            ]
            
            for edge in edges:
                points = corners[edge]
                ax.plot3D(*points.T, color=color, linewidth=2, alpha=0.7)
            
            # Marquer le centre
            ax.scatter(*box.center, color=color, s=50, alpha=0.8)
            
            # Ajouter les informations textuelles
            info = box.get_info_dict()
            text = (f"{box.class_name.upper()}\n"
                   f"L:{info['dimensions']['length']}cm\n"
                   f"W:{info['dimensions']['width']}cm\n"
                   f"H:{info['dimensions']['height']}cm\n"
                   f"Vol:{info['volume_cm3']}cm³\n"
                   f"~{info['estimated_weight_kg']}kg")
            
            ax.text(box.center[0], box.center[1], box.center[2] + box.dimensions[2]/2,
                   text, fontsize=8, ha='center', va='bottom')
        
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')  
        ax.set_zlabel('Z (cm)')
        ax.set_title('Vue 3D des Colis Détectés')
        
        # Légende
        import matplotlib.patches as mpatches
        box_patch = mpatches.Patch(color='blue', label='Box')
        fragile_patch = mpatches.Patch(color='red', label='Fragile')
        ax.legend(handles=[box_patch, fragile_patch])
    
    def draw_3d_boxes_on_image(self, image: np.ndarray, boxes_3d: List[BoundingBox3D]) -> np.ndarray:
        """Dessiner les boîtes 3D projetées sur l'image 2D"""
        result_image = image.copy()
        
        for box in boxes_3d:
            # Obtenir les coins 3D et les projeter en 2D
            corners_3d = box.get_corners()
            corners_2d = self.camera.project_3d_to_2d(corners_3d)
            corners_2d = corners_2d.astype(int)
            
            # Couleur selon la classe
            color = (0, 0, 255) if box.is_fragile else (0, 255, 0)  # Rouge pour fragile, vert pour box
            
            # Dessiner les arêtes projetées
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # face inférieure
                [4, 5], [5, 6], [6, 7], [7, 4],  # face supérieure  
                [0, 4], [1, 5], [2, 6], [3, 7]   # arêtes verticales
            ]
            
            for edge in edges:
                pt1, pt2 = corners_2d[edge[0]], corners_2d[edge[1]]
                # Vérifier que les points sont dans l'image
                if (0 <= pt1[0] < image.shape[1] and 0 <= pt1[1] < image.shape[0] and
                    0 <= pt2[0] < image.shape[1] and 0 <= pt2[1] < image.shape[0]):
                    cv2.line(result_image, tuple(pt1), tuple(pt2), color, 2)
            
            # Projeter le centre et ajouter les informations
            center_2d = self.camera.project_3d_to_2d(box.center.reshape(1, -1))[0].astype(int)
            if (0 <= center_2d[0] < image.shape[1] and 0 <= center_2d[1] < image.shape[0]):
                info = box.get_info_dict()
                
                # Texte principal
                main_text = f"{box.class_name.upper()}"
                dim_text = f"L:{info['dimensions']['length']} W:{info['dimensions']['width']} H:{info['dimensions']['height']}"
                vol_text = f"Vol:{info['volume_cm3']}cm³ ~{info['estimated_weight_kg']}kg"
                conf_text = f"Conf:{info['confidence']:.2f}"
                
                # Position du texte
                text_pos = (center_2d[0] - 60, center_2d[1] - 30)
                
                # Fond semi-transparent pour le texte
                overlay = result_image.copy()
                cv2.rectangle(overlay, (text_pos[0]-5, text_pos[1]-25), 
                            (text_pos[0]+200, text_pos[1]+40), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, result_image, 0.3, 0, result_image)
                
                # Ajouter les textes
                cv2.putText(result_image, main_text, text_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(result_image, dim_text, (text_pos[0], text_pos[1]+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(result_image, vol_text, (text_pos[0], text_pos[1]+30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(result_image, conf_text, (text_pos[0], text_pos[1]+45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return result_image
    
    def save_results_json(self, boxes_3d: List[BoundingBox3D], output_path: str):
        """Sauvegarder les résultats en JSON"""
        results = {
            'total_parcels': len(boxes_3d),
            'fragile_parcels': sum(1 for box in boxes_3d if box.is_fragile),
            'total_volume_cm3': sum(box.get_volume() for box in boxes_3d),
            'total_estimated_weight_kg': sum(box.estimate_weight() for box in boxes_3d),
            'parcels': [box.get_info_dict() for box in boxes_3d]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Résultats sauvegardés dans {output_path}")
        return results

# Script d'entraînement YOLOv8-OBB
class YOLOv8OBBTrainer:
    """Classe pour l'entraînement du modèle YOLOv8-OBB"""
    
    def __init__(self, data_yaml_path: str, base_model: str = "yolov8n-obb.pt"):
        self.data_yaml_path = data_yaml_path
        self.base_model = base_model
        
    def train(self, epochs: int = 50, imgsz: int = 640, batch: int = 2, 
              device: str = 'auto', project: str = 'runs/obb', name: str = 'parcel_obb'):
        """Entraîner le modèle YOLOv8-OBB"""
        
        # Charger le modèle de base
        model = YOLO(self.base_model)
        
        # Configuration d'entraînement
        train_args = {
            'data': self.data_yaml_path,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': device,
            'project': project,
            'name': name,
            'save_period': 10,  # Sauvegarder tous les 10 epochs
            'patience': 10,     # Early stopping
            'cache': False,
            'single_cls': False,  # Multi-classes (box, fragile)
            'rect': False,
            'cos_lr': True,
            'close_mosaic': 0,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'split': 'val',
            'save_json': True,
            'save_hybrid': False,
            'conf': None,
            'iou': 0.7,
            'max_det': 300,
            'half': False,
            'dnn': False,
            'plots': True,
            'source': None,
            'vid_stride': 1,
            'stream_buffer': False,
            'visualize': False,
            'augment': False,
            'agnostic_nms': False,
            'classes': None,
            'retina_masks': False,
            'embed': None,
            'show': False,
            'save_frames': False,
            'save_txt': False,
            'save_conf': False,
            'save_crop': False,
            'show_labels': True,
            'show_conf': True,
            'show_boxes': True,
            'line_width': None
        }
        
        # Lancer l'entraînement
        results = model.train(**train_args)
        
        # Retourner le chemin du meilleur modèle
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        return str(best_model_path)

# Exemple d'utilisation
def main():
    # Exemple d'utilisation du détecteur
    
    # 1. Initialiser le détecteur avec votre modèle entraîné
    model_path = "path/to/your/best.pt"  # Chemin vers votre modèle fine-tuné
    detector = YOLOv8OBBParcelDetector(model_path)
    
    # 2. Charger et traiter une image
    image_path = "path/to/test/image.jpg"
    image = cv2.imread(image_path)
    
    if image is not None:
        # 3. Détecter les colis 3D
        boxes_3d = detector.process_image(image)
        
        # 4. Afficher les résultats
        print(f"Détectés: {len(boxes_3d)} colis")
        for i, box in enumerate(boxes_3d):
            info = box.get_info_dict()
            print(f"Colis {i+1}: {info}")
        
        # 5. Visualiser
        detector.visualize_results(image, boxes_3d, "results_visualization.png")
        
        # 6. Sauvegarder en JSON
        results = detector.save_results_json(boxes_3d, "detection_results.json")
        
        print(f"Résultats: {results['total_parcels']} colis détectés")
        print(f"Volume total: {results['total_volume_cm3']} cm³")
        print(f"Poids estimé: {results['total_estimated_weight_kg']} kg")

if __name__ == "__main__":
    main()