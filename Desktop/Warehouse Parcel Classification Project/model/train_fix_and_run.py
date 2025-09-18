# 🔧 Script de correction NumPy et entraînement
import subprocess
import sys
import os

def install_compatible_numpy():
    """Installe une version compatible de NumPy"""
    print("🔧 Installation de NumPy compatible...")
    try:
        # Downgrader NumPy vers une version compatible
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "numpy<2.0", "--force-reinstall", "--no-deps"
        ], check=True)
        print("✅ NumPy compatible installé")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'installation de NumPy: {e}")
        return False

def main():
    print("🚀 Script de correction et d'entraînement YOLOv8")
    
    # Installer NumPy compatible
    if not install_compatible_numpy():
        return
    
    # Redémarrer Python pour recharger NumPy
    print("\n🔄 Redémarrage du processus avec NumPy compatible...")
    
    # Relancer le script principal
    script_path = os.path.join(os.path.dirname(__file__), "train_standalone.py")
    os.environ["PYTHONPATH"] = ""  # Nettoyer le path
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              cwd=os.path.dirname(__file__))
        return result.returncode
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
