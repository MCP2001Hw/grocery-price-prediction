import subprocess
import sys

REQUIREMENTS_FILE = "CNN/requirements.txt"

def install_dependencies():
    try:
        print(f"Installing dependencies from {REQUIREMENTS_FILE}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])
        print("Installation complete.")
    except FileNotFoundError:
        print(f"Error: {REQUIREMENTS_FILE} not found.")
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_dependencies()
