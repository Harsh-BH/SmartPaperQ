import subprocess
import sys
import os

def setup_nltk():
    """Download required NLTK resources."""
    print("Setting up NLTK resources...")
    try:
        import nltk
        
        # Create a specific directory for NLTK data
        nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Download resources
        nltk.download('punkt', download_dir=nltk_data_dir)
        nltk.download('stopwords', download_dir=nltk_data_dir)
        
        print("NLTK setup complete.")
    except Exception as e:
        print(f"Error setting up NLTK: {e}")
        sys.exit(1)

def install_requirements():
    """Install required packages."""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully.")
    except Exception as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("Setting up Paper-shaper...")
    install_requirements()
    setup_nltk()
    print("\nSetup complete! You can now run the application with 'streamlit run app.py'")
