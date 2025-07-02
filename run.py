import subprocess
import os
import sys

def run_streamlit():
    """Run the Streamlit application with the correct parameters."""
    try:
        # Get port from environment or use default
        port = os.environ.get("PORT", "8000")
        
        # Build the command
        cmd = [
            "streamlit", "run", 
            "app.py",
            "--server.port", port,
            "--server.address", "0.0.0.0"
        ]
        
        # Add browser flag if needed
        if "--no-browser" in sys.argv:
            cmd.append("--server.headless")
        
        # Run the streamlit command
        print(f"Starting SmartPaperQ on http://localhost:{port}")
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("Shutting down SmartPaperQ...")
    except Exception as e:
        print(f"Error starting SmartPaperQ: {e}")

if __name__ == "__main__":
    run_streamlit()
