import os
import sys
import subprocess

def launch_studio(port: int = 8501):
    """
    Launch the DataPilot Web Studio (Streamlit application).
    
    Args:
        port: The port to run the web application on (default: 8501)
    """
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit is not installed. Please install it using:")
        print("   pip install datapilot-polars[studio]")
        print("   or")
        print("   pip install streamlit")
        sys.exit(1)
        
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    
    print(f"🚀 Launching DataPilot Web Studio on port {port}...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path, "--server.port", str(port)])
    except KeyboardInterrupt:
        print("\n👋 DataPilot Web Studio closed.")
