# Streamlit application main entry point
import os
import streamlit as st
from src.interface.web_app import create_app
from src.utils.debug import add_debug_button_to_app

# Create a download directory with absolute path
download_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "downloads"))
os.makedirs(download_dir, exist_ok=True)

# Add debug option in the sidebar
add_debug_button_to_app(download_dir)

# This will define the Streamlit app
create_app()

# Note: Run this with: streamlit run app.py
