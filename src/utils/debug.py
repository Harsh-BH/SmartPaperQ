import os
import sys
import shutil
import platform
import requests
import streamlit as st
import tempfile
import subprocess

def check_system_info():
    """Return system information for debugging."""
    info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "cwd": os.getcwd(),
        "executable": sys.executable,
        "user_home": os.path.expanduser("~"),
        "temp_dir": tempfile.gettempdir()
    }
    return info

def check_download_permissions(path):
    """Check if the download directory is writable."""
    try:
        test_file = os.path.join(path, "test_write_permission.txt")
        with open(test_file, "w") as f:
            f.write("Test write permission")
        os.remove(test_file)
        return {"writable": True, "path": path}
    except Exception as e:
        return {"writable": False, "path": path, "error": str(e)}

def test_network_connection():
    """Test network connectivity to arxiv.org."""
    try:
        response = requests.get("https://arxiv.org", timeout=5)
        return {
            "connected": response.status_code == 200,
            "status_code": response.status_code
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}

def check_file_permissions(directory):
    """Check permissions for files in the directory."""
    results = []
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                file_stat = os.stat(file_path)
                is_readable = os.access(file_path, os.R_OK)
                is_writable = os.access(file_path, os.W_OK)
                results.append({
                    "file": filename,
                    "size": file_stat.st_size,
                    "readable": is_readable,
                    "writable": is_writable
                })
            except Exception as e:
                results.append({
                    "file": filename,
                    "error": str(e)
                })
        return results
    except Exception as e:
        return [{"error": f"Could not list directory: {str(e)}"}]

def display_debug_info(download_dir):
    """Display debug information in a Streamlit expander."""
    with st.expander("Debug Information", expanded=True):
        st.write("### System Information")
        system_info = check_system_info()
        for key, value in system_info.items():
            st.write(f"**{key}:** {value}")
        
        st.write("### Network Connection")
        network_info = test_network_connection()
        if network_info.get("connected"):
            st.success(f"✅ Connection to arxiv.org successful (Status code: {network_info['status_code']})")
        else:
            st.error(f"❌ Failed to connect to arxiv.org: {network_info.get('error', 'Unknown error')}")
            
        st.write("### Download Directory")
        download_info = check_download_permissions(download_dir)
        st.write(f"**Path:** {download_info['path']}")
        st.write(f"**Writable:** {download_info['writable']}")
        if not download_info['writable']:
            st.error(f"Error: {download_info.get('error', 'Unknown error')}")
            
            # Suggest alternative download locations
            home_dir = os.path.expanduser("~")
            st.write("### Suggested Alternative Locations")
            st.write(f"1. Home directory: `{home_dir}`")
            st.write(f"2. Temporary directory: `{tempfile.gettempdir()}`")
            st.write(f"3. Current directory: `{os.getcwd()}`")
            
            # Add a fix button for permissions
            if st.button("Try to fix permissions"):
                try:
                    # Try to change permissions if on Unix
                    if os.name == 'posix':
                        cmd = ["chmod", "-R", "755", download_dir]
                        subprocess.run(cmd, check=True)
                        st.success(f"Changed permissions on {download_dir}")
                    else:
                        st.warning("Automatic permission fixing is only available on Unix systems")
                except Exception as e:
                    st.error(f"Failed to fix permissions: {e}")
                    
            # Add a create alternative directory button
            alt_dir = os.path.join(os.getcwd(), "downloads")
            if st.button("Create alternative download directory"):
                try:
                    os.makedirs(alt_dir, exist_ok=True)
                    if check_download_permissions(alt_dir)["writable"]:
                        st.success(f"Created writable directory at: {alt_dir}")
                    else:
                        st.error(f"Created directory but it's not writable: {alt_dir}")
                except Exception as e:
                    st.error(f"Failed to create directory: {e}")
        
        # List files in the download directory
        st.write("### Files in Download Directory")
        try:
            files = os.listdir(download_dir)
            if files:
                file_permissions = check_file_permissions(download_dir)
                for file_info in file_permissions:
                    if "error" in file_info:
                        st.error(f"Error: {file_info['error']}")
                    else:
                        status = "✅" if file_info["readable"] and file_info["writable"] else "❌"
                        permissions = f"Read: {'✓' if file_info['readable'] else '✗'}, Write: {'✓' if file_info['writable'] else '✗'}"
                        st.write(f"{status} {file_info['file']} ({file_info['size']} bytes) - {permissions}")
            else:
                st.write("No files found")
        except Exception as e:
            st.error(f"Error listing files: {e}")
            
        # Test downloading a sample file
        st.write("### Test Download")
        if st.button("Test downloading a small file"):
            try:
                # Try to download a small test file
                url = "https://arxiv.org/favicon.ico"
                test_file = os.path.join(download_dir, "test_download.ico")
                
                with st.spinner("Downloading test file..."):
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    with open(test_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                
                if os.path.exists(test_file) and os.path.getsize(test_file) > 0:
                    st.success("✅ Test download successful!")
                else:
                    st.error("❌ Test file is empty or doesn't exist")
            except Exception as e:
                st.error(f"❌ Test download failed: {e}")

def add_debug_button_to_app(download_dir):
    """Add a debug button to the app."""
    if st.button("Debug Download Issues"):
        display_debug_info(download_dir)
        
        # Additional ArXiv-specific tests
        with st.expander("Test ArXiv API"):
            try:
                import arxiv
                st.info("Testing arxiv API connection...")
                
                client = arxiv.Client()
                search = arxiv.Search(query="deep learning", max_results=1)
                
                with st.spinner("Searching for a test paper..."):
                    results = list(client.results(search))
                
                if results:
                    st.success(f"✅ Found paper: {results[0].title}")
                    
                    # Try to get the PDF URL
                    st.info(f"PDF URL: {results[0].pdf_url}")
                    
                    # Test if the URL is accessible
                    try:
                        response = requests.head(results[0].pdf_url, timeout=5)
                        if response.status_code == 200:
                            st.success(f"✅ PDF URL is accessible (Status: {response.status_code})")
                        else:
                            st.warning(f"⚠️ PDF URL returned status code: {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ Failed to access PDF URL: {e}")
                else:
                    st.warning("No results found for the test query")
            except Exception as e:
                st.error(f"❌ Error testing arxiv API: {e}")

def add_debug_button_to_app(download_dir):
    """Adds a debug button that shows system information to help troubleshoot issues."""
    if st.button("Show Debug Info"):
        st.write("### System Information")
        st.code(f"""
Python version: {sys.version}
Platform: {platform.platform()}
Download directory: {download_dir}
Directory exists: {os.path.exists(download_dir)}
Directory writable: {os.access(download_dir, os.W_OK)}
Contents: {os.listdir(download_dir) if os.path.exists(download_dir) else 'N/A'}
        """)
