import streamlit as st
import subprocess
import webbrowser
import psutil
import sys
import os

st.set_page_config(
    page_title="ML Projects Launcher",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 ML Projects Launcher Hub")
st.markdown("Start and stop your individual project servers from here. Each app will open in a new tab.")

# --- Project Definitions ---
# Added 'cwd' (current working directory) to fix model path errors.
PROJECTS = {
    "palette_extractor": {
        "name": "🎨 Pixel Palette Extractor (Gradio)",
        "command": [sys.executable, "app_pix.py"],
        "cwd": "pixel-palette-extractor",
        "port": 7860,
        "url": "http://127.0.0.1:7860"
    },
    "tictactoe": {
        "name": " Tic-Tac-Toe AI (Streamlit)",
        "command": ["streamlit", "run", "app_tic.py", "--server.port", "8502"],
        "cwd": "tictactoe_tree",
        "port": 8502,
        "url": "http://127.0.0.1:8502"
    },
    "cellular_automata": {
        "name": "🧬 Cellular Automata (Streamlit)",
        "command": ["streamlit", "run", "app_cell.py", "--server.port", "8503"],
        "cwd": "cellular-automata-classifier",
        "port": 8503,
        "url": "http://127.0.0.1:8503"
    }
}

# Session state now tracks the Popen object directly for safe termination.
if 'processes' not in st.session_state:
    st.session_state.processes = {key: None for key in PROJECTS}


def find_process_by_port(port):
    """Checks if a port is occupied."""
    for conn in psutil.net_connections():
        if conn.laddr.port == port and conn.status == 'LISTEN':
            try:
                return psutil.Process(conn.pid)
            except psutil.NoSuchProcess:
                continue
    return None


# --- UI Layout ---
for key, config in PROJECTS.items():
    st.divider()
    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        st.subheader(config['name'])
        st.caption(f"Running from: `{os.path.abspath(config['cwd'])}`")

    # Check the real-time status of the port
    process_on_port = find_process_by_port(config['port'])

    with col2:
        if st.button("Start Server", key=f"start_{key}", disabled=bool(process_on_port)):
            with st.spinner(f"Starting {config['name']}..."):
                flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                # Use the 'cwd' argument to set the correct working directory
                proc = subprocess.Popen(
                    config['command'],
                    cwd=config['cwd'],
                    creationflags=flags
                )
                st.session_state.processes[key] = proc
                st.rerun()

        if st.button("Stop Server", key=f"stop_{key}", disabled=not bool(process_on_port)):
            if process_on_port:
                try:
                    # Terminate the specific process this app is tracking
                    parent = process_on_port
                    for child in parent.children(recursive=True):
                        child.terminate()
                    parent.terminate()
                    st.success(f"Stopped server for {config['name']}.")
                    st.session_state.processes[key] = None  # Clear from state
                    st.rerun()
                except psutil.NoSuchProcess:
                    st.warning("Process was already stopped.")
                    st.session_state.processes[key] = None  # Clear from state
                    st.rerun()
                except Exception as e:
                    st.error(f"Error stopping process: {e}")
                    st.session_state.processes[key] = None
            else:
                st.warning("No process found to stop.")

    with col3:
        if process_on_port:
            st.success(f"✅ A server is running on port {config['port']}")
            if st.button("Open App", key=f"open_{key}"):
                webbrowser.open_new_tab(config['url'])
            st.markdown(f"Access at: [{config['url']}]({config['url']})")
        else:
            st.warning(f"⚪ Server is stopped.")