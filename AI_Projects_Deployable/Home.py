
import streamlit as st

st.set_page_config(
    page_title="ML Projects Hub",
    page_icon="🤖",
    layout="wide"
)

st.title("Welcome to the AI Projects Hub! 👋")
st.write("A central showcase of machine learning projects. Use the sidebar to navigate to each application.")
st.divider()

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.header("🎨 Palette Extractor")
    st.image("https://i.imgur.com/Sd1eE9j.png")
    st.write("Upload an image to extract its dominant color palette using **K-Means Clustering**.")

with col2:
    st.header(" Tic-Tac-Toe AI")
    st.image("https://i.imgur.com/L7sT3hV.png")
    st.write("Play against an unbeatable AI that uses a **Decision Tree** and the **Minimax algorithm**.")

with col3:
    st.header("🧬 Cellular Automata")
    st.image("https://i.imgur.com/a8hA37J.gif")
    st.write("A simulation of Conway's Game of Life that uses **K-Means Clustering** to classify emerging patterns in real-time.")

st.sidebar.success("Select an application above.")
