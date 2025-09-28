
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
import io

def create_palette_image(colors, swatch_size=50, spacing=10):
    num_colors = len(colors)
    width = (swatch_size * num_colors) + (spacing * (num_colors + 1))
    height = swatch_size + (2 * spacing)
    palette = Image.new("RGB", (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(palette)
    for i, color in enumerate(colors):
        x0, y0 = spacing + i * (swatch_size + spacing), spacing
        x1, y1 = x0 + swatch_size, y0 + swatch_size
        draw.rectangle([x0, y0, x1, y1], fill=tuple(color), outline=(50, 50, 50))
    return palette

def extract_palette(input_image, num_colors):
    if not input_image: return None
    image = input_image.resize((200, 200))
    np_image = np.array(image)
    pixels = np_image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=int(num_colors), random_state=42, n_init='auto')
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    return create_palette_image(dominant_colors)

st.set_page_config(layout="wide", page_title="Pixel Palette Extractor")
st.title("🎨 Pixel Palette Extractor")
st.markdown("Upload an image to extract its dominant color palette using **K-Means Clustering**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
num_colors = st.slider("Number of Colors", 2, 20, 8)

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(input_image, caption="Uploaded Image", use_column_width=True)
    if st.button("Generate Palette", type="primary", use_container_width=True):
        with st.spinner("Extracting colors..."):
            palette = extract_palette(input_image, num_colors)
            with col2: st.image(palette, caption=f"Generated {num_colors}-Color Palette")
else:
    st.info("Upload an image to get started.")
