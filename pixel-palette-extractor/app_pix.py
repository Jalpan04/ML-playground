# app_pix.py

import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
import gradio as gr

def create_palette_image(colors, swatch_size=50, spacing=10):
    """Creates a PIL image displaying the color palette swatches. 🎨"""
    num_colors = len(colors)
    # Calculate image dimensions based on the number of colors
    width = (swatch_size * num_colors) + (spacing * (num_colors + 1))
    height = swatch_size + (2 * spacing)

    # Create a new image with a light gray background for contrast
    palette = Image.new("RGB", (width, height), (230, 230, 230))
    draw = ImageDraw.Draw(palette)

    # Draw a colored rectangle for each dominant color
    for i, color in enumerate(colors):
        x0 = spacing + i * (swatch_size + spacing)
        y0 = spacing
        x1 = x0 + swatch_size
        y1 = y0 + swatch_size
        draw.rectangle([x0, y0, x1, y1], fill=tuple(color), outline=(50, 50, 50))

    return palette

def extract_palette(input_image, num_colors):
    """
    Extracts the dominant colors from an image using K-Means clustering
    and returns a combined image of the original and the generated palette.
    """
    if input_image is None:
        raise gr.Error("Please upload an image first!")

    # --- 1. Image Processing ---
    # Resize for performance. Clustering on a huge image is slow.
    image = input_image.resize((200, 200))

    # Convert the PIL image to a NumPy array
    np_image = np.array(image)

    # --- 2. Data Reshaping ---
    # Reshape the array to be a list of pixels (N_pixels, 3)
    pixels = np_image.reshape(-1, 3)

    # --- 3. K-Means Clustering ---
    kmeans = KMeans(n_clusters=int(num_colors), random_state=42, n_init='auto')
    kmeans.fit(pixels)

    # --- 4. Get Dominant Colors ---
    # The cluster centers are the average colors of each cluster.
    dominant_colors = kmeans.cluster_centers_.astype(int)

    # --- 5. Create and Combine Output Images ---
    palette_img = create_palette_image(dominant_colors)

    # Create a new blank canvas to place the original image and the palette on
    combined_width = max(input_image.width, palette_img.width)
    combined_height = input_image.height + palette_img.height + 20
    combined_image = Image.new("RGB", (combined_width, combined_height), (230, 230, 230))

    # Paste the original uploaded image at the top
    combined_image.paste(input_image, ((combined_width - input_image.width) // 2, 10))
    # Paste the generated palette image at the bottom
    combined_image.paste(palette_img, ((combined_width - palette_img.width) // 2, input_image.height + 15))

    return combined_image

# --- Gradio Web Interface ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 Pixel Art Color Palette Extractor")
    gr.Markdown(
        "Upload an image and choose the number of colors to automatically generate a palette using **K-Means Clustering**.")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Your Image")
            num_colors_slider = gr.Slider(
                minimum=2,
                maximum=20,
                value=8,
                step=1,
                label="Number of Colors in Palette"
            )
            submit_btn = gr.Button("Generate Palette", variant="primary")

        with gr.Column(scale=2):
            output_image = gr.Image(label="Result", interactive=False)

    # Define the click action for the button
    submit_btn.click(
        fn=extract_palette,
        inputs=[input_image, num_colors_slider],
        outputs=output_image
    )

# --- Launch the App ---

if __name__ == "__main__":
    demo.launch()