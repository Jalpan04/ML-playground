# ML Report: K-Means for Color Palette Extraction

**Date:** September 18, 2025
**Subject:** Analysis of a project for dominant color extraction from images.

---

## 1.0 Abstract

This report provides a comprehensive analysis of the project, which is designed to extract a dominant color palette from a user-provided image. The core of the application leverages the **K-Means clustering algorithm**, an unsupervised machine learning technique, to identify the most representative colors. 🎨 The project is implemented using libraries such as `scikit-learn` for the clustering model, `Pillow (PIL)` for image manipulation, `NumPy` for efficient numerical operations, and `Gradio` for creating the web-based user interface. This document will dissect the code's functionality, explain the underlying mathematical principles of the K-Means algorithm, and detail its specific application within this context.

---

## 2.0 Code Implementation Analysis

The project is logically structured into three main components: a utility function for palette visualization, the core extraction logic, and the user interface definition.

### 2.1 Dependencies
The implementation relies on the following key Python libraries:
* **NumPy:** Used for high-performance multidimensional array operations, primarily to represent the image as a matrix of pixel data.
* **Pillow (PIL):** A powerful image processing library used for opening, resizing, and creating image objects.
* **Scikit-learn:** Provides the implementation of the K-Means clustering algorithm (`sklearn.cluster.KMeans`).
* **Gradio:** A framework used to build and deploy the interactive web interface for the application.

### 2.2 Function: `create_palette_image`
This is a helper function responsible for rendering the final color palette.
* **Purpose:** To generate a new image file that visually displays the extracted dominant colors as a series of adjacent color swatches.
* **Process:**
    1.  It calculates the necessary dimensions for the output image based on the number of colors (`num_colors`), the size of each swatch (`swatch_size`), and the spacing between them.
    2.  A new `PIL.Image` object is created with a light gray background to ensure color contrast.
    3.  The function iterates through the list of input colors. For each color, it uses `ImageDraw.Draw.rectangle` to draw a filled rectangle (a swatch) at a calculated position on the new image canvas.

### 2.3 Function: `extract_palette`
This is the central function where the core logic of color extraction resides. It orchestrates the entire process from image input to final output.

* **Step 1: Image Processing & Pre-processing**
    * The input `PIL` image is first resized to a smaller, fixed dimension (200x200 pixels). This is a critical optimization step to reduce the number of data points (pixels) fed into the clustering algorithm, significantly decreasing computation time without substantial loss of color information.
    * The resized image is then converted into a NumPy array of shape `(height, width, 3)`, where the third dimension represents the R, G, B channels.

* **Step 2: Data Reshaping**
    * The 3D NumPy array is reshaped into a 2D array of shape `(N_pixels, 3)`, where `N_pixels = height * width`. This transformation is fundamental, as it converts the image from a spatial grid of pixels into a list of data points. Each row in this new array represents a single pixel, and the three columns correspond to its R, G, and B values. This format is the required input for the `scikit-learn` K-Means model.

* **Step 3: K-Means Clustering**
    * An instance of `sklearn.cluster.KMeans` is initialized.
    * The `n_clusters` parameter is set to the user-defined `num_colors`. This directly corresponds to **K**, the number of clusters to be identified.
    * `random_state=42` is used to ensure the initial placement of centroids is deterministic, making the output reproducible.
    * `n_init='auto'` is set to use the recommended default strategy for running the algorithm multiple times with different centroid seeds to find a better optimum.
    * The `kmeans.fit(pixels)` method is called. This executes the K-Means algorithm on the reshaped pixel data.

* **Step 4: Extraction of Dominant Colors**
    * After the model is fitted, the dominant colors are retrieved from the `kmeans.cluster_centers_` attribute. Each cluster center (or **centroid**) represents the mean R, G, and B values of all pixels belonging to that cluster. These centroids are, by definition, the average and most representative colors in the image, hence the dominant colors.
    * The values are converted to integers to be used as valid RGB color codes.

* **Step 5: Output Composition**
    * The extracted `dominant_colors` are passed to the `create_palette_image` function to generate the palette visualization.
    * A new, larger canvas is created to hold both the original input image and the generated color palette for a consolidated final output.

---

## 3.0 Algorithmic and Mathematical Foundations: K-Means Clustering

K-Means is an iterative, unsupervised machine learning algorithm used to partition a dataset into a pre-determined number ($K$) of distinct, non-overlapping subgroups (clusters). The goal is to group similar data points together while minimizing the variance within each cluster. 🧠

### 3.1 The Algorithm
The algorithm operates as follows:
1.  **Initialization:** Choose $K$ initial data points from the dataset to serve as the initial cluster centers (centroids). This can be done randomly or using a more sophisticated method (e.g., k-means++).
2.  **Assignment Step:** For each data point in the dataset, calculate its distance to every centroid. Assign the data point to the cluster whose centroid is closest. The most common distance metric used is the Euclidean distance.
3.  **Update Step:** Recalculate the centroid of each cluster by taking the mean of all data points assigned to that cluster.
4.  **Iteration:** Repeat the **Assignment** and **Update** steps until a convergence criterion is met. This typically occurs when the centroids no longer change their positions significantly between iterations, or when the assignments of data points to clusters stabilize.

### 3.2 Mathematical Formulation
Let the dataset be a set of points $X = \{x_1, x_2, ..., x_n\}$, where each $x_p$ is a vector in a $d$-dimensional space (in this case, $d=3$ for RGB). The objective of K-Means is to partition the $n$ points into $K$ sets $S = \{S_1, S_2, ..., S_K\}$ so as to minimize the **within-cluster sum of squares (WCSS)**, also known as inertia.

The objective function $J$ is given by:
$$J = \sum_{i=1}^{K} \sum_{x \in S_i} ||x - \mu_i||^2$$
where:
* $K$ is the number of clusters.
* $S_i$ is the set of all points belonging to cluster $i$.
* $\mu_i$ is the centroid (mean vector) of cluster $S_i$.
* $||x - \mu_i||^2$ is the squared Euclidean distance between a point $x$ and its cluster's centroid $\mu_i$.

The iterative process aims to find the optimal centroids $\mu_i$ and assignments $S_i$ that minimize this function $J$.

* **Assignment Step (Mathematically):** In iteration $t$, each point $x_p$ is assigned to cluster $S_i$ such that:
    $$S_i^{(t)} = \{ x_p : ||x_p - \mu_i^{(t)}||^2 \le ||x_p - \mu_j^{(t)}||^2 \quad \forall j, 1 \le j \le K \}$$

* **Update Step (Mathematically):** The new centroid for the next iteration, $t+1$, is calculated as the mean of all points assigned to the cluster in the current iteration:
    $$\mu_i^{(t+1)} = \frac{1}{|S_i^{(t)}|} \sum_{x_p \in S_i^{(t)}} x_p$$

### 3.3 Application in the Script
* **Data Points:** Each pixel of the input image is treated as a single data point in a 3-dimensional RGB color space. For a pixel with color $(R, G, B)$, its corresponding data point is the vector $x = [R, G, B]$.
* **Number of Clusters (K):** This is specified by the user via the `num_colors_slider`. It determines how many dominant colors the algorithm will find.
* **Centroids:** The final cluster centroids, $\mu_1, \mu_2, ..., \mu_K$, are the resulting dominant colors. Each centroid is a 3-dimensional vector representing the average RGB value for one of the identified color groups in the image.

---

## 4.0 Conclusion

The project is a well-structured and effective implementation for extracting color palettes from images. It correctly applies the principles of K-Means clustering by treating pixels as data points in a 3D color space. The pre-processing step of resizing the image is an important and practical optimization for performance. The separation of concerns between the core logic (`extract_palette`) and visualization (`create_palette_image`) demonstrates good software design. Overall, the project serves as an excellent practical demonstration of applying an unsupervised machine learning algorithm to solve a common problem in digital art and computer graphics.