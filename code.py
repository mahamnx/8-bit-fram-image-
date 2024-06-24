import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

def open_image():
    filepath = filedialog.askopenfilename()
    if filepath:
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is None:
            messagebox.showerror("Error", "Unable to open image file!")
            return
        display_image(image, "Original Grayscale Image")
        return image
    return None

def display_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def split_into_bit_planes(image):
    bit_planes = [(image >> i) & 1 for i in range(8)]
    return bit_planes

def display_bit_planes(bit_planes):
    fig, axes = plt.subplots(1, 8, figsize=(20, 3))
    for i in range(8):
        axes[i].imshow(bit_planes[i], cmap='gray')
        axes[i].set_title(f'Bit Plane {i}')
        axes[i].axis('off')
    plt.show()

def reconstruct_from_bit_planes(bit_planes):
    reconstructed = sum(plane << i for i, plane in enumerate(bit_planes))
    return reconstructed

def memory_reduction_analysis(original_image, bit_planes):
    original_size = original_image.nbytes
    bit_planes_size = original_image.size//8
    reduction_percentage = (original_size - bit_planes_size) / original_size * 100
    print(f'Original Image Size: {original_size} bytes')
    print(f'Bit Planes Size: {bit_planes_size} bytes')
    print(f'Memory Reduction: {reduction_percentage:.2f}%')

def thresholding(image):
    _, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return thresholded

def edge_detection(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

def compare_methods(image):
    thresholded = thresholding(image)
    edges = edge_detection(image)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(thresholded, cmap='gray')
    plt.title('Thresholding')

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection')
    plt.show()

def main():
    # Create a simple GUI for image selection
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    messagebox.showinfo("Information", "Select a grayscale image file.")
    grayscale_image = open_image()

    if grayscale_image is not None:
        # Split grayscale image into bit planes
        bit_planes = split_into_bit_planes(grayscale_image)
        display_bit_planes(bit_planes)

        # Reconstruct the grayscale image from bit planes
        reconstructed_image = reconstruct_from_bit_planes(bit_planes)
        display_image(reconstructed_image, "Reconstructed Grayscale Image")

        # Memory reduction analysis
        memory_reduction_analysis(grayscale_image, bit_planes)

        # Compare bit plane technique with other methods
        compare_methods(grayscale_image)

if __name__ == "__main__":
    main()