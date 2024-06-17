import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from scipy.ndimage import zoom

def generate_laplacian_pyramid(image, levels):
    """
    Generates a Laplacian pyramid for a given image.

    Parameters:
    image (ndarray): The input image.
    levels (int): Number of levels in the pyramid.

    Returns:
    list: A list containing the Laplacian pyramid images.
    """
    gaussian_pyramid = [image]
    for i in range(levels):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)
    
    W,H = gaussian_pyramid[0].shape[1], gaussian_pyramid[0].shape[0]
    laplacian_pyramid = [cv2.resize(image,(W,H))]
    for i in range(levels, 0, -1):
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i])
        if gaussian_expanded.shape != gaussian_pyramid[i-1].shape:
            gaussian_expanded = cv2.resize(gaussian_expanded, (gaussian_pyramid[i-1].shape[1], gaussian_pyramid[i-1].shape[0]))
        laplacian = gaussian_pyramid[i-1].copy()
        laplacian[...,:3] = cv2.subtract(laplacian[...,:3], gaussian_expanded[...,:3])
        laplacian_pyramid.append(cv2.resize(laplacian,(W,H)))
    
    return laplacian_pyramid

def process_images(input_dir, output_dir, levels=3):
    """
    Processes images from the input directory to create and save their Laplacian pyramids.

    Parameters:
    input_dir (str): Path to the input directory containing images.
    output_dir (str): Path to the output directory where results will be saved.
    levels (int): Number of levels in the Laplacian pyramid.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            # image = cv2.imread(image_path)
            if image is None:
                continue

            laplacian_pyramid = generate_laplacian_pyramid(image, levels-1)

            base_name, ext = os.path.splitext(filename)
            pyramid_dir = os.path.join(output_dir, base_name)
            if not os.path.exists(pyramid_dir):
                os.makedirs(pyramid_dir)

            for i, laplacian in enumerate(laplacian_pyramid):
                laplacian_filename = os.path.join(pyramid_dir, f'laplacian_level_{i}.png')
                cv2.imwrite(laplacian_filename, laplacian)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Laplacian Pyramid for images in a directory.')
    parser.add_argument('--input_dir', type=str, help='Path to the input directory containing images.')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory where results will be saved.')
    parser.add_argument('--levels', type=int, default=3, help='Number of levels in the Laplacian pyramid (default is 3).')

    args = parser.parse_args()

    process_images(args.input_dir, args.output_dir, args.levels)
