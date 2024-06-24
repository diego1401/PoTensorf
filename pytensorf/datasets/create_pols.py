import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from scipy.ndimage import zoom

def up_sample_until_target(x,target):
    while x.shape != target.shape:
            x = cv2.pyrUp(x)
    return x

def generate_laplacian_pyramid(input_image, levels):
    """
    Generates a Laplacian pyramid for a given image.

    Parameters:
    image (ndarray): The input image.
    levels (int): Number of levels in the pyramid.

    Returns:
    list: A list containing the Laplacian pyramid images.
    # """

    # Construct Gaussian pyramid
    gaussian_pyramid = [input_image] # I_0
    for k in range(levels):
        I_k = cv2.pyrDown(gaussian_pyramid[-1])
        gaussian_pyramid.append(I_k)

    # Gaussian pyramid = [I_0,I_1, ..., I_K] -> K = levels + 1

    # Construct Pyramid Of Laplacians
    H,W = input_image.shape[0],input_image.shape[1]
    pyramid_of_laplacians = [gaussian_pyramid[-1]] # h_K = I_K
    
    for k in range(levels-1,-1,-1): #iterate over [I_{K-1},..., I_0]
        I_k = gaussian_pyramid[k]
        I_k_plus_1 = gaussian_pyramid[k+1]
        I_k = np.int32(up_sample_until_target(I_k,input_image))
        I_k_plus_1 = np.int32(up_sample_until_target(I_k_plus_1,input_image))
        h_k = I_k - I_k_plus_1
        pyramid_of_laplacians.append(h_k)

    # Up sampling
    for k in range(levels+1):
        while pyramid_of_laplacians[k].shape != input_image.shape:
            pyramid_of_laplacians[k] = cv2.pyrUp(pyramid_of_laplacians[k])
        pyramid_of_laplacians[k][...,3]= cv2.resize(input_image[...,3],(h_k.shape[0],h_k.shape[1]))
    return pyramid_of_laplacians

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
    
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                continue

            laplacian_pyramid = generate_laplacian_pyramid(image, levels-1)

            base_name, ext = os.path.splitext(filename)
            pyramid_dir = os.path.join(output_dir, base_name)
            if not os.path.exists(pyramid_dir):
                os.makedirs(pyramid_dir)

            for i, laplacian in enumerate(laplacian_pyramid):
                laplacian_filename = os.path.join(pyramid_dir, f'laplacian_level_{i}.exr')
                cv2.imwrite(laplacian_filename, laplacian.astype("float32")/255.0)

def example(levels):
    input_dir = 'data/nerf_synthetic/lego/train/r_44'
    result = None
    for i in range(levels):
        filename = f'laplacian_level_{i}.exr'
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if i == 0:
            result = image.copy()
        else:
            result[...,:3] = result[...,:3] + image[...,:3]
            
    print('result',result.min(),result.max())
    cv2.imwrite('example.png', result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Laplacian Pyramid for images in a directory.')
    parser.add_argument('--input_dir', type=str, help='Path to the input directory containing images.')
    parser.add_argument('--output_dir', type=str, help='Path to the output directory where results will be saved.')
    parser.add_argument('--levels', type=int, default=3, help='Number of levels in the Laplacian pyramid (default is 3).')

    args = parser.parse_args()

    process_images(args.input_dir, args.output_dir, args.levels)
    example(args.levels)
