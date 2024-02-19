#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 14:50:18 2023

@author: devalisson
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:44:40 2023

@author: isi-er
"""
import cv2
import numpy as np
import torch
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt

def test (path_testing, index, model , cuda_available):
    
    model.eval()
    
    # The resolution of resized training images and the corresponding masks
    training_resize = 384
    # The number of resized training pairs used for data augmentation
    training_number = 400
    # The resolution of resized testing images
    testing_resize = int(608 * training_resize / 400)
    if testing_resize % 2 == 1:
        testing_resize += 1
    # Load a testing image
    #input = np.array(Image.open(f'{path_testing}/test_{index}/test_{index}.png')).astype('float32') / 255
    #input = np.array(Image.open(f'{path_testing}/images/({index}).png')).astype('float32') / 255
    input = np.array(Image.open("../../test/images/(23).png")).astype('float32') / 255
    # Resize the testing image
    input = resize(input, (testing_resize, testing_resize))

    # Divide the resized testing image into four patches, one at each corner.
    input_patches = testing_patch_extracting(input, training_resize, testing_resize)
    input_patches = torch.from_numpy(input_patches).float()

    # Predict the mask of the four patches
    if cuda_available:
        output_patches = model(input_patches.cuda()).detach().cpu().numpy()
    else:
        output_patches = model(input_patches).detach().numpy()

    # Merge the four masks into one resized mask
    output = testing_patch_assembling(output_patches, training_resize, testing_resize)[0, :, :]

    # Restore the resized mask to the original resolution
    output = resize(output, (608, 608))
      
    # Binarize the array
    threshold = 0.55
    binary = (output > threshold).astype(int)

    output_normalized = (output * 255).astype(np.uint8)

    # Crie uma imagem em escala de cinza a partir da matriz
    output_image = cv2.cvtColor(output_normalized, cv2.COLOR_GRAY2BGR)

    min_pixel_value = np.min(output_image)
    max_pixel_value = np.max(output_image)
    normalized_image = ((output_image - min_pixel_value) / (max_pixel_value - min_pixel_value) * 255).astype(np.uint8)
    
    input = resize(input, (608, 608))
    if binary.shape[:2] != input.shape[:2]:
        raise ValueError("As dimensões da imagem binária não correspondem às dimensões da imagem original.")

    # Crie uma máscara usando a imagem binária
    opacity = 0.002 
    # Crie uma máscara usando a imagem binária
    mask = np.zeros_like(input, dtype=input.dtype)
    mask[binary == 1] = (0,255,0)  # Pixels na imagem binária que são 1 ficarão totalmente brancos na máscara

    # Aplique a mistura alfa entre a imagem de entrada e a máscara
    highlighted_image = cv2.addWeighted(input, 1 - opacity, mask, opacity, 0)
    

    fig, ax = plt.subplots(1,3, figsize=(12,6))
    ax[0].imshow(output);
    ax[0].set_title('Prediction Result')
    ax[1].imshow(normalized_image)
    ax[1].set_title('Binarized Image')
    ax[2].imshow(highlighted_image)
    ax[2].set_title('Segmented Image')