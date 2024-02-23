#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:37:25 2023

@author: isi-er
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:31:50 2023

@author: isi-er
"""
import os
import sys
import numpy as np
import torch
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from skimage.transform import resize
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# Diretório de entrada com imagens .tif
input_directory = "/home/isi-er/Documentos/DATASET_2/MASCARA"

# Diretório de saída para imagens .png
output_directory = "/home/isi-er/Documentos/DATASET_PNG/groundtruth"

cont=774

# Lista todos os arquivos no diretório de entrada
file_list = os.listdir(input_directory)
file_list.sort()
# Loop através dos arquivos e converta .tif em .png
for filename in file_list:
    if filename.endswith(".tif"):
        # Abra a imagem .tif
        imagem_original = np.array(Image.open(os.path.join(input_directory, filename))).astype('float32') / 255
        #binarizar a imagem
        gray_mask = imagem_original[:, :, 0]  # Canal vermelho

        # Normalize a imagem para o intervalo [0, 255]
        gray_mask = (gray_mask / np.max(gray_mask) * 255).astype('uint8')

        # Crie uma imagem em escala de cinza a partir do canal selecionado
        output_image = cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2BGR)
        gray_image = output_image[:, :, 0]

        image = Image.fromarray(output_image)
        image = image.convert('L')
        
            
        # Obtenha as dimensões da imagem original
        width, height = image.size

        # Defina as dimensões das imagens menores
        largura_imagem_menor = 400
        altura_imagem_menor = 400

        # Divida a imagem original em 16 imagens menores
        imagens_menores = []
        for y in range(0, 2897, altura_imagem_menor):
            for x in range(0, 8251, largura_imagem_menor):
                regiao = (x, y, x + largura_imagem_menor, y + altura_imagem_menor)
                imagem_menor = image.crop(regiao)
                imagens_menores.append(imagem_menor)

            # Salve as 16 imagens menores
            for i, imagem_menor in enumerate(imagens_menores):
                
                imagem_menor.save(f"/home/isi-er/Documentos/DATASET_PNG/groundtruth/{i+cont}.png")
                
    #cont=cont+16

print("Imagens de 400x400 criadas e salvas com sucesso.")



# mask = np.array(Image.open(f'/home/isi-er/Documentos/DATASET_2/groundtruth/04_2023_resample_0_0.tif')).astype('float32') / 255

# gray_mask = imagem_original[:, :, 0]  # Canal vermelho

# # Normalize a imagem para o intervalo [0, 255]
# gray_mask = (gray_mask / np.max(gray_mask) * 255).astype('uint8')

# # Crie uma imagem em escala de cinza a partir do canal selecionado
# output_image = cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2BGR)
# gray_image = output_image[:, :, 0]

# image = Image.fromarray(output_image)
# image = image.convert('L')

# fig, ax = plt.subplots(1,2, figsize=(12,6))
# ax[0].imshow(mask);
# ax[0].set_title('Original Image')
# ax[1].imshow(image)
# ax[1].set_title('Binarized Image')

# image.save("/home/isi-er/Documentos/DATASET_2/groundtruth/teste1.png")

