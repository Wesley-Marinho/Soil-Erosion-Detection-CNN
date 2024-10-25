from PIL import Image
import os

# Diretório de entrada com imagens .tif
input_directory = "/media/wesley/novo_volume/dataset_v0/19.393136_44.380353/files_tiff"

# Diretório de saída para imagens .png
output_directory = (
    "/media/wesley/novo_volume/dataset_v0/19.393136_44.380353/files_tiff/cut"
)

cont = 774

# Lista todos os arquivos no diretório de entrada
file_list = os.listdir(input_directory)
file_list.sort()
# Loop através dos arquivos e converta .tif em .png
for filename in file_list:
    if filename.endswith(".tif"):  # modificar para o tipo da imagem
        # Abra a imagem .tif
        imagem_original = Image.open(os.path.join(input_directory, filename))
        # imagem_original = imagem_original.convert('L') # USAR SOMENTE PARA AS MÁSCARAS
        # Obtenha as dimensões da imagem original
        width, height = imagem_original.size

        # Defina as dimensões das imagens menores
        largura_imagem_menor = 512
        altura_imagem_menor = 512

        # Divida a imagem original em 16 imagens menores
        imagens_menores = []
        for y in range(0, 2897, altura_imagem_menor):
            for x in range(0, 8251, largura_imagem_menor):
                regiao = (x, y, x + largura_imagem_menor, y + altura_imagem_menor)
                imagem_menor = imagem_original.crop(regiao)
                imagens_menores.append(imagem_menor)

            # Salve as 16 imagens menores
            for i, imagem_menor in enumerate(imagens_menores):

                imagem_menor.save(
                    f"/media/wesley/novo_volume/dataset_v0/19.393136_44.380353/files_tiff/cut/{i+cont}.png"
                )

    # cont=cont+16

print("Imagens de 400x400 criadas e salvas com sucesso.")
