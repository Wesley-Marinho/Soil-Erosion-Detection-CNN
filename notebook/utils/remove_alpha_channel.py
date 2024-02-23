from PIL import Image
import os

def remove_alpha_channel(image_path, output_path):
    try:
        # Abrindo a imagem
        img = Image.open(image_path)

        # Verificando se a imagem é PNG e possui 4 canais
        if img.mode == 'RGBA':
            # Removendo o quarto canal (canal alfa)
            img = img.convert('RGB')

            # Salvando a imagem sem o canal alfa
            img.save(output_path)
            print(f"Canal alfa removido de {image_path} e a imagem foi salva em {output_path}")
        else:
            print(f"A imagem {image_path} não possui canal alfa. Ignorando.")

    except Exception as e:
        print(f"Erro ao processar a imagem {image_path}: {e}")

def processar_pasta(pasta_entrada, pasta_saida):
    # Criando pasta de saída se não existir
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)

    # Percorrendo todos os arquivos na pasta de entrada
    for filename in os.listdir(pasta_entrada):
        if filename.endswith('.png'):
            # Caminho completo para o arquivo de entrada
            input_path = os.path.join(pasta_entrada, filename)
            # Caminho completo para o arquivo de saída
            output_path = os.path.join(pasta_saida, filename)
            # Removendo o canal alfa
            remove_alpha_channel(input_path, output_path)

# Pasta de entrada e pasta de saída
pasta_entrada = r'C:\Users\wesleymarinho\Documents\GitHub\Mestrado\test\images'
pasta_saida = r'C:\Users\wesleymarinho\Documents\saida'

# Processamento da pasta
processar_pasta(pasta_entrada, pasta_saida)