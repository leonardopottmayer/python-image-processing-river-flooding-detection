import cv2 
import os
import numpy as np
import matplotlib.pyplot as plt

# Função para exibir uma imagem com título.
def show_image(title, img, debugFileName):
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    # plt.axis('off')  # Remover eixos.
    plt.show()
    
    # Salvar a imagem no diretório images/debug.
    cv2.imwrite(f'images/debug/{debugFileName}', img)

# ----- LIMPAR O DIRETÓRIO DE DEBUG -----
debug_dir = 'images/debug'

# Verifica se a pasta de debug existe.
if not os.path.exists(debug_dir):
    os.makedirs(debug_dir)

# Apagar todos os arquivos da pasta images/debug.
for f in os.listdir(debug_dir):
    os.remove(os.path.join(debug_dir, f))

# ----- PREPARAÇÃO DAS IMAGENS -----

# Ler imagens do período chuvoso e de estiagem.
# Lê as imagens já e escala de cinza.
image_rainy = cv2.imread('images/raw/chuvoso.png', cv2.IMREAD_GRAYSCALE)
image_dry = cv2.imread('images/raw/estiagem.png', cv2.IMREAD_GRAYSCALE)

show_image('Imagem período chuvoso', image_rainy, '1 - Imagem período chuvoso.png')
show_image('Imagem período de estiagem', image_dry, '2 - Imagem período de estiagem.png')

# Redimensionar imagens para ambas terem o mesmo tamanho.
# Aqui redimensionamos a imagem do período de estiagem para o mesmo tamanho da imagem do período chuvoso.
height, width = image_rainy.shape[:2]
image_dry = cv2.resize(image_dry, (width, height))

# ----- PROCESSAMENTO DA IMAGEM -----

# Passo 1: Imagem negativa em tons de cinza da imagem processada.
# Este passo inverte os valores de brilho, transformando pixels pretos em brancos e pixels brancos em pretos.
image_rainy_neg = cv2.bitwise_not(image_rainy)
image_dry_neg = cv2.bitwise_not(image_dry)

# Passo 2: Pré-Processamento Morfológico (Etapa de realce).
# Aqui criamos um kernel 5x5 (uma matriz de 1's) para realizar as operações morfológicas.
kernel = np.ones((5, 5), np.uint8)

# Aplicar a dilatação para aumentar as áreas brancas na imagem, expandindo os contornos das feições.
dilated_rainy = cv2.dilate(image_rainy_neg, kernel, iterations=1)

# Aplicar a Erosão para reduzir as áreas brancas, diminuindo os contornos das feições.
eroded_dry = cv2.erode(image_dry_neg, kernel, iterations=1)

show_image('Imagem período chuvoso - Dilatada', dilated_rainy, '3 - Imagem período chuvoso - Dilatada.png')
show_image('Imagem período de estiagem - Erodida', eroded_dry, '4 - Imagem período de estiagem - Erodida.png')

# Passo 3: Binarização.
# Converte as imagens para binárias: aqui, os pixels abaixo de thresh ficam pretos e acima de maxval ficam brancos.
# Aqui é necessário ajustar o thresh e maxval para melhorar o resultado.

binary_rainy_thresh = 140
binary_rainy_maxval = 255

binary_dry_thresh = 80
binary_dry_maxval = 255

_, binary_rainy = cv2.threshold(dilated_rainy, binary_rainy_thresh, binary_rainy_maxval, cv2.THRESH_BINARY)
_, binary_dry = cv2.threshold(eroded_dry, binary_dry_thresh, binary_dry_maxval, cv2.THRESH_BINARY)

show_image('Imagem período chuvoso - Binarizada', binary_rainy, '5 - Imagem período chuvoso - Binarizada.png')
show_image('Imagem período de estiagem - Binarizada', binary_dry, '6 - Imagem período de estiagem - Binarizada.png')

# Passo 4: Pós-Processamento Morfológico.
# Remover pequenos objetos isolados (ruídos) na imagem binária.
clean_rainy = cv2.morphologyEx(binary_rainy, cv2.MORPH_OPEN, kernel)
clean_dry = cv2.morphologyEx(binary_dry, cv2.MORPH_OPEN, kernel)

show_image('Imagem período chuvoso - Pós-processada', clean_rainy, '7 - Imagem período chuvoso - Pós-processada.png')
show_image('Imagem período de estiagem - Pós-processada', clean_dry, '8 - Imagem período de estiagem - Pós-processada.png')

# Passo 5: Subtração das imagens.
# Aqui subtraimos a imagem do período seco da imagem do período chuvoso (chuvoso - seco) para identificar inundações.
flooded_areas = cv2.subtract(clean_rainy, clean_dry)

show_image('Áreas Inundadas', flooded_areas, '9 - Áreas Inundadas.png')

# Extração das Áreas Inundadas.
# Conta o número de pixels brancos (valor 255) que representam as áreas inundadas.
flooded_area_pixels = np.sum(flooded_areas == 255)
print(f'Área inundada em pixels: {flooded_area_pixels}')