# FURB 2024.2 - Processamento de Imagens - Trabalho 1 - Operadores Morfológicos
# Alunos: Leonardo Gian Pottmayer e Rael dos Santos Nehring

import cv2 
import os
import matplotlib.pyplot as plt

# Função para exibir uma imagem com título.
def show_image(title, img, debugFileName):
    plt.figure("FURB 2024.2 - Processamento de Imagens - Trabalho 1 - Operadores Morfológicos")
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
    
    cv2.imwrite(f'images/debug/{debugFileName}', img)

# Função para limpar o diretório images/debug.
def clear_debug_folder():
    debug_dir = 'images/debug'

    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    for f in os.listdir(debug_dir):
        os.remove(os.path.join(debug_dir, f))