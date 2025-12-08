import cv2
import matplotlib.pyplot as plt
import numpy as np


def process_image(image_path):
    # Leitura da imagem
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicação de filtro para redução de ruído
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Limiarização
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imshow('Thresholded Image', thresh)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Criar lista para armazenar áreas dos núcleos
    new_nuclei_areas = []

    # Analisar contornos
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:  # Considerar apenas áreas significativas
            new_nuclei_areas.append(area)

    return new_nuclei_areas, len(contours)


def plot_statistics(areas_list):
    # Calcular estatísticas
    means = [np.mean(areas) for areas in areas_list]
    medians = [np.median(areas) for areas in areas_list]
    std_devs = [np.std(areas) for areas in areas_list]
    counts = [len(areas) for areas in areas_list]

    # Plotar estatísticas
    time_points = range(len(areas_list))

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(time_points, means, label='Média')
    plt.xlabel('Tempo')
    plt.ylabel('Área Média dos Núcleos')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(time_points, medians, label='Mediana')
    plt.xlabel('Tempo')
    plt.ylabel('Área Mediana dos Núcleos')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(time_points, std_devs, label='Desvio Padrão')
    plt.xlabel('Tempo')
    plt.ylabel('Desvio Padrão da Área dos Núcleos')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(time_points, counts, label='Número de Núcleos')
    plt.xlabel('Tempo')
    plt.ylabel('Contagem de Núcleos')
    plt.legend()

    plt.tight_layout()
    plt.show()
    

# Lista de caminhos das imagens em diferentes tempos
image_paths = ["images/Normal.jpg", "images/21h.jpg", "images/29h.jpg", "images/44h.jpg", "images/53h.jpg", "images/73h.jpg", "images/96h.jpg"]

# Processar cada imagem
nuclei_areas_list = []
for path in image_paths:
    nuclei_areas, _ = process_image(path)
    nuclei_areas_list.append(nuclei_areas)

# Plotar as estatísticas
plot_statistics(nuclei_areas_list)
