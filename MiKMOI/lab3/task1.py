import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
image_path = 'task_1_lowpass.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Проверка
if img is None:
    raise FileNotFoundError(f"Ошибка: Не удалось загрузить изображение по пути '{image_path}'. Проверьте имя файла и путь к нему.")

img = img.astype(np.float32)
rows, cols = img.shape

# Центрирование спектра (умножение на (-1)^(x+y))
u = np.arange(rows)
v = np.arange(cols)
U, V = np.meshgrid(u, v, indexing='ij')
centered_img = img * ((-1) ** (U + V))

# Прямое двумерное БПФ
f_transform = np.fft.fft2(centered_img)

# Функция создания фильтра Баттерворта НЧ
def butterworth_lowpass(shape, D0, n):
    rows, cols = shape
    center_r, center_c = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u, v, indexing='ij')
    
    # Расстояние от каждой точки до центра
    D = np.sqrt((U - center_r) ** 2 + (V - center_c) ** 2)
    
    # Передаточная функция фильтра Баттерворта
    H = 1.0 / (1.0 + (D / D0) ** (2 * n))
    return H

# Параметры из задания
D0 = 30
orders = [1, 2, 4, 10] # Исследуем влияние порядка

plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Исходное изображение')
plt.axis('off')

for i, n in enumerate(orders):
    # Создаем фильтр
    H = butterworth_lowpass((rows, cols), D0, n)
    
    # Применение фильтра
    filtered_f = f_transform * H
    
    # Обратное БПФ
    inv_fft = np.fft.ifft2(filtered_f)
    
    # Убираем центрирование и берем вещественную часть
    result = np.real(inv_fft) * ((-1) ** (U + V))
    
    # Нормализация для отображения
    result = np.clip(result, 0, 255)
    
    plt.subplot(2, 3, i + 2)
    plt.imshow(result, cmap='gray')
    plt.title(f'Баттерворт НЧ (n={n}, D0={D0})')
    plt.axis('off')

plt.tight_layout()
plt.show()
