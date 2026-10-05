import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
image_path = 'task_4_homomorphic.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Ошибка: Не удалось загрузить изображение по пути '{image_path}'. Проверьте имя файла и путь к нему.")

# Переводим в float и добавляем 1, чтобы избежать ln(0)
img_float = img.astype(np.float32) + 1.0
rows, cols = img_float.shape

# Логарифмирование изображения
img_log = np.log(img_float)

# Прямое двумерное БПФ и сдвиг
f_transform = np.fft.fft2(img_log)
f_shifted = np.fft.fftshift(f_transform)

# Функция создания гомоморфного фильтра
def homomorphic_filter(shape, D0, H_H, H_L, c=1, n=1):
    rows, cols = shape
    center_r, center_c = rows // 2, cols // 2
    
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u, v, indexing='ij')
    
    # Расстояние до центра спектра
    D = np.sqrt((U - center_r) ** 2 + (V - center_c) ** 2)
    
    # Формула гомоморфного фильтра
    H = (H_H - H_L) * (1.0 - np.exp(-c * (D / D0) ** (2 * n))) + H_L
    return H

# Базовые параметры из задания
H_L = 0.5
H_H = 1.5
d0_values = [10, 40, 80] # Значения D0

plt.figure(figsize=(16, 5))
plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title('Исходное изображение')
plt.axis('off')

for i, D0 in enumerate(d0_values):
    # Создание фильтра
    H = homomorphic_filter((rows, cols), D0, H_H, H_L)
    
    # Фильтрация в частотной области
    filtered_shifted = f_shifted * H
    
    # Обратный сдвиг и обратное БПФ
    f_ishift = np.fft.ifftshift(filtered_shifted)
    inv_fft = np.real(np.fft.ifft2(f_ishift))
    
    # Экспоненциальное преобразование
    result = np.exp(inv_fft) - 1.0
    
    # Нормализация
    result = cv2.normalize(result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    plt.subplot(1, 4, i + 2)
    plt.imshow(result, cmap='gray')
    plt.title(f'Гомоморфный (D0={D0})\nHL={H_L}, HH={H_H}')
    plt.axis('off')

plt.tight_layout()
plt.show()
