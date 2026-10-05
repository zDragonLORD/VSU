import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
image_path = 'task_2_highpass.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

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

# Функция создания фильтра Баттерворта ВЧ (HPF)
def butterworth_highpass(shape, D0, n):
    rows, cols = shape
    center_r, center_c = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u, v, indexing='ij')
    
    # Расстояние от центра спектра
    D = np.sqrt((U - center_r) ** 2 + (V - center_c) ** 2)
    
    # Избегаем деления на ноль в центре спектра (D = 0)
    D = np.where(D == 0, 1e-10, D)
    
    # Передаточная функция фильтра Баттерворта ВЧ
    H = 1.0 / (1.0 + (D0 / D) ** (2 * n))
    return H

# Параметры из задания
D0 = 30
orders = [1, 2, 4, 10]

plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Исходное изображение')
plt.axis('off')

for i, n in enumerate(orders):
    # Создаем фильтр ВЧ для текущего порядка n
    H = butterworth_highpass((rows, cols), D0, n)
    
    # Применение фильтра в частотной области
    filtered_f = f_transform * H
    
    # Обратное БПФ
    inv_fft = np.fft.ifft2(filtered_f)
    result = np.real(inv_fft) * ((-1) ** (U + V))
    
    # Нормализация
    result = cv2.normalize(result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    result = np.clip(result, 0, 255)
    plt.subplot(2, 3, i + 2)
    plt.imshow(result, cmap='gray')
    plt.title(f'Баттерворт ВЧ (n={n}, D0={D0})')
    plt.axis('off')

plt.tight_layout()
plt.show()
