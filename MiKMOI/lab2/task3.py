import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Путь к изображению
image_path = 'variant_4_3.jpg'

# Проверяем наличие файла
if not os.path.exists(image_path):
  raise FileNotFoundError(
      f"Файл '{image_path}' не найден! Проверьте путь к изображению."
  )

# Загрузка полутонового изображения
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
  raise ValueError(
      f"Не удалось загрузить изображение из файла '{image_path}'."
  )

# Оператор Собеля с пороговым преобразованием
# Вычисляем производные по X и Y с помощью встроенной функции Собеля (размер ядра 3x3)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Вычисляем модуль градиента: M = sqrt(Gx^2 + Gy^2)
sobel_mag = cv2.magnitude(sobel_x, sobel_y)

# Нормализуем в диапазон 0-255 для удобства применения порога
sobel_mag_u8 = cv2.normalize(
    sobel_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
)

# Пороговое преобразование для Собеля
# Пиксели выше порога станут белыми (255), ниже — черными (0)
threshold_value = 70
_, sobel_thresh = cv2.threshold(
    sobel_mag_u8, threshold_value, 255, cv2.THRESH_BINARY
)

# Простой градиент с пороговым преобразованием
# Простые ядра для поиска разностей соседних пикселей
simple_kernel_x = np.array([[-1, 1]], dtype=np.float32)
simple_kernel_y = np.array([[-1], [1]], dtype=np.float32)

simple_x = cv2.filter2D(image, cv2.CV_64F, simple_kernel_x)
simple_y = cv2.filter2D(image, cv2.CV_64F, simple_kernel_y)

# Модуль простого градиента
simple_mag = cv2.magnitude(simple_x, simple_y)
simple_mag_u8 = cv2.normalize(
    simple_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
)

# Пороговое преобразование для простого градиента
_, simple_thresh = cv2.threshold(
    simple_mag_u8, threshold_value, 255, cv2.THRESH_BINARY
)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Исходное изображение и модули градиентов
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('1. Оригинал')
axes[0, 0].axis('off')

axes[0, 1].imshow(sobel_mag_u8, cmap='gray')
axes[0, 1].set_title('2. Модуль градиента (Собель)')
axes[0, 1].axis('off')

axes[0, 2].imshow(simple_mag_u8, cmap='gray')
axes[0, 2].set_title('3. Модуль градиента (Простой)')
axes[0, 2].axis('off')

# Результаты после порогового преобразования
axes[1, 0].imshow(image, cmap='gray')
axes[1, 0].set_title('Для сравнения (Оригинал)')
axes[1, 0].axis('off')

axes[1, 1].imshow(sobel_thresh, cmap='gray')
axes[1, 1].set_title(f'4. Собель + Порог (thresh = {threshold_value})')
axes[1, 1].axis('off')

axes[1, 2].imshow(simple_thresh, cmap='gray')
axes[1, 2].set_title(f'5. Простой градиент + Порог (thresh = {threshold_value})')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

