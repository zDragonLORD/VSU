import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Путь к изображению
image_path = 'variant_4_2.jpg'

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

# Размер ядра фильтра
kernel_size = 15

# Обычный однородный усредняющий фильтр
# Все коэффициенты матрицы равны 1 / (kernel_size * kernel_size)
box_kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (
    kernel_size * kernel_size
)
img_box = cv2.filter2D(image, -1, box_kernel)

# Взвешенный усредняющий фильтр (Фильтр Гаусса)
# Веса распределены по закону Гаусса (в центре максимальные, к краям убывают)
img_weighted = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)

# Вырезаем фрагмент для сравнения краев
h, w = image.shape
crop_size = min(120, h, w)
crop_slice = (slice(0, crop_size), slice(0, crop_size))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Изображения целиком
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Оригинал\n(целиком)')
axes[0, 0].axis('off')

axes[0, 1].imshow(img_box, cmap='gray')
axes[0, 1].set_title(
    f'Однородный фильтр (Box)\nразмер ядра: {kernel_size}x{kernel_size}'
)
axes[0, 1].axis('off')

axes[0, 2].imshow(img_weighted, cmap='gray')
axes[0, 2].set_title(
    f'Взвешенный фильтр (Gaussian)\nразмер ядра: {kernel_size}x{kernel_size}'
)
axes[0, 2].axis('off')

# Крупный план (фрагмент края/угла)
axes[1, 0].imshow(image[crop_slice], cmap='gray')
axes[1, 0].set_title('Оригинал\n(фрагмент)')
axes[1, 0].axis('off')

axes[1, 1].imshow(img_box[crop_slice], cmap='gray')
axes[1, 1].set_title('Однородный фильтр\n(фрагмент)')
axes[1, 1].axis('off')

axes[1, 2].imshow(img_weighted[crop_slice], cmap='gray')
axes[1, 2].set_title('Взвешенный фильтр\n(фрагмент)')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

