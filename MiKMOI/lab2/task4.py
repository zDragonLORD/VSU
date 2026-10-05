import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Путь к изображению
image_path = 'variant_4_4.jpg'

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

# Сглаживающий фильтр (Фильтр Гаусса)
# Убираем шум, размер ядра
kernel_size_smooth = 15
img_smoothed = cv2.GaussianBlur(
    image, (kernel_size_smooth, kernel_size_smooth), sigmaX=0
)

# Повышение резкости с помощью Лапласиана
# Применяем Лапласиан для выделения вторых производных
laplacian = cv2.Laplacian(img_smoothed, cv2.CV_64F, ksize=3)

# Нормализуем результат Лапласиана в диапазон 0-255
laplacian_u8 = cv2.normalize(
    np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
)

# Метод повышения резкости
# img_sharpened = img_smoothed - c * laplacian (c — коэффициент усиления)
img_float = img_smoothed.astype(np.float64)
lap_float = laplacian.astype(np.float64)
alpha = 5  # Коэффициент влияния резкости
img_sharpened_float = img_float - alpha * lap_float
img_sharpened = np.clip(img_sharpened_float, 0, 255).astype(np.uint8)

# Выделение краев оператором Собеля
sobel_x = cv2.Sobel(img_sharpened, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_sharpened, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = cv2.magnitude(sobel_x, sobel_y)

# Нормализуем модуль градиента Собеля
img_edges = cv2.normalize(
    sobel_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

# Отображаем каждый этап обработки
axes[0].imshow(image, cmap='gray')
axes[0].set_title('1. Оригинал')
axes[0].axis('off')

axes[1].imshow(img_smoothed, cmap='gray')
axes[1].set_title('2. Сглаживание\n(Gaussian)')
axes[1].axis('off')

axes[2].imshow(img_sharpened, cmap='gray')
axes[2].set_title('3. Повышение резкости\n(Laplacian)')
axes[2].axis('off')

axes[3].imshow(img_edges, cmap='gray')
axes[3].set_title('4. Выделение границ\n(Sobel)')
axes[3].axis('off')

plt.tight_layout()
plt.show()
