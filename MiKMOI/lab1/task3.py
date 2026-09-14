import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. Загрузка исходного изображения
image_path = '3_task.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
  raise FileNotFoundError(
      f"Ошибка: не удалось загрузить изображение."
  )

# 2. Применение степенного преобразования (Gamma-коррекция) с γ > 1
gamma = 2.5

def power_law_transform(image, gamma_val):
  # Создаем таблицу перекодировки (LUT) для эффективного применения степенной функции
  # Формула: s = c * (r / 255) ^ gamma * 255, где c = 255
  table = np.array(
      [((i / 255.0) ** gamma_val) * 255 for i in np.arange(0, 256)]
  ).astype('uint8')
  return cv2.LUT(image, table)
img_processed = power_law_transform(img, gamma)

# 3. Визуализация результатов (Изображения и гистограммы)
plt.figure(figsize=(14, 10))

# Исходное изображение
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('1. Исходная термограмма')
plt.axis('off')

# Гистограмма исходного изображения
plt.subplot(2, 2, 2)
plt.hist(
    img.ravel(), bins=256, range=[0, 256], color='dimgray', edgecolor='black'
)
plt.title('Гистограмма исходного изображения')
plt.xlabel('Яркость пикселя')
plt.ylabel('Количество')

# Обработанное изображение
plt.subplot(2, 2, 3)
plt.imshow(img_processed, cmap='gray', vmin=0, vmax=255)
plt.title(f'2. Обработанное изображение (γ = {gamma})')
plt.axis('off')

# Гистограмма обработанного изображения
plt.subplot(2, 2, 4)
plt.hist(
    img_processed.ravel(),
    bins=256,
    range=[0, 256],
    color='dimgray',
    edgecolor='black',
)
plt.title('Гистограмма обработанного изображения')
plt.xlabel('Яркость пикселя')
plt.ylabel('Количество')

plt.tight_layout()
plt.show()

