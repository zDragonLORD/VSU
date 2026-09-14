import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. Загрузка исходного изображения
image_path = '2_task.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
  raise FileNotFoundError(
      f"Ошибка: не удалось загрузить изображение."
  )

# 2. Реализация кусочно-линейной функции преобразования яркости
def piecewise_linear_transform_v4(image):
  img_f = image.astype(np.float32)
  out = np.zeros_like(img_f)

  # Условия кусочно-линейной функции:
  mask1 = img_f < 40
  out[mask1] = 255
  mask2 = (img_f >= 40) & (img_f < 80)
  out[mask2] = 255 - 255 * (img_f[mask2] - 40) / 40
  mask3 = img_f >= 80
  out[mask3] = 0

  return out.astype(np.uint8)

img_transformed = piecewise_linear_transform_v4(img)

# 3. Построение графика кусочно-линейной функции преобразования
x_vals = np.arange(0, 256)
y_vals = np.zeros_like(x_vals, dtype=np.float32)

# Расчет графика для визуализации
for i, x in enumerate(x_vals):
  if x < 40:
    y_vals[i] = 255
  elif 40 <= x < 80:
    y_vals[i] = 255 - 255 * (x - 40) / 40
  else:
    y_vals[i] = 0

# 4. Визуализация результатов (Изображения + График функции)
plt.figure(figsize=(15, 5))

# Исходное изображение
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('1. Исходный снимок')
plt.axis('off')

# График функции преобразования
plt.subplot(1, 3, 2)
plt.plot(x_vals, y_vals, color='blue', linewidth=2)
plt.title('3. График кусочно-линейной функции')
plt.xlabel('Входная яркость (x)')
plt.ylabel('Выходная яркость f(x)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim([0, 255])
plt.ylim([-10, 265])

# Преобразованное изображение
plt.subplot(1, 3, 3)
plt.imshow(img_transformed, cmap='gray', vmin=0, vmax=255)
plt.title('2. Результат')
plt.axis('off')

plt.tight_layout()
plt.show()

