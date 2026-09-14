import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. Загрузка исходного изображения
image_path = '4.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
  raise FileNotFoundError(
      f"Ошибка: не удалось загрузить изображение"
  )

# Функция для извлечения битовых плоскостей и реконструкции из младших (0, 1, 2, 3)
def reconstruct_lower_planes(image):
  mask = 0x0F
  reconstructed = image & mask
  reconstructed_visual = cv2.normalize(
      reconstructed, None, 0, 255, cv2.NORM_MINMAX
  )
  return reconstructed, reconstructed_visual

img_lower_raw, img_lower_vis = reconstruct_lower_planes(img)

# 2. Расчет количественных метрик
def calculate_metrics(orig, recon):
  # MSE (Mean Squared Error)
  mse = np.mean((orig.astype(np.float32) - recon.astype(np.float32)) ** 2)

  # PSNR (Peak Signal-to-Noise Ratio)
  if mse == 0:
    psnr = float('inf')
  else:
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

  # Энтропия изображения 
  def image_entropy(image):
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32)
    total_pixels = image.size
    # Исключаем нулевые вероятности для избежания логарифма от нуля
    probabilities = hist[hist > 0] / total_pixels
    return -np.sum(probabilities * np.log2(probabilities))

  orig_entropy = image_entropy(orig)
  recon_entropy = image_entropy(recon)

  # Коэффициент сжатия
  compression_ratio = 8.0 / 4.0

  return mse, psnr, orig_entropy, recon_entropy, compression_ratio

mse_val, psnr_val, ent_orig, ent_recon, comp_ratio = calculate_metrics(img, img_lower_raw)

print('--- Количественные метрики ---')
print(f'MSE: {mse_val:.2f}')
print(f'PSNR: {psnr_val:.2f} дБ')
print(f'Энтропия исходного изображения: {ent_orig:.2f} бит/пиксель')
print(
    f'Энтропия реконструированного изображения (младшие биты):'
    f' {ent_recon:.2f} бит/пиксель'
)
print(f'Коэффициент сжатия: {comp_ratio:.1f}:1')

# 3. Визуализация (Изображения и гистограммы)
plt.figure(figsize=(14, 10))

# Исходное изображение
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('1. Исходное изображение')
plt.axis('off')

# Гистограмма исходного
plt.subplot(2, 2, 2)
plt.hist(img.ravel(), bins=256, range=[0, 256], color='gray', edgecolor='black')
plt.title('Гистограмма исходного')
plt.xlabel('Яркость')
plt.ylabel('Частота')

# Реконструированное из младших плоскостей (визуальное)
plt.subplot(2, 2, 3)
plt.imshow(img_lower_vis, cmap='gray', vmin=0, vmax=255)
plt.title('2. Реконструкция из младших плоскостей (0-3)')
plt.axis('off')

# Гистограмма реконструированного
plt.subplot(2, 2, 4)
plt.hist(
    img_lower_raw.ravel(),
    bins=16,
    range=[0, 16],
    color='gray',
    edgecolor='black',
)
plt.title('Гистограмма младших плоскостей (0-3)')
plt.xlabel('Значение (0-15)')
plt.ylabel('Частота')

plt.tight_layout()
plt.show()

