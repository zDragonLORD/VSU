import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
image_path = 'task_3_periodic.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Ошибка: Не удалось загрузить изображение по пути '{image_path}'. Проверьте имя файла и путь к нему.")

img = img.astype(np.float32)
rows, cols = img.shape

# Прямое двумерное БПФ и сдвиг центра спектра
f_transform = np.fft.fft2(img)
f_shifted = np.fft.fftshift(f_transform)

# Автоматическое формирование массива координат шумовых пиков
mag = np.abs(f_shifted)
spec_log = np.log(1 + mag)

center_r, center_c = rows // 2, cols // 2
mask_dc_size = 55 # Размер зоны исключения центральной DC-компоненты
spec_log_no_dc = spec_log.copy()
spec_log_no_dc[center_r - mask_dc_size:center_r + mask_dc_size, 
               center_c - mask_dc_size:center_c + mask_dc_size] = 0

# Порог для выделения самых ярких пиков шума
threshold = np.percentile(spec_log_no_dc, 99.9)
peak_mask = (spec_log_no_dc > threshold).astype(np.uint8) * 255

# Поиск белых точек в спектре
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(peak_mask)

# Формируем массив координат (u0, v0) для каждой найденной точки
notch_coords = []
for i in range(1, num_labels):
    c_x, c_y = centroids[i]
    u0 = int(round(c_y - center_r))
    v0 = int(round(c_x - center_c))
    notch_coords.append((u0, v0))

print(f"Автоматически сформированный массив координат пиков: {notch_coords}")

D0 = 25  # Ширина полосы подавления

# Функция создания гауссовского режекторного фильтра
def gaussian_notch_filter_array(shape, notch_pairs, D0):
    rows, cols = shape
    center_r, center_c = rows // 2, cols // 2
    
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u, v, indexing='ij')
    
    # Инициализируем маску фильтра единицами
    H = np.ones((rows, cols), dtype=np.float32)
    
    # Проходим по каждой паре координат из массива и перемножаем фильтры
    for (u0, v0) in notch_pairs:
        D1 = np.sqrt((U - center_r - u0) ** 2 + (V - center_c - v0) ** 2)
        D2 = np.sqrt((U - center_r + u0) ** 2 + (V - center_c + v0) ** 2)
        
        D1 = np.where(D1 == 0, 1e-10, D1)
        D2 = np.where(D2 == 0, 1e-10, D2)
        
        H_k = 1.0 - np.exp(-0.5 * ((D1 * D2) / (D0 ** 2)))
        H *= H_k
        
    return H

# Если вдруг автоматика ничего не нашла, зададим безопасный запасной вариант
if not notch_coords:
    notch_coords = [(0, 0)]

# Создание итоговой маски фильтра по массиву координат
H = gaussian_notch_filter_array((rows, cols), notch_coords, D0)

# Применение фильтра на сдвинутом спектре
filtered_shifted = f_shifted * H

# Обратное преобразование Фурье
f_ishift = np.fft.ifftshift(filtered_shifted)
inv_fft = np.fft.ifft2(f_ishift)
result = np.real(inv_fft)
result = np.clip(result, 0, 255)

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Исходное (с шумом)')
plt.axis('off')

plt.subplot(2, 2, 2)
spec_orig = np.log(1 + np.abs(f_shifted))
plt.imshow(spec_orig, cmap='gray')
plt.title('Спектр (с авто-пиками)')
plt.axis('off')

plt.subplot(2, 2, 3)
spec_filtered = np.log(1 + np.abs(filtered_shifted))
plt.subplot(2, 2, 3)
plt.imshow(spec_filtered, cmap='gray')
plt.title('Спектр после фильтра (точки перекрыты)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(result, cmap='gray')
plt.title('Результат фильтрации')
plt.axis('off')

plt.tight_layout()
plt.show()
