# Математические и компьютерные методы обработки изображений

## Лабораторная работа № 2

#### Выполнил: Валиков Егор

#### Группа: 19

---

## Задание 1: Исследование методов обработки границ при фильтрации изображений

### Задача

Исследовать влияние различных методов обработки границ (краевых эффектов) при выполнении линейной фильтрации (свертки) изображения. Сравнить результаты для методов: заполнение константой (`cv2.BORDER_CONSTANT`), повторение (`cv2.BORDER_REPLICATE`), отражение/симметрия (`cv2.BORDER_REFLECT`) и циклическое заполнение (`cv2.BORDER_WRAP`).

### Программа для решения

Ссылка на код: [`task1.py`](task1.py)

```py
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Путь к изображению
image_path = 'variant_4_1.jpg'

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

kernel_size = 9

# Создаем квадратное ядро фильтра
kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (
    kernel_size * kernel_size
)

# Словарь методов обработки границ
methods = {
    'Заполнение нулями\n(Constant 0)': cv2.BORDER_CONSTANT,
    'Replicate\n(Повторение)': cv2.BORDER_REPLICATE,
    'Symmetric\n(Симметрия)': cv2.BORDER_REFLECT,
    'Circular\n(Циклическое)': cv2.BORDER_WRAP,
}

pad = kernel_size // 2
filtered_images = {}

for name, border_type in methods.items():
  try:
    filtered_images[name] = cv2.filter2D(
        image, -1, kernel, borderType=border_type
    )
  except cv2.error:
    if border_type == cv2.BORDER_WRAP:
      padded = np.pad(image, pad, mode='wrap')
    elif border_type == cv2.BORDER_REFLECT:
      padded = np.pad(image, pad, mode='reflect')
    elif border_type == cv2.BORDER_REPLICATE:
      padded = np.pad(image, pad, mode='edge')
    else:
      padded = np.pad(image, pad, mode='constant', constant_values=0)

    filtered_full = cv2.filter2D(
        padded, -1, kernel, borderType=cv2.BORDER_CONSTANT
    )
    h_orig, w_orig = image.shape
    filtered_images[name] = filtered_full[
        pad : pad + h_orig, pad : pad + w_orig
    ]

# Определяем область для среза границы
h, w = image.shape
crop_size = min(120, h, w)
crop_slice = (slice(0, crop_size), slice(0, crop_size))

fig, axes = plt.subplots(2, 5, figsize=(16, 7))

titles = [
    'Оригинал',
    'Заполнение нулями\n(Constant 0)',
    'Replicate\n(Повторение)',
    'Symmetric\n(Симметрия)',
    'Circular\n(Циклическое)',
]

# Изображения целиком
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title(titles[0] + '\n(целиком)', fontsize=10)
axes[0, 0].axis('off')

for col_idx, (name, res_img) in enumerate(filtered_images.items(), start=1):
  axes[0, col_idx].imshow(res_img, cmap='gray')
  axes[0, col_idx].set_title(titles[col_idx] + '\n(целиком)', fontsize=10)
  axes[0, col_idx].axis('off')

# Фрагменты краев
axes[1, 0].imshow(image[crop_slice], cmap='gray')
axes[1, 0].set_title(titles[0] + '\n(фрагмент края)', fontsize=10)
axes[1, 0].axis('off')

for col_idx, (name, res_img) in enumerate(filtered_images.items(), start=1):
  axes[1, col_idx].imshow(res_img[crop_slice], cmap='gray')
  axes[1, col_idx].set_title(
      titles[col_idx] + '\n(фрагмент края)', fontsize=10
  )
  axes[1, col_idx].axis('off')

plt.tight_layout()
plt.show()
```

### Результаты обработки

![Результаты Задания 1](Screenshots_lab2/task1_1.png)

На скриншоте представлены результаты применения различных методов обработки границ для изображения целиком, а также детальные фрагменты приграничных областей.

### Выводы по результатам

Различные методы обработки границ по-разному проявляют себя на периферии изображения:

- `BORDER_CONSTANT` создает искусственную темную рамку по краям из-за примеси нулевых значений.
- `BORDER_REPLICATE` растягивает крайние пиксели, что выглядит естественно для фоновых областей, но может искажать структуру контуров.
- `BORDER_REFLECT` обеспечивает зеркальное отражение пикселей, предотвращая резкие скачки яркости.
- `BORDER_WRAP` переносит противоположные стороны друг на друга, что применимо только для периодических текстур.

Рекомендации по использованию:

- Желательный размер ядра (`kernel_size`) для стандартных задач фильтрации составляет от `3 x 3` до `9 x 9`. Большие размеры (>= 15) приводят к сильному размытию краев и артефактам на границах.
- Для большинства реальных изображений рекомендуется использовать `cv2.BORDER_REFLECT` или `cv2.BORDER_REPLICATE`, так как они минимизируют появление ложных контрастных границ по периметру.

---

## Задание 2: Сравнение однородного и взвешенного усредняющих фильтров

### Задача

Реализовать программу, создающую обычный однородный усредняющий фильтр (`Box Filter`) и взвешенный усредняющий фильтр (`Gaussian Filter`). Сравнить их способность сохранять резкость краев и деталей изображения.

### Программа для решения

Ссылка на код: [`task2.py`](task2.py)

```py
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
```

### Результаты обработки

![Результаты Задания 2](Screenshots_lab2/task2_1.png)

На скриншоте приведены результаты фильтрации однородным (`Box`) и взвешенным (`Gaussian`) фильтрами на общем плане и на детализированных фрагментах.

### Выводы по результатам

Взвешенный фильтр Гаусса демонстрирует лучшие показатели сохранения структуры и контуров объектов по сравнению с однородным усреднением, поскольку распределение весов плавно убывает от центра к периферии.

Рекомендации по использованию:
- Для подавления шумов с минимальным размытием контуров рекомендуется использовать фильтр Гаусса (`cv2.GaussianBlur`).
- Оптимальные параметры: `kernel_size` в диапазоне от `3 x 3` до `11 x 11`. При больших значениях (`kernel_size` >= 15) детали полностью теряются. Параметр `sigmaX` лучше оставлять равным `0`, чтобы функция OpenCV автоматически рассчитала его на основе размера ядра.

---
