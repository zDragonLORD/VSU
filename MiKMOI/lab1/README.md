# Математические и компьютерные методы обработки изображений

---

## Лабораторная работа № 1

#### Выполнил: Валиков Егор

#### Группа: 19

---

## Задание 1: Линейное контрастирование изображений

### Задача: Определить динамический диапазон входного изображения. Осуществить линейное контрастирование входного изображения в заданный динамический диапазон яркостей. Вывести изображения и их гистограммы.

## Программа для решения

Ссылка на код: [`task1.py`](zDragonLORD/VSU/blob/main/MiKMOI/lab1/task1.py)

```py
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. Загрузка исходного изображения
image_path = '4.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
  raise FileNotFoundError(
      f"Ошибка: не удалось загрузить изображение."
  )

# 2. Определение динамического диапазона входного изображения
i_min = np.min(img)
i_max = np.max(img)
print(f'Минимальная яркость (min): {i_min}')
print(f'Максимальная яркость (max): {i_max}')
print(f'Динамический диапазон входного изображения: [{i_min}, {i_max}]')

# 3. Линейное контрастирование в заданный диапазон (0 - 255)
out_min, out_max = 0, 255

def linear_contrast(image, in_min, in_max, out_min, out_max):
  img_float = image.astype(np.float32)

  # Формула линейного контрастирования:
  # g(x,y) = ((f(x,y) - in_min) / (in_max - in_min)) * (out_max - out_min) + out_min
  if in_max == in_min:
    return np.zeros_like(image, dtype=np.uint8)
  out = ((img_float - in_min) / (in_max - in_min) * (out_max - out_min) + out_min)
  out = np.clip(out, out_min, out_max).astype(np.uint8)
  return out

img_contrasted = linear_contrast(img, i_min, i_max, out_min, out_max)

# 4. Вывод изображений и их гистограмм
plt.figure(figsize=(14, 10))

# Исходное изображение
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Исходное изображение')
plt.axis('off')

# Гистограмма исходного изображения
plt.subplot(2, 2, 2)
plt.hist(img.ravel(), bins=256, range=[0, 256], color='gray', edgecolor='black')
plt.title('Гистограмма исходного изображения')
plt.xlabel('Яркость пикселя')
plt.ylabel('Количество')

# Контрастированное изображение
plt.subplot(2, 2, 3)
plt.imshow(img_contrasted, cmap='gray', vmin=0, vmax=255)
plt.title('Контрастированное изображение')
plt.axis('off')

# Гистограмма контрастированного изображения
plt.subplot(2, 2, 4)
plt.hist(
    img_contrasted.ravel(),
    bins=256,
    range=[0, 256],
    color='gray',
    edgecolor='black',
)
plt.title('Гистограмма контрастированного изображения')
plt.xlabel('Яркость пикселя')
plt.ylabel('Количество')

plt.tight_layout()
plt.show()
```



