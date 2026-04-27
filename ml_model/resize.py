import cv2
import os
import numpy as np
import random
import pickle
from pathlib import Path

# Константы путей
INPUT_DIR = Path("datasets/wheat_leaf/stripe_rust")
OUTPUT_DIR = Path("datasets/processed_data")

def resize_image(img):
    image = cv2.resize(img, (256, 256))  # Первые размеры берём 256х256
    return image

def blur_image(img):
    blur_matrix = random.randrange(3, 8, 2)
    blur = cv2.GaussianBlur(img, (blur_matrix, blur_matrix), 0)
    return blur

def shading_image(img):
    shade_img = np.uint8(np.clip((random.randint(65, 130) * 0.01 * img + 10), 0, 255))
    return shade_img


def augment_and_save(img, filename_base):
    # Поворот на 90/180/270 градусов и применение эффектов
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    for angle in range(0, 360, 90):
        # Вращение
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, matrix, (w, h))

        # Генерация комбинаций эффектов
        variants = {
            f"angle_{angle}": rotated,
            f"angle_{angle}_blur": blur_image(rotated),
            f"angle_{angle}_shade": shading_image(rotated),
            f"angle_{angle}_mix": blur_image(shading_image(rotated))
        }

        # Сохранение всех вариантов
        for suffix, image_variant in variants.items():
            save_path = OUTPUT_DIR / f"{filename_base}_{suffix}.jpg"
            cv2.imwrite(str(save_path), image_variant)

if __name__ == "__main__":
    # Создаем папку для результата, если её нет
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = list(INPUT_DIR.glob("*.JPG")) + list(INPUT_DIR.glob("*.jpg"))
    
    print(f"Найдено изображений: {len(image_files)}. Начинаю обработку...")

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            base_name = img_path.stem 
            processed_img = resize_image(img)
            augment_and_save(processed_img, base_name)
    
    print("Обработка завершена успешно.")



