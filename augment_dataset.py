import os
import cv2
import numpy as np

# Корень датасета
DATASET_DIR = './dataset'

def adjust_brightness(image, factor):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def add_noise(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def gamma_correction(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def augment_image(image):
    aug_images = []

    # Оригинал оставляем в покое — не трогаем
    aug_images.append(cv2.flip(image, 1))  # горизонтальное зеркало

    for factor in [0.6, 1.4]:  # тьма и свет
        aug_images.append(adjust_brightness(image, factor))

    for gamma in [0.5, 2.0]:  # выжженные глаза и ночное зрение
        aug_images.append(gamma_correction(image, gamma))

    aug_images.append(add_noise(image))  # цифровой шум

    combo = add_noise(adjust_brightness(image, 1.2))  # хаос-комбо
    aug_images.append(combo)

    return aug_images

def run():
    total_augmented = 0
    for label_dir in os.listdir(DATASET_DIR):
        class_path = os.path.join(DATASET_DIR, label_dir)
        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            if not filename.endswith('.png'):
                continue

            filepath = os.path.join(class_path, filename)
            image = cv2.imread(filepath)
            if image is None:
                print(f'Не могу прочитать: {filepath}')
                continue

            aug_images = augment_image(image)
            name_base = filename[:-4]

            for i, aug_img in enumerate(aug_images):
                out_path = os.path.join(class_path, f"{name_base}_aug_{i}.png")
                cv2.imwrite(out_path, aug_img)
                total_augmented += 1

    print(f'Повелитель, в каждую папку было влито {total_augmented} демонических дубликатов.')

if __name__ == '__main__':
    run()
