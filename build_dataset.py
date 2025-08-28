import cv2
import os
import numpy as np

VIDEO_SOURCES = {
    "fap": "fap.mp4",
    "nofap": "nofap.mp4"
}
OUTPUT_ROOT = "./dataset"
INTERVAL_SEC = 1
FRAME_SIZE = (224, 224)  # для тренировки мобилнет лучше чем 32x32

def adjust_brightness(image, factor):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def add_noise(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def gamma_correction(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def augment_image(image):
    aug_images = []
    aug_images.append(cv2.flip(image, 1))  # зеркалка
    for factor in [0.6, 1.4]:
        aug_images.append(adjust_brightness(image, factor))
    for gamma in [0.5, 2.0]:
        aug_images.append(gamma_correction(image, gamma))
    aug_images.append(add_noise(image))
    combo = add_noise(adjust_brightness(image, 1.2))
    aug_images.append(combo)
    return aug_images

def extract_and_augment(video_path, output_dir, interval_sec=1, size=(224,224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: не могу открыть видео {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{video_path}: FPS={fps}, Frames={frame_count}, Interval={frame_interval}")

    count = 0
    saved_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            base_name = f"frame_{saved_frames:05d}"
            out_path = os.path.join(output_dir, base_name + ".png")
            cv2.imwrite(out_path, resized)

            # augment
            aug_images = augment_image(resized)
            for i, aug in enumerate(aug_images):
                aug_path = os.path.join(output_dir, f"{base_name}_aug_{i}.png")
                cv2.imwrite(aug_path, aug)

            saved_frames += 1
        count += 1

    cap.release()
    print(f"Готово! {video_path} → {saved_frames} кадров (+ аугментации)")

if __name__ == "__main__":
    for label, video in VIDEO_SOURCES.items():
        out_dir = os.path.join(OUTPUT_ROOT, label)
        extract_and_augment(video, out_dir, interval_sec=INTERVAL_SEC, size=FRAME_SIZE)
