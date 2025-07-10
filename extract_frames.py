import cv2
import os
import sys

def extract_frames(video_path, output_dir, interval_sec=2, size=(32, 32)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: не могу открыть видео {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS: {fps}, Всего кадров: {frame_count}, Интервал кадров: {frame_interval}")

    count = 0
    saved_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            resized_frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            filename = os.path.join(output_dir, f"frame_{saved_frames:05d}.png")
            cv2.imwrite(filename, resized_frame)
            print(f"Сохранил {filename}")
            saved_frames += 1

        count += 1

    cap.release()
    print(f"Готово! Сохранено {saved_frames} кадров.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Использование: python extract_frames.py <путь_к_видео> <папка_для_кадров>")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2]

    extract_frames(video_path, output_dir)
