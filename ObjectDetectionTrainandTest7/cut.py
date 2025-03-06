import subprocess
import sys
import subprocess
import torch


# Функция для установки библиотек из requirements.txt
def install_requirements():
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при установке библиотек: {e}")


#Вызов функции установки
install_requirements()

from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')



import cv2
import os
import shutil
import random
from datetime import datetime


def create_yolo_annotations_video(original_video_path, cropped_images_path, class_id, frame_index, object_name):
    """
    Создает аннотации в формате YOLO для вырезанных объектов из конкретного кадра видео.
    Адаптировано для работы с именем объекта.

    Args:
        original_video_path: Путь к исходному видеофайлу.
        cropped_images_path: Путь к папке с вырезанными изображениями.
        class_id: Идентификатор класса для вырезанных объектов.
        frame_index: Номер кадра видео, для которого нужно создать аннотации.
        object_name: Имя объекта (используется для сопоставления файлов).
    """
    annotations_path = fr"{os.path.dirname(original_video_path)}/labels"
    os.makedirs(annotations_path, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(original_video_path))[0]
    annotation_file = os.path.join(annotations_path, f"{video_name}_frame_{frame_index}.txt")

    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {original_video_path}")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, original_frame = cap.read()
    cap.release()

    if not ret or original_frame is None:
        print(f"Не удалось прочитать кадр {frame_index} из видео {original_video_path}")
        return

    height, width, _ = original_frame.shape

    with open(annotation_file, 'w') as f:
        for cropped_image_name in os.listdir(cropped_images_path):
            # Проверяем, соответствует ли имя объекта и номер кадра
            if cropped_image_name.startswith(f"{object_name}_frame_{frame_index}_"):
                cropped_image_path = os.path.join(cropped_images_path, cropped_image_name)
                cropped_image = cv2.imread(cropped_image_path)
                if cropped_image is None:
                    print(f"Не удалось загрузить изображение: {cropped_image_path}")
                    continue
                h_crop, w_crop, _ = cropped_image.shape

                result = cv2.matchTemplate(original_frame, cropped_image, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                x1, y1 = max_loc
                x2, y2 = x1 + w_crop, y1 + h_crop

                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height

                f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")


# Функция для вырезания и аннотирования видео


def crop_and_annotate_video(original_video_path, cropped_images_path, class_id, output_cropped_path, start_frame=0, end_frame=-1, frame_step=1):
    """
    Вырезает объекты из видео на основе аннотаций.  Использует create_yolo_annotations_video.

    Args:
        original_video_path: Путь к видео.
        cropped_images_path: Путь к папке с вырезанными изображениями.
        class_id:  Идентификатор класса.
        output_cropped_path: Путь для сохранения вырезанных кадров.
        start_frame: Начальный кадр.
        end_frame: Конечный кадр. -1 - до конца.
        frame_step: Шаг.
    """
    annotations_path = fr"{os.path.dirname(original_video_path)}/labels"
    os.makedirs(output_cropped_path, exist_ok=True)

    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {original_video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame == -1 or end_frame > frame_count:
        end_frame = frame_count

    for frame_index in range(start_frame, end_frame, frame_step):
        # Получаем имя видеофайла без расширения
        video_name = os.path.splitext(os.path.basename(original_video_path))[0]
        # Получаем все имена обьектов, которые аннотированы
        object_names = set()
        for cropped_image_name in os.listdir(cropped_images_path):
            if f"_frame_{frame_index}_" in cropped_image_name:
                object_name = cropped_image_name.split('_frame_')[0]
                object_names.add(object_name)
        # Для каждого найденного обьекта - создаем аннотации и вырезаем
        for object_name in object_names:
            create_yolo_annotations_video(original_video_path, cropped_images_path, class_id, frame_index, object_name)

            annotation_file = os.path.join(annotations_path, f"{video_name}_frame_{frame_index}.txt")

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, original_frame = cap.read()
            if not ret or original_frame is None:
                print(f"Не удалось прочитать кадр {frame_index} из видео {original_video_path}")
                continue

            if not os.path.exists(annotation_file):
                print(f"Предупреждение: Файл аннотаций не найден для кадра {frame_index} видео {original_video_path}. Пропускается.")
                continue

            with open(annotation_file, 'r') as f:
                for i, line in enumerate(f):
                    try:
                        class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())

                        height, width, _ = original_frame.shape

                        x1 = int((x_center - box_width / 2) * width)
                        y1 = int((y_center - box_height / 2) * height)
                        x2 = int((x_center + box_width / 2) * width)
                        y2 = int((y_center + box_height / 2) * height)

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width, x2)
                        y2 = min(height, y2)

                        cropped_image = original_frame[y1:y2, x1:x2]

                        cropped_filename = f"{object_name}_frame_{frame_index}_cropped_{i}.png"  # Учитываем имя объекта
                        cropped_filepath = os.path.join(output_cropped_path, cropped_filename)
                        cv2.imwrite(cropped_filepath, cropped_image)

                    except ValueError:
                        print(f"Предупреждение: Не удалось разобрать строку аннотации: {line.strip()} в {annotation_file}")
                        continue
    cap.release()

#Функция интерактивной аннотации видео

drawing = False
x1, y1 = -1, -1
original_frame = None  # Объявляем глобальную переменную
cropped_images_path = "train1/images"

def mouse_callback(event, x, y, flags, param):
    global x1, y1, drawing, original_frame

    if original_frame is None:
        return  # Нечего делать, если нет кадра

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            temp_frame = original_frame.copy()
            cv2.rectangle(temp_frame, (x1, y1), (x, y), (0, 255, 0), 2)
            cv2.imshow("Frame", temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(original_frame, (x1, y1), (x, y), (0, 255, 0), 2)
        x2, y2 = x, y

        cropped_image = original_frame[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
        if cropped_image.size > 0:
            object_name = input("Введите имя объекта: ")
            if object_name: # Чтобы пользователь мог пропустить ввод имени
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                cropped_filename = f"{object_name}_frame_{frame_index}_{timestamp}.png"
                cropped_filepath = os.path.join(cropped_images_path, cropped_filename)
                cv2.imwrite(cropped_filepath, cropped_image)
                print(f"Сохранен вырезанный объект: {cropped_filename}")
            else:
                print("Аннотация пропущена (имя объекта не введено)")
        else:
            print("Предупреждение: Вырезана пустая область")

def annotate_video(video_path):
    global original_frame, frame_index

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}")
        return

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mouse_callback)

    global frame_index
    frame_index = 0

    while True:
        ret, original_frame = cap.read()
        if not ret:
            break

        cv2.imshow("Frame", original_frame)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite(f"frame_{frame_index}.png", original_frame)
        elif key == ord("n"):  #Переход к следующему кадру
            frame_index += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index) # Переходим к следующему кадру
        elif key == ord("p"): #переход к предыдущему кадру
            frame_index = max(0, frame_index - 1) # Нельзя выйти за рамки
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        elif key == ord("r"):
            # Реализация перезагрузки текущего кадра
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, original_frame = cap.read()
            if not ret or original_frame is None:
                print("Не удалось перезагрузить кадр")

    cap.release()
    cv2.destroyAllWindows()

def crop_and_annotate_video(original_video_path, cropped_images_path, class_id, output_cropped_path, start_frame=0, end_frame=-1, frame_step=1):
    """
    Вырезает объекты из видео на основе аннотаций (БЕЗ КЛАССИФИКАЦИИ).

    Args:
        original_video_path: Путь к видео.
        cropped_images_path: Путь к папке с вырезанными изображениями.
        class_id:  Идентификатор класса.
        output_cropped_path: Путь для сохранения вырезанных кадров.
        start_frame: Начальный кадр.
        end_frame: Конечный кадр. -1 - до конца.
        frame_step: Шаг.
    """
    annotations_path = fr"{os.path.dirname(original_video_path)}/labels"
    os.makedirs(output_cropped_path, exist_ok=True)

    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {original_video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame == -1 or end_frame > frame_count:
        end_frame = frame_count

    for frame_index in range(start_frame, end_frame, frame_step):
        # Получаем имя видеофайла без расширения
        video_name = os.path.splitext(os.path.basename(original_video_path))[0]
        # Получаем все имена обьектов, которые аннотированы в cropped_images_path
        object_names = set()
        for cropped_image_name in os.listdir(cropped_images_path):
            if f"_frame_{frame_index}_" in cropped_image_name:
                object_name = cropped_image_name.split('_frame_')[0]
                object_names.add(object_name)
        # Для каждого найденного обьекта - создаем аннотации и вырезаем.
        for object_name in object_names:
            create_yolo_annotations_video(original_video_path, cropped_images_path, class_id, frame_index, object_name)

            annotation_file = os.path.join(annotations_path, f"{video_name}_frame_{frame_index}.txt")

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, original_frame = cap.read()
            if not ret or original_frame is None:
                print(f"Не удалось прочитать кадр {frame_index} из видео {original_video_path}")
                continue

            if not os.path.exists(annotation_file):
                print(f"Предупреждение: Файл аннотаций не найден для кадра {frame_index} видео {original_video_path}. Пропускается.")
                continue

            with open(annotation_file, 'r') as f:
                for i, line in enumerate(f):
                    try:
                        class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())

                        height, width, _ = original_frame.shape

                        x1 = int((x_center - box_width / 2) * width)
                        y1 = int((y_center - box_height / 2) * height)
                        x2 = int((x_center + box_width / 2) * width)
                        y2 = int((y_center + box_height / 2) * height)

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width, x2)
                        y2 = min(height, y2)

                        cropped_image = original_frame[y1:y2, x1:x2]

                        cropped_filename = f"{object_name}_frame_{frame_index}_cropped_{i}.png"  # Учитываем имя объекта
                        cropped_filepath = os.path.join(output_cropped_path, cropped_filename)
                        cv2.imwrite(cropped_filepath, cropped_image)

                    except ValueError:
                        print(f"Предупреждение: Не удалось разобрать строку аннотации: {line.strip()} в {annotation_file}")
                        continue
    cap.release()


# Пример использования
if __name__ == "__main__":
    original_video_path = "test/video/3.3.mp4"
    cropped_images_path = "train1/images"  # Папка для вырезанных изображений (шаблонов)
    output_cropped_path = "result_video"  # Папка для вырезанных объектов из видео
    class_id = 0

    os.makedirs(cropped_images_path, exist_ok=True)
    os.makedirs(output_cropped_path, exist_ok=True)


    # Автоматическое вырезание и аннотирование
    print("Начинаем автоматическое вырезание и создание аннотаций...")
    crop_and_annotate_video(original_video_path, cropped_images_path, class_id, output_cropped_path)

    print("Готово.")