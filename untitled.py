import subprocess
import sys

import torch


# Функция для установки библиотек из requirements.txt
def install_requirements():
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при установке библиотек: {e}")


#Вызов функции установки
install_requirements()


from PyQt5 import QtWidgets, uic, QtCore, QtGui
import sys
import os
import yaml
import cv2
from ultralytics import YOLO
import time
import torch
import gc

class TrainingThread(QtCore.QThread):
    progress_update = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal()

    def __init__(self, model, data_yaml, epochs):
        super().__init__()
        self.model = model
        self.data_yaml = data_yaml
        self.epochs = epochs


        self.is_running = True  #



    def run(self):
        try:
            # Запускаем обучение модели. Метод train возвращает объект с атрибутом save_dir.
            results = self.model.train(data=self.data_yaml, epochs=self.epochs, verbose=False)
            best_weights_dir = os.path.join(results.save_dir, "weights")
            best_weights = os.path.join(best_weights_dir, "best.pt") #Улучшенная обработка ошибки

            self.progress_update.emit(f"Обучение завершено!\nBest weights: {best_weights}\nСохранено в {best_weights_dir} \nПоле val в data.yaml НЕ изменено. (Это плохая практика)")
        except Exception as e:
            self.progress_update.emit(f"Ошибка во время обучения: {e}")
        finally:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None  #Гарантируем, что ссылка удалена

            torch.cuda.empty_cache()
            gc.collect()

            if not self.is_running:
                self.interrupted_signal.emit()
            self.finished_signal.emit()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        #Загружаем интерфейс из файла main.ui
        uic.loadUi("main.ui", self)
        self.setup_ui()
        self.show()
        self.training_thread = None
        self.model = None  #модель YOLO
        self.classes = [] #Список классов (тегов)

    def setup_ui(self):
        #Режим обучения
        self.sourceImagesLineEdit = self.findChild(QtWidgets.QLineEdit, "sourceImagesLineEdit")
        self.selectSourceImagesButton = self.findChild(QtWidgets.QPushButton, "selectSourceImagesButton")
        self.selectCroppedImagesLineEdit = self.findChild(QtWidgets.QLineEdit, "selectCroppedImagesLineEdit")
        self.selectCroppedImagesButton = self.findChild(QtWidgets.QPushButton, "selectCroppedImagesButton")
        self.configFilePathLineEdit = self.findChild(QtWidgets.QLineEdit, "configFilePathLineEdit")
        self.selectConfigFileButton = self.findChild(QtWidgets.QPushButton, "selectConfigFileButton")
        self.trainingLogTextEdit = self.findChild(QtWidgets.QTextEdit, "trainingLogTextEdit")
        self.newTagLineEdit = self.findChild(QtWidgets.QLineEdit, "newTagLineEdit")
        self.tagComboBox = self.findChild(QtWidgets.QComboBox, "tagComboBox")
        self.trainButton = self.findChild(QtWidgets.QPushButton, "trainButton")
        self.resetButton = self.findChild(QtWidgets.QPushButton, "resetButton")
        self.trainMoreButton = self.findChild(QtWidgets.QPushButton, "trainMoreButton") #  Получаем ссылку на кнопку "Дообучить"

        #Режим обрезки
        self.sourceImagesLineEdit_2 = self.findChild(QtWidgets.QLineEdit, "sourceImagesLineEdit_2")
        self.selectSourceImagesButton_2 = self.findChild(QtWidgets.QPushButton, "selectSourceImagesButton_2")
        self.outputImagesLineEdit = self.findChild(QtWidgets.QLineEdit, "outputImagesLineEdit")
        self.selectOutputImagesButton = self.findChild(QtWidgets.QPushButton, "selectOutputImagesButton")
        self.borderSizeSpinBox = self.findChild(QtWidgets.QSpinBox, "borderSizeSpinBox")
        self.startButton = self.findChild(QtWidgets.QPushButton, "startButton")

        # Режим обрезки и тестирования (Работа)
        self.sourceImagesLineEdit_2 = self.findChild(QtWidgets.QLineEdit, "sourceImagesLineEdit_2")
        self.selectCroppedImagesLineEdit_2 = self.findChild(QtWidgets.QLineEdit, "selectCroppedImagesLineEdit_2")
        self.selectCroppedImagesButton_2 = self.findChild(QtWidgets.QPushButton, "selectCroppedImagesButton_2")
        self.pushButton_3 = self.findChild(QtWidgets.QPushButton, "pushButton_3")
        self.epochSpinBox = self.findChild(QtWidgets.QSpinBox, "epochSpinBox")
        self.image_label = self.findChild(QtWidgets.QLabel, "image_label")
        # Блок для тестирования
        self.testFolderLineEdit = self.findChild(QtWidgets.QLineEdit, "testFolderLineEdit")
        self.selectTestFolderButton = self.findChild(QtWidgets.QPushButton, "selectTestFolderButton")
        self.testButton = self.findChild(QtWidgets.QPushButton, "testButton")

        if self.selectSourceImagesButton_2:
            self.selectSourceImagesButton_2.clicked.connect(self.select_source_images_folder_work)
        if self.selectCroppedImagesButton_2:
            self.selectCroppedImagesButton_2.clicked.connect(self.select_cropped_images_folder_work)
        if self.pushButton_3:
            self.pushButton_3.clicked.connect(self.process_images)
        if self.testButton:
            self.testButton.clicked.connect(self.start_testing)

        #Подключаем кнопки "Обзор..."
        if self.selectSourceImagesButton:
            self.selectSourceImagesButton.clicked.connect(self.select_source_images_folder_training)
        if self.selectConfigFileButton:
            self.selectConfigFileButton.clicked.connect(self.select_config_file)
        if self.selectCroppedImagesButton:
            self.selectCroppedImagesButton.clicked.connect(self.select_cropped_images_folder_training)
        if self.trainButton:
            self.trainButton.clicked.connect(self.start_training)
        if self.resetButton:
            self.resetButton.clicked.connect(self.reset_network)
        if self.trainMoreButton:
            self.trainMoreButton.clicked.connect(self.start_training_more)

        #Режим обрезки
        if self.selectSourceImagesButton_2:
            self.selectSourceImagesButton_2.clicked.connect(self.select_source_images_folder_cropping)
        if self.selectOutputImagesButton:
            self.selectOutputImagesButton.clicked.connect(self.select_output_images_folder_cropping)
        if self.selectCroppedImagesButton_2:
            self.selectCroppedImagesButton_2.clicked.connect(self.select_cropped_images_folder_work)
    #Методы для режима обучения
    def select_source_images_folder_training(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку с исходными кадрами")
        if folder and self.sourceImagesLineEdit:
            self.sourceImagesLineEdit.setText(folder)

    def select_config_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите YAML файл", "",
                                                             "YAML Files (*.yaml *.yml)")
        if file_path and self.configFilePathLineEdit:
            self.configFilePathLineEdit.setText(file_path)
            #Загружаем классы из YAML файла
            try:
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f) 
                    classes = data.get('names', []) #Предполагаем, что имена классов находятся в ключе 'names'
                    self.classes = classes  #Сохраняем список классов
                    #Очищаем и заполняем tagComboBox
                    self.tagComboBox.clear()
                    self.tagComboBox.addItems(classes)
                    self.trainingLogTextEdit.append(f"Загружены классы из {file_path}: {classes}")

            except FileNotFoundError:
                self.trainingLogTextEdit.append(f"Ошибка: Файл {file_path} не найден")
            except yaml.YAMLError as e:
                self.trainingLogTextEdit.append(f"Ошибка при чтении YAML файла: {e}")
            except Exception as e:
                self.trainingLogTextEdit.append(f"Произошла ошибка: {e}")

    def select_cropped_images_folder_training(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку с вырезанными объектами")
        if folder and self.croppedImagesLineEdit:
            self.selectCroppedImagesLineEdit.setText(folder)

    def reset_network(self):
        self.model = None
        if self.trainingLogTextEdit:
            self.trainingLogTextEdit.append("Нейросеть сброшена. При следующем обучении будет создана новая")

    def start_training(self):
        #Необходимо указать три пути: для исходных кадров и обрезанных, для файла конфигурации
        source_path = self.sourceImagesLineEdit.text() if self.sourceImagesLineEdit else ""
        cropped_path = self.croppedImagesLineEdit.text() if self.croppedImagesLineEdit else ""
        config_path = self.configFilePathLineEdit.text() if self.configFilePathLineEdit else ""
        if not source_path or not config_path:
            QtWidgets.QMessageBox.warning(self, "Предупреждение",
                                          "Укажите три пути")
            return
        #YAML файл для обучения уже существует
        data_yaml = config_path
        if not os.path.exists(data_yaml):
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Файл {data_yaml} не найден")
            return
        if self.configFilePathLineEdit is None:
            QtWidgets.QMessageBox.critical(self, "Ошибка",
                                           "Виджет 'configFilePathLineEdit' не найден в файле main1.ui.")
            return
        data_yaml_path = self.configFilePathLineEdit.text()
        if not data_yaml_path:
            QtWidgets.QMessageBox.warning(self, "Предупреждение", "Пожалуйста, укажите путь к YAML файлу с данными")
            return
        epochs = 5  # Количество эпох
        try:
            self.trainingLogTextEdit.clear()
            self.model = YOLO("yolov8m.pt")
            self.training_thread = TrainingThread(self.model, data_yaml_path, epochs)
            self.training_thread.progress_update.connect(self.update_training_log)
            self.training_thread.finished_signal.connect(self.training_finished)
            self.training_thread.start()
            self.trainingLogTextEdit.append(f"Начинаем обучение модели yolov8m.pt на {epochs} эпох...")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка при запуске обучения: {e}")

    def start_training_more(self):
        #Получаем пути к папкам и файлу конфигурации
        source_images_path = self.sourceImagesLineEdit.text()
        cropped_images_path = self.croppedImagesLineEdit.text()
        config_file_path = self.configFilePathLineEdit.text()

        #Проверяем, что все пути указаны
        if not source_images_path or not config_file_path:
            self.trainingLogTextEdit.append("Ошибка: Укажите все пути!")
            return

        #Проверяем, что модель уже загружена
        if not self.model:
            self.trainingLogTextEdit.append("Ошибка: Сначала обучите модель или загрузите конфигурационный файл!")
            return

        #Отключаем кнопки, чтобы избежать повторных запусков
        self.trainButton.setEnabled(False)
        self.resetButton.setEnabled(False)
        self.trainMoreButton.setEnabled(False)

        #Запускаем обучение в отдельном потоке
        data_yaml = config_file_path
        epochs = 5
        self.training_thread = TrainingThread(self.model, data_yaml, epochs, self.classes)  #режим "Дообучение"
        self.training_thread.progress_update.connect(self.update_training_log)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.start()

    def update_training_log(self, message):
        if self.trainingLogTextEdit:
            self.trainingLogTextEdit.append(message)
        QtWidgets.QApplication.processEvents()

    def training_finished(self):
        self.training_thread = None
        if self.trainButton:
            self.trainButton.setEnabled(True)
        if self.resetButton:
            self.resetButton.setEnabled(True)
        if self.trainMoreButton:
            self.trainMoreButton.setEnabled(True)

    #Методы для режима обрезки
    def select_source_images_folder_cropping(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку с исходными кадрами")
        if folder and self.sourceImagesLineEdit_2:
            self.sourceImagesLineEdit_2.setText(folder)

    def select_output_images_folder_cropping(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения обрезанных изображений")
        if folder and self.outputImagesLineEdit:
            self.outputImagesLineEdit.setText(folder)

    def create_network_config(self):
        #Формируем список тегов: берем содержимое tagComboBox и новый тег, если он был введен
        tags = []
        if self.tagComboBox:
            for i in range(self.tagComboBox.count()):
                tags.append(self.tagComboBox.itemText(i))
        new_tag = self.newTagLineEdit.text().strip() if self.newTagLineEdit else ""
        if new_tag and new_tag not in tags:
            tags.append(new_tag)
        #Подсчитываем количество объектов по файлам в папке, указанной в configFilePathLineEdit
        cropped_folder = self.configFilePathLineEdit.text() if self.configFilePathLineEdit else ""
        object_count = 0
        if os.path.isdir(cropped_folder):
            for file in os.listdir(cropped_folder):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    object_count += 1
        config = {
            "tags": tags,
            "object_count": object_count
        }
        try:
            with open("network_config.yaml", "w") as f:
                yaml.dump(config, f, explicit_start=True)
            if self.trainingLogTextEdit:
                self.trainingLogTextEdit.append("Конфигурационный файл сети (network_config.yaml) создан")
        except Exception as e:
            if self.trainingLogTextEdit:
                self.trainingLogTextEdit.append(f"Ошибка при создании конфигурационного файла сети: {e}")

    def resize_image(image, max_size=640):
        """Изменяет размер изображения, сохраняя пропорции."""
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            if height > width:
                new_height = max_size
                new_width = int(width * (max_size / height))
            else:
                new_width = max_size
                new_height = int(height * (max_size / width))
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized_image
        return image
    #Методы для режима обрезки (Работа)
    def select_source_images_folder_work(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку с исходными кадрами")
        if folder and self.sourceImagesLineEdit_2:
            self.sourceImagesLineEdit_2.setText(folder)

    def select_cropped_images_folder_work(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения обрезанных объектов")
        if folder and self.croppedImagesLineEdit_2:
            self.croppedImagesLineEdit_2.setText(folder)

    def process_images(self):
        source_folder = self.sourceImagesLineEdit_2.text() if self.sourceImagesLineEdit_2 else ""
        dest_folder = self.croppedImagesLineEdit_2.text() if self.croppedImagesLineEdit_2 else ""
        if not source_folder or not dest_folder:
            QtWidgets.QMessageBox.warning(self, "Предупреждение",
                                          "Укажите пути к исходным кадрам и к папке для обрезанных объектов")
            return
        if not os.path.isdir(source_folder):
            QtWidgets.QMessageBox.warning(self, "Предупреждение", "Указанная папка с исходными кадрами не найдена")
            return
        border = self.epochSpinBox.value() if self.epochSpinBox else 0
        image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            QtWidgets.QMessageBox.warning(self, "Предупреждение", "В исходной папке нет изображений")
            return
        for filename in image_files:
            image_path = os.path.join(source_folder, filename)
            img = cv2.imread(image_path)
            if img is None:
                continue
            try:
                #Если модель не обучена, используем базовую модель для детекции
                if self.model is None:
                    self.model = YOLO("yolov8m.pt")
                results = self.model.predict(img)
                crop_counter = 1
                for result in results:
                    for box in result.boxes:
                        xyxy = box.xyxy[0].tolist()
                        x1 = max(int(xyxy[0]) - border, 0)
                        y1 = max(int(xyxy[1]) - border, 0)
                        x2 = min(int(xyxy[2]) + border, img.shape[1])
                        y2 = min(int(xyxy[3]) + border, img.shape[0])
                        cropped = img[y1:y2, x1:x2]
                        base, ext = os.path.splitext(filename)
                        save_name = f"{base}_{crop_counter}{ext}"
                        save_path = os.path.join(dest_folder, save_name)
                        cv2.imwrite(save_path, cropped)
                        crop_counter += 1
                #Отображаем одно обрезанное изображение для демонстрации
                if crop_counter > 1:
                    first_crop = os.path.join(dest_folder, f"{os.path.splitext(filename)[0]}_1{ext}")
                    disp_img = cv2.imread(first_crop)
                    self.display_image(disp_img)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Ошибка", f"Ошибка при обработке {filename}: {e}")
        QtWidgets.QMessageBox.information(self, "Обрезка", "Обработка изображений завершена")

    def display_image(self, image):
        if image is None:
            return
        if len(image.shape) == 2:
            qformat = QtGui.QImage.Format_Indexed8
        else:
            qformat = QtGui.QImage.Format_RGB888 if image.shape[2] == 3 else QtGui.QImage.Format_RGBA8888
        outImage = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        pixmap = QtGui.QPixmap.fromImage(outImage)
        if self.image_label:
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)

    #Методы для тестирования модели (Работа)
    def select_test_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку с тестовыми данными")
        if folder and self.testFolderLineEdit:
            self.testFolderLineEdit.setText(folder)

    def start_testing(self):
        test_folder = self.testFolderLineEdit.text() if self.testFolderLineEdit else ""
        if not test_folder:
            QtWidgets.QMessageBox.warning(self, "Предупреждение", "Укажите путь к тестовым данным")
            return
        if not os.path.isdir(test_folder):
            QtWidgets.QMessageBox.warning(self, "Предупреждение", "Папка с тестовыми данными не найдена")
            return
        if self.model is None:
            QtWidgets.QMessageBox.warning(self, "Предупреждение", "Модель не обучена. Сначала обучите модель")
            return
        image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            QtWidgets.QMessageBox.warning(self, "Предупреждение", "В тестовой папке нет изображений")
            return
        test_image_path = os.path.join(test_folder, image_files[0])
        img = cv2.imread(test_image_path)
        if img is None:
            QtWidgets.QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить изображение: {test_image_path}")
            return
        try:
            results = self.model.predict(img)
            annotated = self.annotate_image(img, results)
            self.display_image(annotated)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка при тестировании модели: {e}")

    def annotate_image(self, image, results):
        try:
            for result in results:
                for box in result.boxes:
                    xyxy = box.xyxy[0].tolist()
                    class_id = int(box.cls[0].item())
                    confidence = box.conf[0].item()
                    label = f"{result.names[class_id]} {confidence:.2f}"
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            return image
        except Exception as e:
            raise RuntimeError(f"Ошибка при аннотации изображения: {e}")



    def create_network_config(self):
        #Формируем список тегов: берем содержимое tagComboBox и новый тег, если он был введен
        tags = []
        if self.tagComboBox:
            for i in range(self.tagComboBox.count()):
                tags.append(self.tagComboBox.itemText(i))
        new_tag = self.newTagLineEdit.text().strip() if self.newTagLineEdit else ""
        if new_tag and new_tag not in tags:
            tags.append(new_tag)
        #Подсчитываем количество объектов по файлам в папке, указанной в configFilePathLineEdit
        cropped_folder = self.configFilePathLineEdit.text() if self.configFilePathLineEdit else ""
        object_count = 0
        if os.path.isdir(cropped_folder):
            for file in os.listdir(cropped_folder):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    object_count += 1
        config = {
            "tags": tags,
            "object_count": object_count
        }
        try:
            with open("network_config.yaml", "w") as f:
                yaml.dump(config, f, explicit_start=True)
            if self.trainingLogTextEdit:
                self.trainingLogTextEdit.append("Конфигурационный файл сети (network_config.yaml) создан")
        except Exception as e:
            if self.trainingLogTextEdit:
                self.trainingLogTextEdit.append(f"Ошибка при создании конфигурационного файла сети: {e}")

    #Методы для режима обрезки (Работа)
    def select_source_images_folder_work(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку с исходными кадрами")
        if folder and self.sourceImagesLineEdit_2:
            self.sourceImagesLineEdit_2.setText(folder)

    def select_cropped_images_folder_work(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения обрезанных объектов")
        if folder and self.croppedImagesLineEdit_2:
            self.croppedImagesLineEdit_2.setText(folder)

    def process_images(self):
        source_folder = self.sourceImagesLineEdit_2.text() if self.sourceImagesLineEdit_2 else ""
        dest_folder = self.croppedImagesLineEdit_2.text() if self.croppedImagesLineEdit_2 else ""
        if not source_folder or not dest_folder:
            QtWidgets.QMessageBox.warning(self, "Предупреждение",
                                          "Укажите пути к исходным кадрам и к папке для обрезанных объектов")
            return
        if not os.path.isdir(source_folder):
            QtWidgets.QMessageBox.warning(self, "Предупреждение", "Указанная папка с исходными кадрами не найдена")
            return
        border = self.epochSpinBox.value() if self.epochSpinBox else 0
        image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            QtWidgets.QMessageBox.warning(self, "Предупреждение", "В исходной папке нет изображений")
            return
        for filename in image_files:
            image_path = os.path.join(source_folder, filename)
            img = cv2.imread(image_path)
            if img is None:
                continue
            try:
                #Если модель не обучена, используем базовую модель для детекции
                if self.model is None:
                    self.model = YOLO("yolov8m.pt")
                results = self.model.predict(img)
                crop_counter = 1
                for result in results:
                    for box in result.boxes:
                        xyxy = box.xyxy[0].tolist()
                        x1 = max(int(xyxy[0]) - border, 0)
                        y1 = max(int(xyxy[1]) - border, 0)
                        x2 = min(int(xyxy[2]) + border, img.shape[1])
                        y2 = min(int(xyxy[3]) + border, img.shape[0])
                        cropped = img[y1:y2, x1:x2]
                        base, ext = os.path.splitext(filename)
                        save_name = f"{base}_{crop_counter}{ext}"
                        save_path = os.path.join(dest_folder, save_name)
                        cv2.imwrite(save_path, cropped)
                        crop_counter += 1
                #Отображаем одно обрезанное изображение для демонстрации
                if crop_counter > 1:
                    first_crop = os.path.join(dest_folder, f"{os.path.splitext(filename)[0]}_1{ext}")
                    disp_img = cv2.imread(first_crop)
                    self.display_image(disp_img)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Ошибка", f"Ошибка при обработке {filename}: {e}")
        QtWidgets.QMessageBox.information(self, "Обрезка", "Обработка изображений завершена")

    def display_image(self, image):
        if image is None:
            return
        if len(image.shape) == 2:
            qformat = QtGui.QImage.Format_Indexed8
        else:
            qformat = QtGui.QImage.Format_RGB888 if image.shape[2] == 3 else QtGui.QImage.Format_RGBA8888
        outImage = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        pixmap = QtGui.QPixmap.fromImage(outImage)
        if self.image_label:
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)

    #Методы для тестирования модели (Работа)
    def select_test_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Выберите папку с тестовыми данными")
        if folder and self.testFolderLineEdit:
            self.testFolderLineEdit.setText(folder)

    def start_testing(self):
        test_folder = self.testFolderLineEdit.text() if self.testFolderLineEdit else ""
        if not test_folder:
            QtWidgets.QMessageBox.warning(self, "Предупреждение", "Укажите путь к тестовым данным")
            return
        if not os.path.isdir(test_folder):
            QtWidgets.QMessageBox.warning(self, "Предупреждение", "Папка с тестовыми данными не найдена")
            return
        if self.model is None:
            QtWidgets.QMessageBox.warning(self, "Предупреждение", "Модель не обучена. Сначала обучите модель")
            return
        image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            QtWidgets.QMessageBox.warning(self, "Предупреждение", "В тестовой папке нет изображений")
            return
        test_image_path = os.path.join(test_folder, image_files[0])
        img = cv2.imread(test_image_path)
        if img is None:
            QtWidgets.QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить изображение: {test_image_path}")
            return
        try:
            results = self.model.predict(img)
            annotated = self.annotate_image(img, results)
            self.display_image(annotated)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f"Ошибка при тестировании модели: {e}")

    def annotate_image(self, image, results):
        try:
            for result in results:
                for box in result.boxes:
                    xyxy = box.xyxy[0].tolist()
                    class_id = int(box.cls[0].item())
                    confidence = box.conf[0].item()
                    label = f"{result.names[class_id]} {confidence:.2f}"
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            return image
        except Exception as e:
            raise RuntimeError(f"Ошибка при аннотации изображения: {e}")



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
