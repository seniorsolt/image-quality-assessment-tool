import functools
import json
import os
import sys

import numpy as np
import pandas as pd
import piq
import torch
import torchvision
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget,
                             QFileDialog, QProgressBar, QComboBox, QTextEdit)
from deepface import DeepFace
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor
from transformers import AutoProcessor, AutoModel

CONFIG_FILE = "app_config.json"


def exception_catcher(func):
    @functools.wraps(func)
    def wrapper(*arg, **kwargs):
        try:
            return func(*arg, **kwargs)
        except Exception as e:
            gui.log(str(e))

    return wrapper


class ImageComparisonApp(QMainWindow):
    def __init__(self):
        super().__init__(None)
        self.setWindowTitle("Image Comparison App")
        self.setGeometry(100, 100, 500, 400)

        layout = QVBoxLayout()

        self.org_images_label = QLabel("Select folder with original images:")
        self.org_images_button = QPushButton("Browse")
        self.org_images_button.clicked.connect(self.select_org_images_folder)

        self.generated_images_label = QLabel("Select folder with generated images:")
        self.generated_images_button = QPushButton("Browse")
        self.org_images_path_label = QLabel("No folder selected")
        self.generated_images_path_label = QLabel("No folder selected")
        self.generated_images_button.clicked.connect(self.select_generated_images_folder)

        self.model_selection_label = QLabel("Select face recognition model:")
        self.model_selection_combo = QComboBox(None)
        self.model_selection_combo.addItems(["VGG-Face", "Facenet", "Facenet512", "OpenFace",
                                             "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"])

        self.metric_selection_label = QLabel("Select similarity metric:")
        self.metric_selection_combo = QComboBox(None)
        self.metric_selection_combo.addItems(["cosine", "euclidean", "euclidean_l2"])

        self.backend_selection_label = QLabel("Select backend model:")
        self.backend_selection_combo = QComboBox(None)
        self.backend_selection_combo.addItems(['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe',
                                              'yolov8', 'yunet', 'fastmtcnn',])

        self.start_button = QPushButton("Start Comparison")
        self.start_button.clicked.connect(self.start_comparison)

        self.export_json_button = QPushButton("Export as JSON")
        self.export_json_button.clicked.connect(self.export_as_json)

        self.export_excel_button = QPushButton("Export as Excel")
        self.export_excel_button.clicked.connect(self.export_as_excel)

        self.progress_label = QLabel("Progress:")
        self.progress_bar = QProgressBar(None)
        self.progress_bar.setValue(0)

        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)

        for widget in [self.org_images_label, self.org_images_path_label, self.org_images_button,
                       self.generated_images_label, self.generated_images_path_label,
                       self.generated_images_button, self.model_selection_label, self.model_selection_combo,
                       self.metric_selection_label, self.metric_selection_combo, self.backend_selection_label,
                       self.backend_selection_combo,
                       self.start_button, self.export_json_button, self.export_excel_button,
                       self.progress_label, self.progress_bar, self.log_text_edit]:
            layout.addWidget(widget)

        central_widget = QWidget(None)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.org_images_folder_path = ""
        self.generated_images_folder_path = ""

        if app.config:
            self.org_images_folder_path = app.config['org_images_folder_path']
            self.org_images_path_label.setText(self.org_images_folder_path)
            self.generated_images_folder_path = app.config['generated_images_folder_path']
            self.generated_images_path_label.setText(self.generated_images_folder_path)
            index_model = self.model_selection_combo.findText(app.config['selected_model'])
            if index_model != -1:
                self.model_selection_combo.setCurrentIndex(index_model)
            index_metric = self.metric_selection_combo.findText(app.config['selected_metric'])
            if index_metric != -1:
                self.metric_selection_combo.setCurrentIndex(index_metric)
            index_backend = self.backend_selection_combo.findText(app.config['selected_backend'])
            if index_backend != -1:
                self.backend_selection_combo.setCurrentIndex(index_backend)
        self.results = None
        self.comparison_thread = None

    @exception_catcher
    def select_org_images_folder(self, *args, **kwargs) -> None:
        self.org_images_folder_path = QFileDialog.getExistingDirectory(self, "Select folder "
                                                                             "with original images")
        self.org_images_path_label.setText(self.org_images_folder_path)
        self.log("Selected folder with original images:" + self.org_images_folder_path)

    @exception_catcher
    def select_generated_images_folder(self, *args, **kwargs) -> None:
        self.generated_images_folder_path = QFileDialog.getExistingDirectory(self, "Select folder "
                                                                                   "with generated images")
        self.generated_images_path_label.setText(self.generated_images_folder_path)
        self.log("Selected folder with generated images:" + self.generated_images_folder_path)

    @exception_catcher
    def start_comparison(self, *args, **kwargs) -> None:
        app.save_app_settings(self)
        selected_model = self.model_selection_combo.currentText()
        selected_metric = self.metric_selection_combo.currentText()
        selected_backend = self.backend_selection_combo.currentText()
        if not self.org_images_folder_path or not self.generated_images_folder_path:
            self.log("Please select both original and generated image folders.")
            return

        self.comparison_thread = ComparisonThread(self.org_images_folder_path, self.generated_images_folder_path,
                                                  selected_model, selected_metric, selected_backend)
        self.comparison_thread.progress_updated.connect(self.update_progress)
        self.comparison_thread.comparison_finished.connect(self.comparison_finished)
        self.comparison_thread.log_message.connect(self.log)
        self.comparison_thread.start()

    def log(self, message: str) -> None:
        self.log_text_edit.append(message)

    @exception_catcher
    def update_progress(self, value: int) -> None:
        self.progress_bar.setValue(value)

    @exception_catcher
    def comparison_finished(self, results) -> None:
        self.progress_bar.setValue(100)
        self.log("Image comparison finished.")
        self.results = results

    @exception_catcher
    def export_as_json(self, *args, **kwargs) -> None:
        if self.results is None:
            self.log("No results to export.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save Results", filter="JSON Files (*.json)")
        if filename:
            # Преобразование всех значений float32 в float
            results_to_save = []
            for result in self.results:
                # Для каждого результата преобразуем все числа с плавающей точкой в стандартный тип float
                result_converted = [float(value) if isinstance(value, (float, np.float32)) else value for value in
                                    result]
                results_to_save.append(result_converted)

            with open(filename, 'w') as file:
                json.dump(results_to_save, file, indent=4)
            self.log(f"Results exported as JSON to {filename}.")

    @exception_catcher
    def export_as_excel(self, *args, **kwargs) -> None:
        try:
            if self.results is None:
                self.log("No results to export.")
                return
            filename, _ = QFileDialog.getSaveFileName(self, "Save Results", filter="Excel Files (*.xlsx)")
            if filename:
                columns = ['Original Image', 'Generated Image', 'Model', 'Similarity Metric', 'Face Similarity',
                           'PickScore', 'BRISQUE Score', 'CLIP-IQA Score']
                df = pd.DataFrame(self.results, columns=columns)
                df.to_excel(filename, index=False)
                self.log(f"Results exported as Excel to {filename}.")
        except Exception as e:
            self.log(f"Ошибка: {e}")


class ComparisonThread(QThread):
    progress_updated = pyqtSignal(int)
    comparison_finished = pyqtSignal(list)
    log_message = pyqtSignal(str)

    def __init__(self, org_images_folder_path, generated_images_folder_path, model, metric, backend):
        super().__init__(None)
        self.org_images_folder_path = org_images_folder_path
        self.generated_images_folder_path = generated_images_folder_path
        self.model = model
        self.metric = metric
        self.backend = backend

    @exception_catcher
    def run(self, *args, **kwargs) -> None:
        results = []
        processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        model_clip = AutoModel.from_pretrained(r"C:\Users\Max\Desktop\deepface_script\PickScore_v1").eval().to('cuda')

        org_images = [image for image in os.listdir(self.org_images_folder_path)
                      if os.path.isfile(os.path.join(self.org_images_folder_path, image))]

        generated_images = [image for image in os.listdir(self.generated_images_folder_path)
                            if os.path.isfile(os.path.join(self.generated_images_folder_path, image))]

        total = len(org_images) * len(generated_images)
        count = 0

        for org_image in org_images:
            org_image_path = os.path.join(self.org_images_folder_path, org_image)
            for gen_image in generated_images:
                gen_image_path = os.path.join(self.generated_images_folder_path, gen_image)
                try:
                    analysis = DeepFace.verify(org_image_path, gen_image_path, model_name=self.model,
                                               distance_metric=self.metric, detector_backend=self.backend)
                    face_similarity = analysis["distance"]
                except Exception as e:
                    self.log_message.emit(f"Error comparing {org_image} and {gen_image}: {e}")
                    face_similarity = None

                # PickScore calculation
                with Image.open(gen_image_path) as image:
                    pil_images = [image]
                    pick_score = self.calc_pick_score(pil_images, model_clip, processor)
                    score_brisque, score_clip_iqa = self.compute_additional_metrics(image)

                results.append([
                    org_image, gen_image, self.model, self.metric, face_similarity,
                    pick_score, score_brisque, score_clip_iqa
                ])

                count += 1
                progress = int((count / total) * 100)
                self.progress_updated.emit(progress)

        self.comparison_finished.emit(results)

    @exception_catcher
    def calc_pick_score(self, images: list[Image.Image], model: AutoModel.from_pretrained,
                        processor: AutoProcessor.from_pretrained) -> int | float:

        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        text_inputs = processor(text=[""], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            # embed
            image_embs = model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            normalized_text_embs = F.normalize(text_embs, p=2, dim=-1)
            normalized_image_embs = F.normalize(image_embs, p=2, dim=-1)
            cosine_similarity = F.cosine_similarity(normalized_text_embs, normalized_image_embs)

            # score
            scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            # normalization within 15-25 interval just for convenient view
            scores_norm = (scores.item()-15)/(25-15)
            # get probabilities if you have multiple images to choose from
            probs = torch.softmax(scores, dim=-1)

        return scores_norm

    @exception_catcher
    @torch.no_grad()
    def compute_additional_metrics(self, image: Image.Image) -> tuple[float, float]:
        """
        CLIP-IQA is considered a relative metric,
        so calculation is processed image by image to make it suitable for comparing
        """
        image_tensor = torchvision.transforms.functional.pil_to_tensor(image).unsqueeze(0)
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        else:
            image_tensor = image_tensor.cpu()

        # Вычисление BRISQUE
        score_brisque = piq.brisque(image_tensor, data_range=255, reduction='none').item()

        # Вычисление CLIP-IQA
        clip_iqa = piq.CLIPIQA(data_range=255).to(image_tensor.device)
        score_clip_iqa = clip_iqa(image_tensor).item()

        return score_brisque, score_clip_iqa


class App(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        try:
            with open(CONFIG_FILE, 'r') as file:
                self.config = json.load(file)
        except Exception:
            self.config = None

    @staticmethod
    def save_app_settings(app_instance=ImageComparisonApp) -> None:
        fresh_config = {
            'org_images_folder_path': app_instance.org_images_folder_path,
            'generated_images_folder_path': app_instance.generated_images_folder_path,
            'selected_model': app_instance.model_selection_combo.currentText(),
            'selected_metric': app_instance.metric_selection_combo.currentText(),
            'selected_backend': app_instance.backend_selection_combo.currentText(),
            }
        with open(CONFIG_FILE, 'w') as file:
            json.dump(fresh_config, file, indent=4)

    def run(self, window: ImageComparisonApp) -> None:
        window.show()
        sys.exit(self.exec_())


if __name__ == "__main__":
    app = App(sys.argv)
    gui = ImageComparisonApp()
    app.run(gui)
