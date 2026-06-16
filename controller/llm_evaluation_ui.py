import os
import json
import base64
import urllib.request
import urllib.error
import shutil
from pathlib import Path
import re
import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QTextEdit, QProgressBar, QFileDialog, QSplitter,
    QGroupBox, QMessageBox, QSizePolicy, QListWidget, QListWidgetItem,
    QGridLayout, QScrollArea, QSpacerItem
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QFont


class OllamaWorker(QThread):
    finished = Signal(dict)
    
    def __init__(self, model: str, prompt: str, image_paths: list = None):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.image_paths = image_paths or []

    def run(self):
        url = "http://127.0.0.1:11434/api/generate"
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "stream": False
        }
        
        # Support multiple images
        if self.image_paths:
            images_b64 = []
            for img_path in self.image_paths:
                if os.path.exists(img_path):
                    try:
                        with open(img_path, "rb") as img_file:
                            b64_str = base64.b64encode(img_file.read()).decode("utf-8")
                            images_b64.append(b64_str)
                    except Exception as e:
                        pass
            if images_b64:
                payload["images"] = images_b64
                
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode())
                self.finished.emit({"success": True, "response": result.get("response", "")})
        except Exception as e:
            self.finished.emit({"success": False, "error": str(e)})


class ImageThumbnail(QWidget):
    """Widget for displaying image thumbnail with remove button"""
    remove_clicked = Signal(str)
    
    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Image label
        self.thumb_label = QLabel()
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setStyleSheet("border: 1px solid #ccc; background: #f5f5f5;")
        self.thumb_label.setFixedSize(120, 120)
        
        # Load and scale image
        pixmap = QPixmap(self.image_path).scaled(
            110, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.thumb_label.setPixmap(pixmap)
        
        # Filename label
        self.name_label = QLabel(os.path.basename(self.image_path)[:15] + "...")
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet("font-size: 10px; color: #666;")
        
        # Remove button
        self.remove_btn = QPushButton("X")
        self.remove_btn.setFixedSize(24, 24)
        self.remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.remove_btn.clicked.connect(self.on_remove)
        
        layout.addWidget(self.thumb_label)
        layout.addWidget(self.name_label)
        layout.addWidget(self.remove_btn, alignment=Qt.AlignRight)
        
        self.setLayout(layout)
        
    def on_remove(self):
        self.remove_clicked.emit(self.image_path)


class LLMEvaluationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.knowledge_dir = Path("data/ai_knowledge")
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.categories = ["haircut", "waiting", "washing", "other"]
        for cat in self.categories:
            (self.knowledge_dir / cat).mkdir(exist_ok=True)
            
        self.selected_images: list[str] = []
        self.worker = None
        
        self.setup_ui()
        self.load_models()
        self.update_knowledge_counts()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # 1. Top Bar: Model Selection & Fetch
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Ollama Model (VLM or LLM):"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        top_bar.addWidget(self.model_combo)
        
        refresh_models_btn = QPushButton("Refresh Models")
        refresh_models_btn.clicked.connect(self.load_models)
        top_bar.addWidget(refresh_models_btn)
        top_bar.addStretch()
        layout.addLayout(top_bar)
        
        # Splitter for Main Content
        splitter = QSplitter(Qt.Horizontal)
        
        # ---------------- LEFT PANEL: Image Selection & Test Area ----------------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Image selection buttons
        btn_layout = QHBoxLayout()
        
        self.add_images_btn = QPushButton("📁 Add Images")
        self.add_images_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        self.add_images_btn.clicked.connect(self.add_images)
        btn_layout.addWidget(self.add_images_btn)
        
        self.clear_images_btn = QPushButton("🗑 Clear All")
        self.clear_images_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 8px;")
        self.clear_images_btn.clicked.connect(self.clear_images)
        btn_layout.addWidget(self.clear_images_btn)
        
        btn_layout.addStretch()
        left_layout.addLayout(btn_layout)
        
        # Image count label
        self.image_count_label = QLabel("Selected: 0 images")
        self.image_count_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        left_layout.addWidget(self.image_count_label)
        
        # Scrollable image grid
        self.images_scroll = QScrollArea()
        self.images_scroll.setWidgetResizable(True)
        self.images_scroll.setMaximumHeight(250)
        self.images_scroll.setStyleSheet("QScrollArea { border: 2px dashed #aaa; background: #f9f9f9; }")
        
        self.images_container = QWidget()
        self.images_grid = QGridLayout()
        self.images_container.setLayout(self.images_grid)
        self.images_scroll.setWidget(self.images_container)
        
        left_layout.addWidget(self.images_scroll)
        
        # Analyze Button
        self.analyze_btn = QPushButton("🔍 Analyze All Images")
        self.analyze_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 12px; font-size: 16px;")
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.analyze_btn.setEnabled(False)
        left_layout.addWidget(self.analyze_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        left_layout.addWidget(self.progress_bar)
        
        splitter.addWidget(left_panel)
        
        # ---------------- RIGHT PANEL: Results & Feedback ----------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Result Box
        res_group = QGroupBox("Analysis Results")
        res_layout = QVBoxLayout()
        
        # Summary results
        self.summary_label = QLabel("Summary: 0 images processed")
        self.summary_label.setFont(QFont("Arial", 14, QFont.Bold))
        res_layout.addWidget(self.summary_label)
        
        # Per-image results
        self.results_list = QListWidget()
        self.results_list.setAlternatingRowColors(True)
        self.results_list.setStyleSheet("""
            QListWidget {
                font-family: monospace;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #eee;
            }
        """)
        res_layout.addWidget(self.results_list)
        
        # Raw output
        self.raw_output_text = QTextEdit()
        self.raw_output_text.setReadOnly(True)
        self.raw_output_text.setMaximumHeight(200)
        self.raw_output_text.setPlaceholderText("Detailed LLM output will appear here...")
        res_layout.addWidget(self.raw_output_text)
        
        res_group.setLayout(res_layout)
        right_layout.addWidget(res_group)
        
        # Feedback / Continuous Learning Loop Box
        fb_group = QGroupBox("Continuous Learning (Instant Feedback)")
        fb_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        fb_layout = QVBoxLayout()
        fb_layout.addWidget(QLabel("Is the AI wrong? Correct it below to save image to Knowledge Base."))
        
        btn_layout = QHBoxLayout()
        for cat in self.categories:
            btn = QPushButton(f"Correct to: {cat.upper()}")
            btn.clicked.connect(lambda checked=False, c=cat: self.save_feedback(c))
            btn_layout.addWidget(btn)
        fb_layout.addLayout(btn_layout)
        
        # Knowledge counts
        self.knowledge_counts_label = QLabel("Knowledge Base: 0 images")
        self.knowledge_counts_label.setStyleSheet("color: #555; margin-top: 10px;")
        fb_layout.addWidget(self.knowledge_counts_label)
        
        fb_group.setLayout(fb_layout)
        right_layout.addWidget(fb_group)
        
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
        self.setLayout(layout)
        
    def load_models(self):
        self.model_combo.clear()
        try:
            req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
            with urllib.request.urlopen(req, timeout=3) as response:
                data = json.loads(response.read().decode())
                models = [m["name"] for m in data.get("models", [])]
                self.model_combo.addItems(models)
                
                # Check if specific model exists to set as default
                for idx, m in enumerate(models):
                    if "qwen2.5-coder" in m or "llava" in m or "llama3" in m:
                        self.model_combo.setCurrentIndex(idx)
                        break
        except Exception as e:
            self.model_combo.addItem("Error (Is Ollama Running?)")

    def update_knowledge_counts(self):
        parts = []
        for cat in self.categories:
            count = len(list((self.knowledge_dir / cat).glob("*.*")))
            parts.append(f"{cat}: {count}")
        self.knowledge_counts_label.setText("Knowledge Base | " + " | ".join(parts))

    def add_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, 
            "Select Test Images", 
            "", 
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        
        if file_paths:
            for path in file_paths:
                if path not in self.selected_images:
                    self.selected_images.append(path)
            
            self.refresh_image_grid()
            self.update_image_count()
            self.analyze_btn.setEnabled(len(self.selected_images) > 0)
            
    def refresh_image_grid(self):
        # Clear existing widgets
        while self.images_grid.count():
            item = self.images_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add thumbnails
        cols = 5
        for idx, img_path in enumerate(self.selected_images):
            row = idx // cols
            col = idx % cols
            
            thumb = ImageThumbnail(img_path)
            thumb.remove_clicked.connect(self.remove_image)
            self.images_grid.addWidget(thumb, row, col, Qt.AlignCenter)
            
        # Add stretch to fill empty space
        if len(self.selected_images) < cols:
            spacer = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
            self.images_grid.addItem(
                spacer, 
                0, 
                len(self.selected_images), 
                1, 
                cols - len(self.selected_images)
            )
    
    def remove_image(self, image_path: str):
        if image_path in self.selected_images:
            self.selected_images.remove(image_path)
            self.refresh_image_grid()
            self.update_image_count()
            self.analyze_btn.setEnabled(len(self.selected_images) > 0)
    
    def clear_images(self):
        if self.selected_images:
            reply = QMessageBox.question(
                self,
                "Clear Images",
                f"Remove all {len(self.selected_images)} selected images?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.selected_images.clear()
                self.refresh_image_grid()
                self.update_image_count()
                self.analyze_btn.setEnabled(False)
                self.results_list.clear()
                self.raw_output_text.clear()
                self.summary_label.setText("Summary: 0 images processed")
    
    def update_image_count(self):
        count = len(self.selected_images)
        self.image_count_label.setText(f"Selected: {count} image{'s' if count != 1 else ''}")
        
    def run_analysis(self):
        if not self.selected_images:
            QMessageBox.warning(self, "Warning", "Please select at least one image.")
            return
            
        model = self.model_combo.currentText()
        if not model or "Error" in model:
            QMessageBox.warning(self, "Warning", "Please select a valid model.")
            return
            
        self.analyze_btn.setEnabled(False)
        self.progress_bar.show()
        self.results_list.clear()
        self.raw_output_text.setText(f"Analyzing {len(self.selected_images)} image(s)... Please wait.\n\nThis may take several seconds per image.")
        self.summary_label.setText(f"Processing: {len(self.selected_images)} images...")
        
        # Construct prompt for multiple images
        prompt = (
            f"You are analyzing {len(self.selected_images)} image(s) simultaneously. "
            "For EACH image, identify the person's activity. "
            "Choose ONLY ONE from: [HAIRCUT, WAITING, WASHING, OTHER]. "
            "Also provide a confidence score from 0 to 100 for EACH image. "
            "Format your response EXACTLY like this for each image:\n\n"
            "IMAGE 1:\nCLASS: <your_choice>\nCONFIDENCE: <number>\n\n"
            "IMAGE 2:\nCLASS: <your_choice>\nCONFIDENCE: <number>\n\n"
            "(continue for all images)\n\n"
            "Hint: Haircut usually involves a barber cape. "
            "Washing involves a sink/basin. "
            "Waiting involves sitting casually on sofas."
        )
        
        self.worker = OllamaWorker(
            model=model, 
            prompt=prompt, 
            image_paths=self.selected_images
        )
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.start()

    def on_analysis_finished(self, result):
        self.analyze_btn.setEnabled(True)
        self.progress_bar.hide()
        
        if result.get("success"):
            response = result.get("response", "")
            self.raw_output_text.setText(response)
            
            # Parse results for each image
            results = self.parse_multi_image_results(response)
            
            # Update results list
            for idx, (img_path, pred_class, confidence) in enumerate(results, 1):
                item_text = f"[{idx}] {os.path.basename(img_path)[:30]}... => {pred_class} ({confidence}%)"
                item = QListWidgetItem(item_text)
                
                # Color coding
                if "HAIRCUT" in pred_class:
                    item.setForeground(Qt.darkYellow)
                elif "WASHING" in pred_class:
                    item.setForeground(Qt.blue)
                elif "WAITING" in pred_class:
                    item.setForeground(Qt.darkGreen)
                else:
                    item.setForeground(Qt.darkGray)
                
                self.results_list.addItem(item)
            
            # Update summary
            class_counts = {}
            for _, pred_class, _ in results:
                class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
            
            summary_parts = [f"{cls}: {cnt}" for cls, cnt in sorted(class_counts.items())]
            self.summary_label.setText(f"Summary: {len(results)} images | " + " | ".join(summary_parts))
            
        else:
            self.raw_output_text.setText(f"Error: {result.get('error')}")
            self.summary_label.setText("Error during analysis")
            
    def parse_multi_image_results(self, response: str) -> list:
        """Parse LLM response for multiple images"""
        results = []
        
        # Pattern to match IMAGE N: blocks
        image_pattern = r'IMAGE\s*(\d+):?\s*CLASS:\s*([A-Za-z]+)\s*CONFIDENCE:\s*([0-9]+)'
        matches = re.findall(image_pattern, response, re.IGNORECASE)
        
        if matches:
            for idx, pred_class, confidence in matches:
                results.append((
                    self.selected_images[min(int(idx) - 1, len(self.selected_images) - 1)] if idx.isdigit() else None,
                    pred_class.upper(),
                    confidence
                ))
        
        # Fallback: if no structured output, try to find any CLASS/CONFIDENCE pairs
        if not results:
            class_matches = re.findall(r'CLASS:\s*([A-Za-z]+)', response, re.IGNORECASE)
            conf_matches = re.findall(r'CONFIDENCE:\s*([0-9]+)', response, re.IGNORECASE)
            
            for i in range(min(len(class_matches), len(conf_matches), len(self.selected_images))):
                results.append((
                    self.selected_images[i],
                    class_matches[i].upper(),
                    conf_matches[i]
                ))
        
        # Last fallback: assign unknown to all images
        if not results:
            for img_path in self.selected_images:
                results.append((img_path, "UNKNOWN", "0"))
        
        return results
            
    def save_feedback(self, correct_category):
        if not self.selected_images:
            QMessageBox.warning(self, "Warning", "No images to save.")
            return
            
        # Save all selected images to the correct category
        target_dir = self.knowledge_dir / correct_category
        saved_count = 0
        
        for img_path in self.selected_images:
            filename = os.path.basename(img_path)
            safe_filename = f"{int(time.time())}_{filename}"
            dest_path = target_dir / safe_filename
            
            try:
                shutil.copy2(img_path, dest_path)
                saved_count += 1
            except Exception as e:
                pass
        
        if saved_count > 0:
            QMessageBox.information(
                self, 
                "Success", 
                f"{saved_count} image(s) correctly saved to {correct_category.upper()} knowledge base!"
            )
            self.update_knowledge_counts()
        else:
            QMessageBox.critical(self, "Error", "Failed to save images.")
