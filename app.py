#!/usr/bin/env python3
"""
Credit Scoring Model GUI Application
Author: Muhammad Faaiz
Date: 2025-08-18
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, QTextEdit, 
                             QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox,
                             QComboBox, QGroupBox, QFormLayout, QScrollArea, QDoubleSpinBox,
                             QSplitter, QHeaderView, QProgressBar, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

# Import the training functions from the modified train.py
from train import train_model, load_data, predict_new_data, evaluate


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("credit-scoring-gui")


class TrainThread(QThread):
    """Thread for training model to prevent GUI freezing."""
    finished = pyqtSignal(object, object, object, object)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, input_path, test_size=0.2):
        super().__init__()
        self.input_path = input_path
        self.test_size = test_size
    
    def run(self):
        try:
            self.progress.emit("Loading data...")
            model_data, X_test, y_test, fi = train_model(self.input_path, self.test_size)
            self.progress.emit("Training completed successfully!")
            self.finished.emit(model_data, X_test, y_test, fi)
        except Exception as e:
            self.error.emit(str(e))


class CreditScoringApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.feature_importances = None
        self.evaluation_results = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Credit Scoring Model')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Create tabs
        self.train_tab = QWidget()
        self.predict_tab = QWidget()
        self.batch_tab = QWidget()
        self.metrics_tab = QWidget()
        self.features_tab = QWidget()
        self.logs_tab = QWidget()
        
        self.tabs.addTab(self.train_tab, "Train Model")
        self.tabs.addTab(self.predict_tab, "Manual Prediction")
        self.tabs.addTab(self.batch_tab, "Batch Prediction")
        self.tabs.addTab(self.metrics_tab, "Model Metrics")
        self.tabs.addTab(self.features_tab, "Feature Importance")
        self.tabs.addTab(self.logs_tab, "Logs")
        
        # Initialize tabs
        self.initTrainTab()
        self.initPredictTab()
        self.initBatchTab()
        self.initMetricsTab()
        self.initFeaturesTab()
        self.initLogsTab()
        
        # Status bar
        self.statusBar().showMessage('Ready')
        
        # Log text widget for status updates
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        # Redirect logging to the text widget
        class QTextEditHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
                
            def emit(self, record):
                msg = self.format(record)
                self.text_widget.append(msg)
                
        handler = QTextEditHandler(self.log_text)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        
    def initTrainTab(self):
        layout = QVBoxLayout(self.train_tab)
        
        # File selection
        file_group = QGroupBox("Training Data")
        file_layout = QHBoxLayout()
        self.train_file_edit = QLineEdit()
        self.train_file_edit.setPlaceholderText("Select training data CSV file")
        file_browse_btn = QPushButton("Browse")
        file_browse_btn.clicked.connect(self.browseTrainFile)
        file_layout.addWidget(self.train_file_edit)
        file_layout.addWidget(file_browse_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Test size selection
        test_size_group = QGroupBox("Test Size")
        test_size_layout = QHBoxLayout()
        self.test_size_spin = QDoubleSpinBox()
        self.test_size_spin.setRange(0.1, 0.5)
        self.test_size_spin.setSingleStep(0.05)
        self.test_size_spin.setValue(0.2)
        self.test_size_spin.setSuffix(" (20% default)")
        test_size_layout.addWidget(QLabel("Test Size Ratio:"))
        test_size_layout.addWidget(self.test_size_spin)
        test_size_layout.addStretch()
        test_size_group.setLayout(test_size_layout)
        layout.addWidget(test_size_group)
        
        # Train button
        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self.trainModel)
        layout.addWidget(self.train_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.train_status_label = QLabel("No model trained yet.")
        layout.addWidget(self.train_status_label)
        
        layout.addStretch()
        
    def initPredictTab(self):
        layout = QVBoxLayout(self.predict_tab)
        
        # Input form for manual prediction
        form_group = QGroupBox("Applicant Information")
        form_layout = QFormLayout()
        
        # Create input fields for all expected features
        self.input_fields = {}
        
        # Numeric features
        numeric_features = [
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
            "Credit_History", "Total_Income", "DTI"
        ]
        
        # Categorical features
        categorical_features = [
            "Gender", "Married", "Dependents", "Education", "Self_Employed", 
            "Property_Area"
        ]
        
        for feature in numeric_features:
            self.input_fields[feature] = QLineEdit()
            self.input_fields[feature].setPlaceholderText("Enter numeric value")
            form_layout.addRow(feature + ":", self.input_fields[feature])
            
        for feature in categorical_features:
            self.input_fields[feature] = QComboBox()
            if feature == "Gender":
                self.input_fields[feature].addItems(["", "Male", "Female"])
            elif feature == "Married":
                self.input_fields[feature].addItems(["", "Yes", "No"])
            elif feature == "Dependents":
                self.input_fields[feature].addItems(["", "0", "1", "2", "3+"])
            elif feature == "Education":
                self.input_fields[feature].addItems(["", "Graduate", "Not Graduate"])
            elif feature == "Self_Employed":
                self.input_fields[feature].addItems(["", "Yes", "No"])
            elif feature == "Property_Area":
                self.input_fields[feature].addItems(["", "Urban", "Semiurban", "Rural"])
            form_layout.addRow(feature + ":", self.input_fields[feature])
            
        form_group.setLayout(form_layout)
        layout.addWidget(form_group)
        
        # Prediction threshold
        threshold_group = QGroupBox("Prediction Settings")
        threshold_layout = QHBoxLayout()
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.5)
        threshold_layout.addWidget(QLabel("Prediction Threshold:"))
        threshold_layout.addWidget(self.threshold_spin)
        threshold_layout.addStretch()
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)
        
        # Predict button
        self.predict_btn = QPushButton("Predict Creditworthiness")
        self.predict_btn.clicked.connect(self.predictManual)
        layout.addWidget(self.predict_btn)
        
        # Results display
        result_group = QGroupBox("Prediction Results")
        result_layout = QVBoxLayout()
        self.prediction_result = QLabel("No prediction made yet.")
        self.prediction_result.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.probability_result = QLabel("")
        result_layout.addWidget(self.prediction_result)
        result_layout.addWidget(self.probability_result)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        layout.addStretch()
        
    def initBatchTab(self):
        layout = QVBoxLayout(self.batch_tab)
        
        # File selection
        file_group = QGroupBox("Batch Prediction File")
        file_layout = QHBoxLayout()
        self.batch_file_edit = QLineEdit()
        self.batch_file_edit.setPlaceholderText("Select CSV file for batch prediction")
        batch_browse_btn = QPushButton("Browse")
        batch_browse_btn.clicked.connect(self.browseBatchFile)
        file_layout.addWidget(self.batch_file_edit)
        file_layout.addWidget(batch_browse_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Threshold selection
        threshold_group = QGroupBox("Prediction Settings")
        threshold_layout = QHBoxLayout()
        self.batch_threshold_spin = QDoubleSpinBox()
        self.batch_threshold_spin.setRange(0.0, 1.0)
        self.batch_threshold_spin.setSingleStep(0.05)
        self.batch_threshold_spin.setValue(0.5)
        threshold_layout.addWidget(QLabel("Prediction Threshold:"))
        threshold_layout.addWidget(self.batch_threshold_spin)
        threshold_layout.addStretch()
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)
        
        # Include probabilities checkbox
        self.include_probs_check = QCheckBox("Include probabilities in output")
        self.include_probs_check.setChecked(True)
        layout.addWidget(self.include_probs_check)
        
        # Predict button
        self.batch_predict_btn = QPushButton("Run Batch Prediction")
        self.batch_predict_btn.clicked.connect(self.predictBatch)
        layout.addWidget(self.batch_predict_btn)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(0)
        self.results_table.setRowCount(0)
        layout.addWidget(self.results_table)
        
        # Export button
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self.exportResults)
        self.export_btn.setEnabled(False)
        layout.addWidget(self.export_btn)
        
        layout.addStretch()
        
    def initMetricsTab(self):
        layout = QVBoxLayout(self.metrics_tab)
        
        # Metrics display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        layout.addWidget(self.metrics_text)
        
        # ROC AUC and PR curves
        curves_group = QGroupBox("Performance Curves")
        curves_layout = QHBoxLayout()
        
        self.roc_curve_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        self.pr_curve_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        
        curves_layout.addWidget(self.roc_curve_canvas)
        curves_layout.addWidget(self.pr_curve_canvas)
        curves_group.setLayout(curves_layout)
        layout.addWidget(curves_group)
        
        # Confusion matrices
        matrices_group = QGroupBox("Confusion Matrices")
        matrices_layout = QHBoxLayout()
        
        self.cm_05_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        self.cm_best_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        
        matrices_layout.addWidget(self.cm_05_canvas)
        matrices_layout.addWidget(self.cm_best_canvas)
        matrices_group.setLayout(matrices_layout)
        layout.addWidget(matrices_group)
        
    def initFeaturesTab(self):
        layout = QVBoxLayout(self.features_tab)
        
        # Feature importance chart
        self.feature_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        layout.addWidget(self.feature_canvas)
        
        # Feature importance table
        self.feature_table = QTableWidget()
        self.feature_table.setColumnCount(2)
        self.feature_table.setHorizontalHeaderLabels(["Feature", "Importance"])
        self.feature_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.feature_table)
        
    def initLogsTab(self):
        layout = QVBoxLayout(self.logs_tab)
        
        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        layout.addWidget(self.log_display)
        
        # Clear logs button
        clear_btn = QPushButton("Clear Logs")
        clear_btn.clicked.connect(self.clearLogs)
        layout.addWidget(clear_btn)
        
    def browseTrainFile(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Training Data", "", "CSV Files (*.csv)"
        )
        if file_path:
            self.train_file_edit.setText(file_path)
            
    def browseBatchFile(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Batch Prediction File", "", "CSV Files (*.csv)"
        )
        if file_path:
            self.batch_file_edit.setText(file_path)
            
    def trainModel(self):
        if not self.train_file_edit.text():
            QMessageBox.warning(self, "Warning", "Please select a training data file.")
            return
            
        # Disable train button and show progress
        self.train_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.train_status_label.setText("Training in progress...")
        
        # Create and start training thread
        self.train_thread = TrainThread(
            self.train_file_edit.text(), 
            self.test_size_spin.value()
        )
        self.train_thread.finished.connect(self.onTrainingFinished)
        self.train_thread.progress.connect(self.onTrainingProgress)
        self.train_thread.error.connect(self.onTrainingError)
        self.train_thread.start()
        
    def onTrainingProgress(self, message):
        self.statusBar().showMessage(message)
        logger.info(message)
        
    def onTrainingFinished(self, model_data, X_test, y_test, fi):
        self.model = model_data
        self.feature_importances = fi
        self.evaluation_results = model_data["evaluation_results"]
        
        # Enable train button and hide progress
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.train_status_label.setText("Training completed successfully!")
        
        # Update metrics and features tabs
        self.updateMetricsTab()
        self.updateFeaturesTab()
        
        # Enable prediction tabs
        self.tabs.setTabEnabled(1, True)  # Manual Prediction
        self.tabs.setTabEnabled(2, True)  # Batch Prediction
        
        logger.info("Model training completed successfully!")
        
    def onTrainingError(self, error_message):
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.train_status_label.setText("Training failed.")
        QMessageBox.critical(self, "Error", f"Training failed: {error_message}")
        logger.error(f"Training error: {error_message}")
        
    def predictManual(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please train a model first.")
            return
            
        try:
            # Collect input data
            input_data = {}
            for feature, widget in self.input_fields.items():
                if isinstance(widget, QLineEdit):
                    value = widget.text().strip()
                    input_data[feature] = float(value) if value else 0.0
                elif isinstance(widget, QComboBox):
                    input_data[feature] = widget.currentText()
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            threshold = self.threshold_spin.value()
            prediction, probability = predict_new_data(
                self.model["pipeline"], input_df, threshold
            )
            
            # Display results
            result = "Creditworthy" if prediction[0] == 1 else "Not Creditworthy"
            color = "green" if prediction[0] == 1 else "red"
            self.prediction_result.setText(f"Prediction: <font color='{color}'>{result}</font>")
            self.probability_result.setText(
                f"Probability: {probability[0]:.4f} (Threshold: {threshold:.2f})"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")
            logger.error(f"Prediction error: {str(e)}")
            
    def predictBatch(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please train a model first.")
            return
            
        if not self.batch_file_edit.text():
            QMessageBox.warning(self, "Warning", "Please select a batch prediction file.")
            return
            
        try:
            # Load data
            df = load_data(self.batch_file_edit.text())
            
            # Check if target column exists and remove it
            if "Loan_Status" in df.columns:
                df = df.drop(columns=["Loan_Status"])
            if "creditworthy" in df.columns:
                df = df.drop(columns=["creditworthy"])
                
            # Make predictions
            threshold = self.batch_threshold_spin.value()
            predictions, probabilities = predict_new_data(
                self.model["pipeline"], df, threshold
            )
            
            # Prepare results
            results_df = df.copy()
            results_df["Prediction"] = ["Creditworthy" if p == 1 else "Not Creditworthy" for p in predictions]
            results_df["Probability"] = probabilities
            
            # Display in table
            self.displayResultsTable(results_df)
            
            # Store results for export
            self.batch_results = results_df
            self.export_btn.setEnabled(True)
            
            logger.info(f"Batch prediction completed for {len(df)} records.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Batch prediction failed: {str(e)}")
            logger.error(f"Batch prediction error: {str(e)}")
            
    def displayResultsTable(self, results_df):
        # Set up table
        self.results_table.setColumnCount(len(results_df.columns))
        self.results_table.setRowCount(len(results_df))
        self.results_table.setHorizontalHeaderLabels(results_df.columns)
        
        # Populate table
        for i, row in results_df.iterrows():
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                # Color code predictions
                if results_df.columns[j] == "Prediction":
                    if value == "Creditworthy":
                        item.setBackground(QColor(200, 255, 200))
                    else:
                        item.setBackground(QColor(255, 200, 200))
                self.results_table.setItem(i, j, item)
                
        # Resize columns to content
        self.results_table.resizeColumnsToContents()
        
    def exportResults(self):
        if not hasattr(self, 'batch_results'):
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "CSV Files (*.csv)"
        )
        
        if file_path:
            # Prepare export data
            export_df = self.batch_results.copy()
            if not self.include_probs_check.isChecked():
                export_df = export_df.drop(columns=["Probability"])
                
            export_df.to_csv(file_path, index=False)
            logger.info(f"Results exported to {file_path}")
            
    def updateMetricsTab(self):
        if self.evaluation_results is None:
            return
            
        # Update metrics text
        metrics_text = f"""
        Model Performance Metrics:
        
        ROC-AUC: {self.evaluation_results['roc_auc']:.4f}
        PR-AUC: {self.evaluation_results['pr_auc']:.4f}
        Best Threshold: {self.evaluation_results['best_threshold']:.3f}
        
        At Default Threshold (0.5):
          Accuracy: {self.evaluation_results['metrics_05']['accuracy']:.4f}
          F1 Score: {self.evaluation_results['metrics_05']['f1']:.4f}
          
        At Best Threshold ({self.evaluation_results['best_threshold']:.3f}):
          Accuracy: {self.evaluation_results['metrics_best']['accuracy']:.4f}
          F1 Score: {self.evaluation_results['metrics_best']['f1']:.4f}
        """
        
        self.metrics_text.setPlainText(metrics_text)
        
        # Plot ROC curve
        self.plotROCCurve()
        
        # Plot PR curve
        self.plotPRCurve()
        
        # Plot confusion matrices
        self.plotConfusionMatrices()
        
    def plotROCCurve(self):
        # For simplicity, we'll just show a placeholder
        # In a real implementation, you would plot the actual ROC curve
        ax = self.roc_curve_canvas.figure.subplots()
        ax.set_title('ROC Curve')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.legend()
        self.roc_curve_canvas.draw()
        
    def plotPRCurve(self):
        # For simplicity, we'll just show a placeholder
        # In a real implementation, you would plot the actual PR curve
        ax = self.pr_curve_canvas.figure.subplots()
        ax.set_title('Precision-Recall Curve')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        self.pr_curve_canvas.draw()
        
    def plotConfusionMatrices(self):
        # Plot confusion matrix at 0.5 threshold
        ax1 = self.cm_05_canvas.figure.subplots()
        cm_05 = np.array(self.evaluation_results['metrics_05']['confusion_matrix'])
        ax1.matshow(cm_05, cmap='Blues', alpha=0.7)
        for i in range(cm_05.shape[0]):
            for j in range(cm_05.shape[1]):
                ax1.text(x=j, y=i, s=cm_05[i, j], va='center', ha='center', size='large')
        ax1.set_title('Confusion Matrix (Threshold=0.5)')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        self.cm_05_canvas.draw()
        
        # Plot confusion matrix at best threshold
        ax2 = self.cm_best_canvas.figure.subplots()
        cm_best = np.array(self.evaluation_results['metrics_best']['confusion_matrix'])
        ax2.matshow(cm_best, cmap='Blues', alpha=0.7)
        for i in range(cm_best.shape[0]):
            for j in range(cm_best.shape[1]):
                ax2.text(x=j, y=i, s=cm_best[i, j], va='center', ha='center', size='large')
        ax2.set_title(f'Confusion Matrix (Threshold={self.evaluation_results["best_threshold"]:.3f})')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        self.cm_best_canvas.draw()
        
    def updateFeaturesTab(self):
        if self.feature_importances is None:
            return
            
        # Plot feature importance
        ax = self.feature_canvas.figure.subplots()
        top_features = self.feature_importances.head(10)
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['importance'], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Feature Importances')
        self.feature_canvas.draw()
        
        # Update feature table
        self.feature_table.setRowCount(len(self.feature_importances))
        for i, row in self.feature_importances.iterrows():
            self.feature_table.setItem(i, 0, QTableWidgetItem(row['feature']))
            self.feature_table.setItem(i, 1, QTableWidgetItem(f"{row['importance']:.6f}"))
            
    def clearLogs(self):
        self.log_display.clear()
        

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style
    
    window = CreditScoringApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()