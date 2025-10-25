"""
Model Evaluation Utilities

This module provides comprehensive evaluation utilities for the ACE framework,
including metrics calculation, visualization, and performance analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation utilities."""
    
    def __init__(self, save_plots: bool = True, plot_dir: str = "plots"):
        """
        Initialize model evaluator.
        
        Args:
            save_plots: Whether to save evaluation plots
            plot_dir: Directory to save plots
        """
        self.save_plots = save_plots
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def calculate_metrics(self, y_true: List[int], y_pred: List[int], y_proba: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics for binary and multi-class classification.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)

        Returns:
            Dictionary of evaluation metrics
        """
        # Determine if binary or multi-class
        num_classes = len(set(y_true))
        is_binary = num_classes == 2

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro')
        }

        # Add per-class metrics for multi-class
        if not is_binary:
            precision_per_class = precision_score(y_true, y_pred, average=None)
            recall_per_class = recall_score(y_true, y_pred, average=None)
            f1_per_class = f1_score(y_true, y_pred, average=None)

            class_names = ['normal', 'bullying', 'harassment', 'womens_harassment', 'mocking', 'threats', 'hate_speech']
            for i, class_name in enumerate(class_names[:num_classes]):
                metrics[f'precision_{class_name}'] = precision_per_class[i]
                metrics[f'recall_{class_name}'] = recall_per_class[i]
                metrics[f'f1_{class_name}'] = f1_per_class[i]

        # Add probability-based metrics if available
        if y_proba is not None:
            if is_binary:
                metrics.update({
                    'auc_roc': self._calculate_auc_roc(y_true, y_proba),
                    'auc_pr': self._calculate_auc_pr(y_true, y_proba)
                })
            else:
                # For multi-class, calculate AUC for each class vs rest
                auc_roc_scores = []
                auc_pr_scores = []
                for i in range(num_classes):
                    y_true_binary = [1 if label == i else 0 for label in y_true]
                    if len(set(y_true_binary)) > 1:  # Only calculate if class exists
                        auc_roc_scores.append(self._calculate_auc_roc(y_true_binary, [prob[i] for prob in y_proba]))
                        auc_pr_scores.append(self._calculate_auc_pr(y_true_binary, [prob[i] for prob in y_proba]))

                if auc_roc_scores:
                    metrics['auc_roc_macro'] = np.mean(auc_roc_scores)
                    metrics['auc_pr_macro'] = np.mean(auc_pr_scores)

        return metrics
    
    def _calculate_auc_roc(self, y_true: List[int], y_proba: List[float]) -> float:
        """Calculate AUC-ROC score."""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            return auc(fpr, tpr)
        except Exception as e:
            logger.warning(f"Could not calculate AUC-ROC: {e}")
            return 0.0
    
    def _calculate_auc_pr(self, y_true: List[int], y_proba: List[float]) -> float:
        """Calculate AUC-PR score."""
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            return auc(recall, precision)
        except Exception as e:
            logger.warning(f"Could not calculate AUC-PR: {e}")
            return 0.0
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                            model_name: str = "Model", save_path: Optional[str] = None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Harassment'],
                   yticklabels=['Normal', 'Harassment'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path or self.save_plots:
            path = save_path or self.plot_dir / f"{model_name.lower()}_confusion_matrix.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {path}")
        
        plt.show()
    
    def plot_roc_curve(self, y_true: List[int], y_proba: List[float], 
                      model_name: str = "Model", save_path: Optional[str] = None):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            model_name: Name of the model
            save_path: Path to save the plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        
        if save_path or self.save_plots:
            path = save_path or self.plot_dir / f"{model_name.lower()}_roc_curve.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: List[int], y_proba: List[float], 
                                  model_name: str = "Model", save_path: Optional[str] = None):
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            model_name: Name of the model
            save_path: Path to save the plot
        """
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        
        if save_path or self.save_plots:
            path = save_path or self.plot_dir / f"{model_name.lower()}_pr_curve.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {path}")
        
        plt.show()
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                            model_name: str = "Model", save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
            model_name: Name of the model
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {model_name}', fontsize=16)
        
        # Plot losses
        if 'train_losses' in history and 'val_losses' in history:
            axes[0, 0].plot(history['train_losses'], label='Training Loss')
            axes[0, 0].plot(history['val_losses'], label='Validation Loss')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
        
        # Plot accuracies
        if 'train_accuracies' in history and 'val_accuracies' in history:
            axes[0, 1].plot(history['train_accuracies'], label='Training Accuracy')
            axes[0, 1].plot(history['val_accuracies'], label='Validation Accuracy')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
        
        # Plot learning rate (if available)
        if 'learning_rates' in history:
            axes[1, 0].plot(history['learning_rates'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
        
        # Plot other metrics (if available)
        if 'f1_scores' in history:
            axes[1, 1].plot(history['f1_scores'], label='F1 Score')
            axes[1, 1].set_title('F1 Score')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path or self.save_plots:
            path = save_path or self.plot_dir / f"{model_name.lower()}_training_history.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history saved to {path}")
        
        plt.show()
    
    def compare_models(self, results: Dict[str, Dict[str, float]], 
                      save_path: Optional[str] = None):
        """
        Compare multiple models.
        
        Args:
            results: Dictionary of model results
            save_path: Path to save the comparison plot
        """
        # Extract metrics
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Create comparison DataFrame
        comparison_data = []
        for model in models:
            for metric in metrics:
                comparison_data.append({
                    'Model': model,
                    'Metric': metric,
                    'Score': results[model].get(metric, 0.0)
                })
        
        df = pd.DataFrame(comparison_data)
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df, x='Metric', y='Score', hue='Model')
        plt.title('Model Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if save_path or self.save_plots:
            path = save_path or self.plot_dir / "model_comparison.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison saved to {path}")
        
        plt.show()
    
    def generate_report(self, results: Dict[str, Any], 
                       save_path: str = "evaluation_report.json"):
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Evaluation results
            save_path: Path to save the report
        """
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'results': results,
            'summary': self._generate_summary(results)
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {save_path}")
        
        return report
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of evaluation results."""
        summary = {
            'best_model': None,
            'best_accuracy': 0.0,
            'model_count': len(results),
            'average_accuracy': 0.0
        }
        
        if results:
            accuracies = [result.get('accuracy', 0.0) for result in results.values()]
            summary['average_accuracy'] = np.mean(accuracies)
            
            best_model = max(results.keys(), key=lambda k: results[k].get('accuracy', 0.0))
            summary['best_model'] = best_model
            summary['best_accuracy'] = results[best_model].get('accuracy', 0.0)
        
        return summary
    
    def analyze_predictions(self, texts: List[str], y_true: List[int], 
                          y_pred: List[int], model_name: str = "Model") -> Dict[str, Any]:
        """
        Analyze prediction patterns.
        
        Args:
            texts: Input texts
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            
        Returns:
            Analysis results
        """
        analysis = {
            'false_positives': [],
            'false_negatives': [],
            'true_positives': [],
            'true_negatives': []
        }
        
        for i, (text, true_label, pred_label) in enumerate(zip(texts, y_true, y_pred)):
            if true_label == 1 and pred_label == 1:
                analysis['true_positives'].append(text)
            elif true_label == 0 and pred_label == 0:
                analysis['true_negatives'].append(text)
            elif true_label == 1 and pred_label == 0:
                analysis['false_negatives'].append(text)
            elif true_label == 0 and pred_label == 1:
                analysis['false_positives'].append(text)
        
        # Calculate counts
        analysis['counts'] = {
            'true_positives': len(analysis['true_positives']),
            'true_negatives': len(analysis['true_negatives']),
            'false_positives': len(analysis['false_positives']),
            'false_negatives': len(analysis['false_negatives'])
        }
        
        return analysis
    
    def plot_prediction_analysis(self, analysis: Dict[str, Any], 
                               model_name: str = "Model", save_path: Optional[str] = None):
        """
        Plot prediction analysis.
        
        Args:
            analysis: Prediction analysis results
            model_name: Name of the model
            save_path: Path to save the plot
        """
        counts = analysis['counts']
        
        # Create pie chart
        plt.figure(figsize=(10, 8))
        
        labels = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives']
        sizes = [counts['true_positives'], counts['true_negatives'], 
                counts['false_positives'], counts['false_negatives']]
        colors = ['green', 'blue', 'red', 'orange']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title(f'Prediction Analysis - {model_name}')
        
        if save_path or self.save_plots:
            path = save_path or self.plot_dir / f"{model_name.lower()}_prediction_analysis.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction analysis saved to {path}")
        
        plt.show()
    
    def cross_validate(self, model, X: List[str], y: List[int], 
                      cv: int = 5, scoring: str = 'accuracy') -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            model: Model to evaluate
            X: Input features
            y: Target labels
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }
    
    def plot_learning_curves(self, train_sizes: List[int], train_scores: List[float], 
                           val_scores: List[float], model_name: str = "Model",
                           save_path: Optional[str] = None):
        """
        Plot learning curves.
        
        Args:
            train_sizes: Training set sizes
            train_scores: Training scores
            val_scores: Validation scores
            model_name: Name of the model
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_sizes, train_scores, 'o-', label='Training Score')
        plt.plot(train_sizes, val_scores, 'o-', label='Validation Score')
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title(f'Learning Curves - {model_name}')
        plt.legend()
        plt.grid(True)
        
        if save_path or self.save_plots:
            path = save_path or self.plot_dir / f"{model_name.lower()}_learning_curves.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            logger.info(f"Learning curves saved to {path}")
        
        plt.show()
    
    def create_dashboard(self, results: Dict[str, Any], 
                        save_path: str = "evaluation_dashboard.html"):
        """
        Create interactive evaluation dashboard.
        
        Args:
            results: Evaluation results
            save_path: Path to save the dashboard
        """
        # This would create an interactive HTML dashboard
        # For now, we'll create a simple text summary
        
        dashboard_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ACE Evaluation Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .model {{ border: 1px solid #ddd; padding: 20px; margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; }}
            </style>
        </head>
        <body>
            <h1>ACE Framework Evaluation Dashboard</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        for model_name, metrics in results.items():
            dashboard_content += f"""
            <div class="model">
                <h2>{model_name.upper()} Model</h2>
                <div class="metric">Accuracy: {metrics.get('accuracy', 0):.4f}</div>
                <div class="metric">Precision: {metrics.get('precision', 0):.4f}</div>
                <div class="metric">Recall: {metrics.get('recall', 0):.4f}</div>
                <div class="metric">F1-Score: {metrics.get('f1_score', 0):.4f}</div>
            </div>
            """
        
        dashboard_content += """
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(dashboard_content)
        
        logger.info(f"Evaluation dashboard saved to {save_path}")
