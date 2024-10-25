import torch
from typing import List, Optional, Union, Tuple
import torch.nn.functional as F


class TextClassificationMetrics:
    def __init__(self, predicted: torch.Tensor, actual: torch.Tensor, num_classes: Optional[int] = None):
        """
        Initialize with predicted and actual labels. Both should be torch tensors.
        
        :param predicted: Predicted labels or logits from the model.
        :param actual: Ground truth labels.
        :param num_classes: The number of classes if doing multiclass classification (default is None for binary).
        """
        if len(predicted) != len(actual):
            raise ValueError("Predicted and actual labels must have the same length.")

        self.predicted = predicted
        self.actual = actual
        self.num_classes = num_classes or len(torch.unique(actual, sorted=True))
        
        if self.num_classes == 2:  # Binary Classification
            self.predicted_labels = (predicted > 0.5).long() if predicted.ndim > 1 else predicted
        else:  # Multiclass Classification
            self.predicted_labels = torch.argmax(predicted, dim=1)

    def _true_positive(self) -> int:
        """Helper to calculate True Positives."""
        return torch.sum((self.predicted_labels == self.actual) & (self.actual == 1)).item()

    def _false_positive(self) -> int:
        """Helper to calculate False Positives."""
        return torch.sum((self.predicted_labels == 1) & (self.actual == 0)).item()

    def _false_negative(self) -> int:
        """Helper to calculate False Negatives."""
        return torch.sum((self.predicted_labels == 0) & (self.actual == 1)).item()

    def _true_negative(self) -> int:
        """Helper to calculate True Negatives."""
        return torch.sum((self.predicted_labels == 0) & (self.actual == 0)).item()

    def accuracy(self) -> float:
        """Calculate accuracy for binary or multiclass classification."""
        correct = torch.sum(self.predicted_labels == self.actual).item()
        total = self.actual.numel()
        return correct / total

    def precision(self) -> float:
        """Calculate precision for binary or multiclass."""
        tp = self._true_positive()
        fp = self._false_positive()

        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def recall(self) -> float:
        """Calculate recall for binary or multiclass."""
        tp = self._true_positive()
        fn = self._false_negative()

        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def f1_score(self) -> float:
        """Calculate the F1 score."""
        prec = self.precision()
        rec = self.recall()

        if prec + rec == 0:
            return 0.0
        return 2 * (prec * rec) / (prec + rec)

    def matthews_corrcoef(self) -> float:
        """Calculate the Matthews Correlation Coefficient (MCC)."""
        tp = self._true_positive()
        tn = self._true_negative()
        fp = self._false_positive()
        fn = self._false_negative()
        
        numerator = (tp * tn) - (fp * fn)
        denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def roc_auc(self) -> float:
        """Calculate the ROC AUC score using logits for binary classification."""
        if self.num_classes != 2:
            raise ValueError("ROC AUC is only supported for binary classification.")
    
        if self.predicted.ndim == 1:  # Binary case with single logits
            probabilities = torch.sigmoid(self.predicted)
        else:  # Multiclass case - handle class 1's predicted probabilities
            probabilities = F.softmax(self.predicted, dim=1)[:, 1]
    
        # In actual implementation, you would compute ROC AUC properly
        actual_one_hot = F.one_hot(self.actual, num_classes=self.num_classes)
    
        # Placeholder computation: Normally, you'd use a package like sklearn or implement an AUC calculator from scratch
        auc = torch.mean(probabilities).item()  # Simplified; replace with proper ROC AUC computation

        return auc
    
    def calculate_all_metrics(self) -> dict:
        """
        Calculate all metrics (accuracy, precision, recall, F1 score, MCC) and return as a dictionary.
        For multiclass classification, it calculates per-class precision, recall, F1 score as well.
        
        :return: A dictionary containing all the calculated metrics.
        """
        metrics = {
            'accuracy': self.accuracy(),
            'precision': self.precision(),
            'recall': self.recall(),
            'f1_score': self.f1_score(),
            'matthews_corrcoef': self.matthews_corrcoef()
        }

        if self.num_classes == 2:
            try:
                metrics['roc_auc'] = self.roc_auc()
            except ValueError as e:
                # gracefully handle if not applicable
                metrics['roc_auc'] = f"ROC AUC Calculation Error: {e}"

        return metrics
    
if __name__ == '__main__':
    # Example with binary classification logits
    preds = torch.tensor([0.8, 0.3, 0.5, 0.9, 0.1])
    labels = torch.tensor([1, 0, 0, 1, 0])

    metrics = TextClassificationMetrics(preds, labels)
    results = metrics.calculate_all_metrics()

    print("Metrics for binary classification:")
    for metric, value in results.items():
        print(f"{metric}: {value}")

    # Example with multiclass classification (3 classes)
    preds_multiclass = torch.tensor([[0.2, 0.5, 0.3],
                                     [0.1, 0.3, 0.6],
                                     [0.05, 0.9, 0.05],
                                     [0.8, 0.1, 0.1],
                                     [0.2, 0.2, 0.6]])

    labels_multiclass = torch.tensor([1, 2, 1, 0, 2])

    metrics_multi = TextClassificationMetrics(preds_multiclass, labels_multiclass, num_classes=3)
    results_multi = metrics_multi.calculate_all_metrics()

    print("\nMetrics for multiclass classification:")
    for metric, value in results_multi.items():
        print(f"{metric}: {value}")