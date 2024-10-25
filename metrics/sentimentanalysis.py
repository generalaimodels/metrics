from typing import Optional, Union, Tuple
import torch
from torch import Tensor
import warnings


class SentimentMetricsPipeline:
    """A robust sentiment analysis metrics pipeline for calculating accuracy and scores.
       Initialize pipeline
        pipeline = SentimentMetricsPipeline(num_classes=3)

        Sample data
        predictions = torch.tensor([0, 1, 2, 1, 0])
        targets = torch.tensor([0, 1, 2, 2, 1])
        weights = torch.tensor([1.0, 2.0, 1.5])

        Update metrics
        pipeline.update(predictions, targets)

        Compute metrics
        accuracy = pipeline.compute_accuracy()
        accuracy_per_class = pipeline.compute_accuracy(per_class=True)
        weighted_score = pipeline.compute_weighted_score(weights)

        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Per-class Accuracy: {accuracy_per_class[1]}")
        print(f"Weighted Score: {weighted_score:.4f}")
    
    
    """ 

    def __init__(
        self,
        num_classes: int,
        device: Optional[str] = None,
        epsilon: float = 1e-7
    ) -> None:
        """
        Initialize the sentiment metrics pipeline.

        Args:
            num_classes: Number of sentiment classes
            device: Computing device ('cpu' or 'cuda')
            epsilon: Small constant to prevent division by zero
        """
        self.num_classes = num_classes
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.epsilon = epsilon
        self._reset_states()

    def _reset_states(self) -> None:
        """Reset internal state variables."""
        self.total_correct = 0
        self.total_samples = 0
        self.class_correct = torch.zeros(self.num_classes).to(self.device)
        self.class_total = torch.zeros(self.num_classes).to(self.device)

    def _validate_inputs(
        self,
        predictions: Tensor,
        targets: Tensor,
        weights: Optional[Tensor] = None
    ) -> None:
        """
        Validate input tensors for shape and type consistency.

        Args:
            predictions: Model predictions
            targets: Ground truth labels
            weights: Optional weights for weighted metrics
        
        Raises:
            ValueError: If inputs are invalid
        """
        if predictions.dim() not in (1, 2):
            raise ValueError("Predictions must be 1D or 2D tensor")
        if predictions.dim() == 2 and predictions.size(1) != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} classes, got {predictions.size(1)}"
            )
        if predictions.size(0) != targets.size(0):
            raise ValueError("Predictions and targets must have same batch size")
        if weights is not None and weights.size(0) != self.num_classes:
            raise ValueError(
                f"Weights must have size {self.num_classes}"
            )

    def update(
        self,
        predictions: Tensor,
        targets: Tensor,
        weights: Optional[Tensor] = None
    ) -> None:
        """
        Update metrics with new batch of predictions and targets.

        Args:
            predictions: Model predictions (logits or class indices)
            targets: Ground truth labels
            weights: Optional class weights
        """
        try:
            predictions = predictions.to(self.device)
            targets = targets.to(self.device)
            if weights is not None:
                weights = weights.to(self.device)

            self._validate_inputs(predictions, targets, weights)

            # Convert logits to predictions if necessary
            if predictions.dim() == 2:
                predictions = torch.argmax(predictions, dim=1)

            # Update accuracy metrics
            correct = (predictions == targets)
            self.total_correct += correct.sum().item()
            self.total_samples += targets.size(0)

            # Update per-class metrics
            for i in range(self.num_classes):
                mask = (targets == i)
                self.class_correct[i] += correct[mask].sum().item()
                self.class_total[i] += mask.sum().item()

        except Exception as e:
            warnings.warn(f"Error in update: {str(e)}")
            raise

    def compute_accuracy(
        self,
        per_class: bool = False
    ) -> Union[float, Tuple[float, Tensor]]:
        """
        Compute sentiment classification accuracy.

        Args:
            per_class: Whether to return per-class accuracies

        Returns:
            Overall accuracy and optionally per-class accuracies
        """
        try:
            overall_acc = (
                self.total_correct / (self.total_samples + self.epsilon)
            )
            
            if not per_class:
                return overall_acc

            class_acc = torch.zeros_like(self.class_total)
            mask = self.class_total > 0
            class_acc[mask] = self.class_correct[mask] / self.class_total[mask]
            
            return overall_acc, class_acc

        except Exception as e:
            warnings.warn(f"Error computing accuracy: {str(e)}")
            raise

    def compute_weighted_score(
        self,
        weights: Optional[Tensor] = None
    ) -> float:
        """
        Compute weighted average sentiment score.

        Args:
            weights: Optional class weights

        Returns:
            Weighted average sentiment score
        """
        try:
            if weights is None:
                weights = torch.ones(self.num_classes).to(self.device)
            else:
                weights = weights.to(self.device)

            class_acc = self.class_correct / (self.class_total + self.epsilon)
            weighted_score = torch.sum(class_acc * weights) / torch.sum(weights)
            
            return weighted_score.item()

        except Exception as e:
            warnings.warn(f"Error computing weighted score: {str(e)}")
            raise


