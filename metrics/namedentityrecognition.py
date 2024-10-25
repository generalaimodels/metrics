import torch
from typing import List, Dict, Tuple

class NERMetrics:
    def __init__(self, num_entities: int, entity_labels: List[str]):
        """
        Initializes the NER Metrics object.

        Args:
            num_entities (int): Number of entity types.
            entity_labels (List[str]): List of entity labels (e.g., ['Person', 'Location']).


        Example:
        Mock entity labels for the demonstration
        entity_labels = ['O', 'Person', 'Location', 'Organization']
        
        Initialize the NER metrics with 4 entity types ("O" (no entity), "Person", "Location", "Organization").
        ner_metrics = NERMetrics(num_entities=4, entity_labels=entity_labels)
        
        Sample predictions (batch_size=2, seq_len=4) and true labels in tensor form
        For example, "Person" is indexed as 1, "Location" as 2, etc.
        predictions = torch.tensor([[1, 2, 0, 0],  # First example's predictions
                                    [1, 0, 2, 1]]) # Second example's predictions
        
        Ground truth labels, where "O" is 0, "Person" is 1, etc.
        true_labels = torch.tensor([[1, 2, 0, 0],  # First example's ground truth
                                    [0, 0, 2, 1]]) # Second example's ground truth
        
        Update metrics based on the current batch of predictions/labels
        ner_metrics.update(predictions, true_labels)
        
        Compute and print final metrics
        metrics = ner_metrics.compute_metrics()
        print(metrics)
        """
        self.num_entities = num_entities
        self.entity_labels = entity_labels
        self.reset()

    def reset(self):
        """
        Resets True Positives, False Positives, and False Negatives for all entities.
        """
        self.tp = torch.zeros(self.num_entities, dtype=torch.long)
        self.fp = torch.zeros(self.num_entities, dtype=torch.long)
        self.fn = torch.zeros(self.num_entities, dtype=torch.long)
    
    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Updates counts of TP, FP, FN based on batch predictions and actual labels.

        Args:
            preds (torch.Tensor): Model predictions tensor of shape (batch_size, seq_len).
            labels (torch.Tensor): Ground truth labels tensor of shape (batch_size, seq_len).
        """
        for entity_idx in range(self.num_entities):
            # Compute True Positives (TP)
            true_entity_mask = (labels == entity_idx)
            pred_entity_mask = (preds == entity_idx)
            
            self.tp[entity_idx] += torch.sum(true_entity_mask & pred_entity_mask).item()
            self.fp[entity_idx] += torch.sum(~true_entity_mask & pred_entity_mask).item()
            self.fn[entity_idx] += torch.sum(true_entity_mask & ~pred_entity_mask).item()

    def precision(self, entity_idx: int) -> float:
        """Calculates precision for a given entity."""
        tp, fp = self.tp[entity_idx], self.fp[entity_idx]
        return tp / (tp + fp) if tp + fp > 0 else 0.0

    def recall(self, entity_idx: int) -> float:
        """Calculates recall for a given entity."""
        tp, fn = self.tp[entity_idx], self.fn[entity_idx]
        return tp / (tp + fn) if tp + fn > 0 else 0.0

    def f1_score(self, entity_idx: int) -> float:
        """Calculates F1 score for a given entity."""
        precision = self.precision(entity_idx)
        recall = self.recall(entity_idx)
        return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Computes precision, recall, and F1 score for each entity type.

        Returns:
            Dict: Dictionary with entity labels as keys and corresponding precision, recall, and F1 score.
        """
        entity_metrics = {}
        for idx, label in enumerate(self.entity_labels):
            precision = self.precision(idx)
            recall = self.recall(idx)
            f1 = self.f1_score(idx)
            
            entity_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        return entity_metrics


