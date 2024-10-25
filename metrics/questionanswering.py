import torch
from typing import List, Tuple

class RobustnessMetricsQA:
    """
    A robust metrics computation pipeline for question-answering tasks using PyTorch.
    Calculates various metrics like Exact Match, F1 Score, Mean Reciprocal Rank, and MAP.

    Example:
      
def test_metrics_pipeline():
    Use the class defined previously
    metrics_pipeline = RobustnessMetricsQA()

    Sample predictions and ground truth data
    predictions = [
        "Apple is a fruit.",
        "Python is a language.",
        "The sky is blue."
    ]
    
    ground_truths = [
        "Apple is a fruit.",
        "Python is a programming language.",
        "The sky looks blue."
    ]
    
    Ranked predictions (each sublist represents the ranked answers for a given query)
    ranked_predictions = [
        ["Apple is a fruit.", "Apple fruit", "Orange is a fruit."],
        ["Python is a language.", "Python programming", "Python snake"],
        ["The sky is blue.", "The sky looks blue.", "The sea is blue."]
    ]
    
    Run the metrics evaluation
    metrics = metrics_pipeline.evaluate(predictions, ground_truths, ranked_predictions)
    
    Print the results
    print("Test Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    Run the test function
    test_metrics_pipeline()

    """

    def __init__(self):
        pass

    @staticmethod
    def exact_match_score(predictions: List[str], ground_truths: List[str]) -> float:
        """
        Calculate the Exact Match (EM) score.
        
        Args:
            predictions (List[str]): List of predicted answers (strings).
            ground_truths (List[str]): List of corresponding ground truth answers (strings).

        Returns:
            float: Exact match score (fraction of matches).
        """
        total = len(predictions)
        correct = sum([1 for pred, truth in zip(predictions, ground_truths) if pred == truth])
        return correct / total if total > 0 else 0.0

    @staticmethod
    def f1_score(prediction: str, ground_truth: str) -> float:
        """
        Compute the F1 score for a single prediction and ground truth.

        Args:
            prediction (str): Predicted answer.
            ground_truth (str): Ground truth answer.

        Returns:
            float: F1 score for token overlap.
        """
        pred_tokens = prediction.lower().split()
        truth_tokens = ground_truth.lower().split()

        common_tokens = set(pred_tokens) & set(truth_tokens)
        num_common = len(common_tokens)

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(truth_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def mean_f1_score(predictions: List[str], ground_truths: List[str]) -> float:
        """
        Calculate average F1 score across all predictions.

        Args:
            predictions (List[str]): List of predicted answers.
            ground_truths (List[str]): List of ground truth answers.

        Returns:
            float: Mean F1 score.
        """
        f1_scores = [RobustnessMetricsQA.f1_score(pred, truth) for pred, truth in zip(predictions, ground_truths)]
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    def reciprocal_rank(self, ranked_predictions: List[str], ground_truth: str) -> float:
        """
        Compute Reciprocal Rank (RR) for a single prediction and ground truth.

        Args:
            ranked_predictions (List[str]): Ranked list of predicted answers.
            ground_truth (str): The correct answer.

        Returns:
            float: Reciprocal rank (1 / rank of first correct prediction), or 0 if not present.
        """
        try:
            rank = ranked_predictions.index(ground_truth) + 1
            return 1.0 / rank
        except ValueError:
            return 0.0

    def mean_reciprocal_rank(self, batch_ranked_predictions: List[List[str]], ground_truths: List[str]) -> float:
        """
        Compute Mean Reciprocal Rank (MRR) across all predictions.

        Args:
            batch_ranked_predictions (List[List[str]]): Batch of ranked lists of predictions.
            ground_truths (List[str]): Corresponding list of ground truth answers.

        Returns:
            float: Mean Reciprocal Rank.
        """
        rr_scores = [self.reciprocal_rank(preds, truth) for preds, truth in zip(batch_ranked_predictions, ground_truths)]
        return torch.mean(torch.tensor(rr_scores)).item() if rr_scores else 0.0

    def average_precision(self, ranked_predictions: List[str], ground_truth: str) -> float:
        """
        Compute Average Precision (AP) for a ranked list of predictions.

        Args:
            ranked_predictions (List[str]): Ranked list of predicted answers.
            ground_truth (str): The correct answer.

        Returns:
            float: Average precision.
        """
        correct_indices = [i for i, pred in enumerate(ranked_predictions) if pred == ground_truth]

        if not correct_indices:
            return 0.0

        ap_sum = 0.0
        for i, idx in enumerate(correct_indices, 1):
            ap_sum += i / (idx + 1)
        
        return ap_sum / len(correct_indices)

    def mean_average_precision(self, batch_ranked_predictions: List[List[str]], ground_truths: List[str]) -> float:
        """
        Compute Mean Average Precision (MAP) across all predictions.

        Args:
            batch_ranked_predictions (List[List[str]]): Batch of ranked lists of predictions.
            ground_truths (List[str]): Corresponding list of ground truth answers.

        Returns:
            float: Mean Average Precision (MAP).
        """
        ap_scores = [self.average_precision(preds, truth) for preds, truth in zip(batch_ranked_predictions, ground_truths)]
        return torch.mean(torch.tensor(ap_scores)).item() if ap_scores else 0.0

    def evaluate(self, predictions: List[str], ground_truths: List[str], ranked_predictions: List[List[str]]) -> dict:
        """
        Evaluate all metrics and return as a dictionary.
        
        Args:
            predictions (List[str]): List of unranked predictions.
            ground_truths (List[str]): List of ground truths.
            ranked_predictions (List[List[str]]): List of ranked predictions per question.
            
        Returns:
            dict: A dictionary containing all the metrics.
        """
        metrics = {
            'Exact Match': self.exact_match_score(predictions, ground_truths),
            'Mean F1 Score': self.mean_f1_score(predictions, ground_truths),
            'Mean Reciprocal Rank': self.mean_reciprocal_rank(ranked_predictions, ground_truths),
            'Mean Average Precision': self.mean_average_precision(ranked_predictions, ground_truths)
        }        
        return metrics
