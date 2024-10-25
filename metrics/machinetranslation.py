from typing import List, Dict, Union, Optional, Tuple
import torch
from collections import Counter
import math
from torch import Tensor


class TranslationMetrics:
    """A comprehensive translation metrics calculator."""
    
    def __init__(
        self,
        max_n_grams: int = 4,
        granularity: str = "word",
        device: Optional[str] = None
    ) -> None:
        """
        Initialize translation metrics calculator.
        
        Args:
            max_n_grams: Maximum n-grams to consider
            granularity: Level of analysis ('word', 'char', 'sentence')
            device: Computation device ('cpu' or 'cuda')
        """
        self.max_n_grams = max_n_grams
        self.granularity = granularity
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._validate_init_params()

    def _validate_init_params(self) -> None:
        """Validate initialization parameters."""
        if self.max_n_grams < 1:
            raise ValueError("max_n_grams must be positive")
        if self.granularity not in ['word', 'char', 'sentence']:
            raise ValueError("granularity must be 'word', 'char', or 'sentence'")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text based on granularity."""
        if self.granularity == 'word':
            return text.strip().split()
        elif self.granularity == 'char':
            return list(text.strip())
        return [text.strip()]

    def _get_ngrams(
        self,
        tokens: List[str],
        n: int
    ) -> Counter:
        """
        Generate n-grams from tokens.
        
        Args:
            tokens: List of tokens
            n: n-gram size
            
        Returns:
            Counter of n-grams
        """
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] += 1
        return ngrams

    def calculate_bleu(
        self,
        hypothesis: str,
        reference: str,
        smooth: bool = True
    ) -> float:
        """
        Calculate BLEU score.
        
        Args:
            hypothesis: Predicted translation
            reference: Ground truth translation
            smooth: Whether to apply smoothing
            
        Returns:
            BLEU score
        """
        hyp_tokens = self._tokenize(hypothesis)
        ref_tokens = self._tokenize(reference)
        
        # Calculate n-gram precisions
        precisions = []
        for n in range(1, self.max_n_grams + 1):
            hyp_ngrams = self._get_ngrams(hyp_tokens, n)
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            
            matches = sum((hyp_ngrams & ref_ngrams).values())
            total = sum(hyp_ngrams.values())
            
            if smooth:
                precision = (matches + 1) / (total + 1)
            else:
                precision = matches / total if total > 0 else 0
            precisions.append(precision)

        # Calculate brevity penalty
        bp = math.exp(min(0, 1 - len(ref_tokens) / len(hyp_tokens)))
        
        # Calculate final BLEU score
        score = bp * math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        return score

    def calculate_ter(
        self,
        hypothesis: str,
        reference: str
    ) -> float:
        """
        Calculate Translation Error Rate (TER).
        
        Args:
            hypothesis: Predicted translation
            reference: Ground truth translation
            
        Returns:
            TER score
        """
        hyp_tokens = self._tokenize(hypothesis)
        ref_tokens = self._tokenize(reference)
        
        m, n = len(hyp_tokens), len(ref_tokens)
        dp = torch.zeros((m + 1, n + 1), device=self.device)
        
        # Initialize first row and column
        for i in range(m + 1):
            dp[i, 0] = i
        for j in range(n + 1):
            dp[0, j] = j
            
        # Fill dynamic programming table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if hyp_tokens[i-1] == ref_tokens[j-1]:
                    dp[i, j] = dp[i-1, j-1]
                else:
                    dp[i, j] = 1 + min(
                        dp[i-1, j],    # deletion
                        dp[i, j-1],    # insertion
                        dp[i-1, j-1]   # substitution
                    )
        
        # Calculate TER score
        return float(dp[m, n] / max(m, n))

    def calculate_rouge(
        self,
        hypothesis: str,
        reference: str,
        rouge_type: str = 'L'
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        
        Args:
            hypothesis: Predicted translation
            reference: Ground truth translation
            rouge_type: Type of ROUGE score ('N' or 'L')
            
        Returns:
            Dictionary containing precision, recall, and F1 scores
        """
        hyp_tokens = self._tokenize(hypothesis)
        ref_tokens = self._tokenize(reference)
        
        if rouge_type == 'L':
            return self._calculate_rouge_l(hyp_tokens, ref_tokens)
        else:
            return self._calculate_rouge_n(hyp_tokens, ref_tokens)

    def _calculate_rouge_l(
        self,
        hyp_tokens: List[str],
        ref_tokens: List[str]
    ) -> Dict[str, float]:
        """Calculate ROUGE-L using LCS."""
        m, n = len(hyp_tokens), len(ref_tokens)
        lcs_table = torch.zeros((m + 1, n + 1), device=self.device)
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if hyp_tokens[i-1] == ref_tokens[j-1]:
                    lcs_table[i, j] = lcs_table[i-1, j-1] + 1
                else:
                    lcs_table[i, j] = max(lcs_table[i-1, j], lcs_table[i, j-1])
        
        lcs_length = lcs_table[m, n]
        
        precision = lcs_length / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
        recall = lcs_length / len(ref_tokens) if len(ref_tokens) > 0 else 0
        f1 = self._calculate_f1(precision, recall)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }

    @staticmethod
    def _calculate_f1(precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    

# import unittest
# from typing import Dict
# import torch


# def test_translation_metrics() -> None:
#     """Test the TranslationMetrics class with various scenarios."""
    
#     # Initialize metrics calculator
#     metrics = TranslationMetrics(max_n_grams=4, granularity='word')
    
#     # Test cases
#     test_cases = [
#         {
#             'name': 'Perfect Match',
#             'hypothesis': 'The cat sits on the mat.',
#             'reference': 'The cat sits on the mat.',
#             'expected_bleu': 1.0,
#             'expected_ter': 0.0,
#         },
#         {
#             'name': 'Partial Match',
#             'hypothesis': 'The cat sits mat.',
#             'reference': 'The cat sits on the mat.',
#             'expected_bleu': lambda x: 0 < x < 1,
#             'expected_ter': lambda x: 0 < x < 1,
#         },
#         {
#             'name': 'No Match',
#             'hypothesis': 'The dog runs in the park.',
#             'reference': 'The cat sits on the mat.',
#             'expected_bleu': lambda x: x < 0.3,
#             'expected_ter': lambda x: x > 0.7,
#         },
#         {
#             'name': 'Empty Strings',
#             'hypothesis': '',
#             'reference': '',
#             'expected_bleu': 1.0,
#             'expected_ter': 0.0,
#         },
#     ]
    
#     # Run tests
#     for test_case in test_cases:
#         print(f"\nTesting: {test_case['name']}")
        
#         # Calculate metrics
#         bleu_score = metrics.calculate_bleu(
#             test_case['hypothesis'],
#             test_case['reference']
#         )
#         ter_score = metrics.calculate_ter(
#             test_case['hypothesis'],
#             test_case['reference']
#         )
#         rouge_scores = metrics.calculate_rouge(
#             test_case['hypothesis'],
#             test_case['reference']
#         )
        
#         # Validate results
#         print(f"BLEU Score: {bleu_score:.4f}")
#         print(f"TER Score: {ter_score:.4f}")
#         print("ROUGE Scores:", {k: f"{v:.4f}" for k, v in rouge_scores.items()})
        
#         # Check if results meet expectations
#         expected_bleu = test_case['expected_bleu']
#         expected_ter = test_case['expected_ter']
        
#         if callable(expected_bleu):
#             assert expected_bleu(bleu_score), f"BLEU score {bleu_score} doesn't meet expectation"
#         else:
#             assert abs(bleu_score - expected_bleu) < 1e-5, \
#                 f"BLEU score {bleu_score} doesn't match expected {expected_bleu}"
        
#         if callable(expected_ter):
#             assert expected_ter(ter_score), f"TER score {ter_score} doesn't meet expectation"
#         else:
#             assert abs(ter_score - expected_ter) < 1e-5, \
#                 f"TER score {ter_score} doesn't match expected {expected_ter}"


# def test_different_granularities() -> None:
#     """Test metrics with different granularities."""
    
#     hypothesis = "The cat sits."
#     reference = "The cat sits."
    
#     for granularity in ['word', 'char', 'sentence']:
#         print(f"\nTesting granularity: {granularity}")
#         metrics = TranslationMetrics(max_n_grams=4, granularity=granularity)
        
#         bleu_score = metrics.calculate_bleu(hypothesis, reference)
#         ter_score = metrics.calculate_ter(hypothesis, reference)
#         rouge_scores = metrics.calculate_rouge(hypothesis, reference)
        
#         print(f"BLEU Score: {bleu_score:.4f}")
#         print(f"TER Score: {ter_score:.4f}")
#         print("ROUGE Scores:", {k: f"{v:.4f}" for k, v in rouge_scores.items()})


# def test_edge_cases() -> None:
#     """Test edge cases and error handling."""
    
#     metrics = TranslationMetrics()
    
#     # Test cases for edge scenarios
#     edge_cases = [
#         ("", "reference"),
#         ("hypothesis", ""),
#         ("   ", "   "),
#         ("a" * 1000, "b" * 1000),  # Very long strings
#         ("!@#$%^", "!@#$%^"),      # Special characters
#         ("123 456", "123 456"),     # Numbers
#     ]
    
#     for hypothesis, reference in edge_cases:
#         try:
#             print(f"\nTesting edge case: '{hypothesis[:20]}' vs '{reference[:20]}'")
            
#             bleu_score = metrics.calculate_bleu(hypothesis, reference)
#             ter_score = metrics.calculate_ter(hypothesis, reference)
#             rouge_scores = metrics.calculate_rouge(hypothesis, reference)
            
#             print(f"BLEU Score: {bleu_score:.4f}")
#             print(f"TER Score: {ter_score:.4f}")
#             print("ROUGE Scores:", {k: f"{v:.4f}" for k, v in rouge_scores.items()})
            
#         except Exception as e:
#             print(f"Error occurred: {str(e)}")


# def main() -> None:
#     """Run all tests."""
#     print("Running translation metrics tests...\n")
    
#     try:
#         print("=== Testing basic scenarios ===")
#         test_translation_metrics()
        
#         print("\n=== Testing different granularities ===")
#         test_different_granularities()
        
#         print("\n=== Testing edge cases ===")
#         test_edge_cases()
        
#         print("\nAll tests completed successfully!")
        
#     except AssertionError as e:
#         print(f"Test failed: {str(e)}")
#     except Exception as e:
#         print(f"Unexpected error: {str(e)}")


# if __name__ == "__main__":
#     main()