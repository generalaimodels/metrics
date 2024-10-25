from typing import List, Dict, Union, Optional, Tuple
import torch
from torch import Tensor
import math
from collections import Counter
import warnings
from dataclasses import dataclass

@dataclass
class MetricsConfig:
    """Configuration class for metrics calculation."""
    pad_token_id: int = 0
    eos_token_id: int = 2
    max_ngram_size: int = 4
    alpha: float = 0.5  # Parameter for content preservation score
    temperature: float = 1.0  # Temperature for softmax

class TextGenerationMetrics:
    """Pipeline for computing various text generation metrics."""
    
    def __init__(self, config: MetricsConfig):
        """Initialize the metrics pipeline with configuration."""
        self.config = config
        
    def _compute_ngrams(
        self, 
        sequence: List[int], 
        n: int
    ) -> Counter:
        """Compute n-grams from a sequence.
        
        Args:
            sequence: Input token sequence
            n: Size of n-grams
            
        Returns:
            Counter of n-grams
        """
        return Counter(tuple(sequence[i:i + n]) 
                      for i in range(len(sequence) - n + 1))

    def calculate_perplexity(
        self, 
        logits: Tensor, 
        target: Tensor, 
        mask: Optional[Tensor] = None
    ) -> float:
        """Calculate perplexity score.
        
        Args:
            logits: Model output logits of shape (batch_size, seq_len, vocab_size)
            target: Target indices of shape (batch_size, seq_len)
            mask: Optional mask for padding
            
        Returns:
            Perplexity score
        """
        try:
            batch_size, seq_len, vocab_size = logits.shape
            
            # Apply temperature scaling
            scaled_logits = logits / self.config.temperature
            
            # Calculate cross entropy loss
            log_probs = torch.log_softmax(scaled_logits, dim=-1)
            nll = -torch.gather(log_probs, -1, target.unsqueeze(-1)).squeeze(-1)
            
            if mask is not None:
                nll = nll * mask
                token_count = mask.sum()
            else:
                token_count = batch_size * seq_len
                
            avg_nll = nll.sum() / token_count
            return torch.exp(avg_nll).item()
            
        except Exception as e:
            warnings.warn(f"Error in perplexity calculation: {str(e)}")
            return float('inf')

    def calculate_bleu(
        self,
        hypotheses: List[List[int]],
        references: List[List[int]],
        max_order: int = 4
    ) -> float:
        """Calculate BLEU score.
        
        Args:
            hypotheses: Generated sequences
            references: Reference sequences
            max_order: Maximum n-gram order
            
        Returns:
            BLEU score
        """
        try:
            total_score = 0.0
            weights = [1/max_order] * max_order
            
            for hyp, ref in zip(hypotheses, references):
                score = 0.0
                for n in range(1, max_order + 1):
                    hyp_ngrams = self._compute_ngrams(hyp, n)
                    ref_ngrams = self._compute_ngrams(ref, n)
                    
                    matches = sum((hyp_ngrams & ref_ngrams).values())
                    total = sum(hyp_ngrams.values())
                    
                    if total > 0:
                        score += weights[n-1] * (matches / total)
                        
                total_score += score
                
            return total_score / len(hypotheses)
            
        except Exception as e:
            warnings.warn(f"Error in BLEU calculation: {str(e)}")
            return 0.0

    def calculate_rouge_l(
        self,
        hypothesis: List[int],
        reference: List[int]
    ) -> float:
        """Calculate ROUGE-L score using LCS.
        
        Args:
            hypothesis: Generated sequence
            reference: Reference sequence
            
        Returns:
            ROUGE-L score
        """
        try:
            len_hyp = len(hypothesis)
            len_ref = len(reference)
            
            # Create LCS matrix
            lcs_matrix = [[0] * (len_ref + 1) for _ in range(len_hyp + 1)]
            
            # Fill LCS matrix
            for i in range(1, len_hyp + 1):
                for j in range(1, len_ref + 1):
                    if hypothesis[i-1] == reference[j-1]:
                        lcs_matrix[i][j] = lcs_matrix[i-1][j-1] + 1
                    else:
                        lcs_matrix[i][j] = max(lcs_matrix[i-1][j], 
                                             lcs_matrix[i][j-1])
            
            lcs_length = lcs_matrix[len_hyp][len_ref]
            
            if len_hyp == 0 or len_ref == 0:
                return 0.0
                
            precision = lcs_length / len_hyp
            recall = lcs_length / len_ref
            
            if precision + recall == 0:
                return 0.0
                
            f1 = 2 * precision * recall / (precision + recall)
            return f1
            
        except Exception as e:
            warnings.warn(f"Error in ROUGE-L calculation: {str(e)}")
            return 0.0

    def calculate_distinct_n(
        self,
        sequences: List[List[int]],
        n: int
    ) -> float:
        """Calculate distinct-n score.
        
        Args:
            sequences: List of token sequences
            n: Size of n-grams
            
        Returns:
            Distinct-n score
        """
        try:
            all_ngrams = set()
            total_ngrams = 0
            
            for sequence in sequences:
                ngrams = self._compute_ngrams(sequence, n)
                all_ngrams.update(ngrams.keys())
                total_ngrams += sum(ngrams.values())
                
            if total_ngrams == 0:
                return 0.0
                
            return len(all_ngrams) / total_ngrams
            
        except Exception as e:
            warnings.warn(f"Error in distinct-n calculation: {str(e)}")
            return 0.0

    def calculate_content_preservation(
        self,
        generated_embeddings: Tensor,
        reference_embeddings: Tensor
    ) -> float:
        """Calculate content preservation score using embeddings.
        
        Args:
            generated_embeddings: Embeddings of generated text
            reference_embeddings: Embeddings of reference text
            
        Returns:
            Content preservation score
        """
        try:
            # Normalize embeddings
            gen_norm = torch.nn.functional.normalize(generated_embeddings, p=2, dim=-1)
            ref_norm = torch.nn.functional.normalize(reference_embeddings, p=2, dim=-1)
            
            # Calculate cosine similarity
            similarity = torch.sum(gen_norm * ref_norm, dim=-1)
            
            return similarity.mean().item()
            
        except Exception as e:
            warnings.warn(f"Error in content preservation calculation: {str(e)}")
            return 0.0

    def compute_all_metrics(
        self,
        generated_sequences: List[List[int]],
        reference_sequences: List[List[int]],
        logits: Optional[Tensor] = None,
        embeddings: Optional[Dict[str, Tensor]] = None
    ) -> Dict[str, float]:
        """Compute all available metrics.
        
        Args:
            generated_sequences: List of generated token sequences
            reference_sequences: List of reference token sequences
            logits: Optional logits from model output
            embeddings: Optional dictionary containing generated and reference embeddings
            
        Returns:
            Dictionary containing all computed metrics
        """
        try:
            metrics: Dict[str, float] = {}
            
            # Calculate perplexity if logits are provided
            if logits is not None and isinstance(logits, Tensor):
                target = torch.tensor(reference_sequences)
                metrics['perplexity'] = self.calculate_perplexity(
                    logits=logits,
                    target=target
                )
            
            # Calculate BLEU score
            metrics['bleu'] = self.calculate_bleu(
                hypotheses=generated_sequences,
                references=reference_sequences,
                max_order=self.config.max_ngram_size
            )
            
            # Calculate ROUGE-L scores
            rouge_scores = []
            for hyp, ref in zip(generated_sequences, reference_sequences):
                rouge_scores.append(self.calculate_rouge_l(hyp, ref))
            metrics['rouge_l'] = sum(rouge_scores) / len(rouge_scores)
            
            # Calculate Distinct-1 and Distinct-2 scores
            metrics['distinct_1'] = self.calculate_distinct_n(
                sequences=generated_sequences,
                n=1
            )
            metrics['distinct_2'] = self.calculate_distinct_n(
                sequences=generated_sequences,
                n=2
            )
            
            # Calculate content preservation if embeddings are provided
            if embeddings is not None:
                if 'generated' in embeddings and 'reference' in embeddings:
                    metrics['content_preservation'] = self.calculate_content_preservation(
                        generated_embeddings=embeddings['generated'],
                        reference_embeddings=embeddings['reference']
                    )
            
            return metrics
            
        except Exception as e:
            warnings.warn(f"Error in computing metrics: {str(e)}")
            return {}

class MetricEvaluator:
    """Wrapper class for batch processing and metric evaluation."""
    
    def __init__(self, config: MetricsConfig):
        """Initialize the evaluator with configuration."""
        self.metrics = TextGenerationMetrics(config)
        
    def evaluate_batch(
        self,
        batch_data: Dict[str, Union[List[List[int]], Tensor, Dict[str, Tensor]]]
    ) -> Dict[str, float]:
        """Evaluate metrics for a batch of data.
        
        Args:
            batch_data: Dictionary containing:
                - generated_sequences: List of generated token sequences
                - reference_sequences: List of reference token sequences
                - logits: Optional model output logits
                - embeddings: Optional dictionary of embeddings
                
        Returns:
            Dictionary of computed metrics
        """
        try:
            return self.metrics.compute_all_metrics(
                generated_sequences=batch_data['generated_sequences'],
                reference_sequences=batch_data['reference_sequences'],
                logits=batch_data.get('logits'),
                embeddings=batch_data.get('embeddings')
            )
            
        except Exception as e:
            warnings.warn(f"Error in batch evaluation: {str(e)}")
            return {}

def main():
    """Example usage of the metrics pipeline."""
    # Configuration
    config = MetricsConfig(
        pad_token_id=0,
        eos_token_id=2,
        max_ngram_size=4,
        alpha=0.5,
        temperature=1.0
    )
    
    # Initialize evaluator
    evaluator = MetricEvaluator(config)
    
    # Example batch data
    batch_data = {
        'generated_sequences': [[1, 2, 3], [4, 5, 6]],
        'reference_sequences': [[1, 2, 3], [4, 5, 6]],
        'logits': torch.randn(2, 3, 100),  # (batch_size, seq_len, vocab_size)
        'embeddings': {
            'generated': torch.randn(2, 768),  # (batch_size, embedding_dim)
            'reference': torch.randn(2, 768)
        }
    }
    
    # Evaluate metrics
    metrics = evaluator.evaluate_batch(batch_data)
    print("Computed Metrics:", metrics)

if __name__ == "__main__":
    main()