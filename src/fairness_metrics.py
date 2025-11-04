"""Fairness metrics for healthcare AI models.

Implements various fairness evaluation metrics including demographic parity,
equalized odds, and disparate impact across demographic groups.
"""

import numpy as np
from typing import Dict, List, Tuple
import pandas as pd


class FairnessMetrics:
    """Compute fairness metrics for model predictions."""
    
    def __init__(self, sensitive_attributes: List[str]):
        """
        Initialize fairness metrics calculator.
        
        Args:
            sensitive_attributes: List of sensitive attribute names (e.g., ['age', 'gender', 'race'])
        """
        self.sensitive_attributes = sensitive_attributes
    
    def demographic_parity(self, y_pred: np.ndarray, sensitive_attr: np.ndarray) -> float:
        """
        Calculate demographic parity difference.
        
        Args:
            y_pred: Model predictions
            sensitive_attr: Sensitive attribute values
            
        Returns:
            Demographic parity difference score
        """
        groups = np.unique(sensitive_attr)
        positive_rates = []
        
        for group in groups:
            mask = sensitive_attr == group
            positive_rate = np.mean(y_pred[mask] > 0.5)
            positive_rates.append(positive_rate)
        
        return max(positive_rates) - min(positive_rates)
    
    def equalized_odds(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       sensitive_attr: np.ndarray) -> Dict[str, float]:
        """
        Calculate equalized odds (TPR and FPR differences).
        
        Args:
            y_true: True labels
            y_pred: Model predictions
            sensitive_attr: Sensitive attribute values
            
        Returns:
            Dictionary with TPR and FPR difference scores
        """
        groups = np.unique(sensitive_attr)
        tpr_list, fpr_list = [], []
        
        for group in groups:
            mask = sensitive_attr == group
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            # Calculate TPR and FPR
            tp = np.sum((y_pred_group > 0.5) & (y_true_group == 1))
            fn = np.sum((y_pred_group <= 0.5) & (y_true_group == 1))
            fp = np.sum((y_pred_group > 0.5) & (y_true_group == 0))
            tn = np.sum((y_pred_group <= 0.5) & (y_true_group == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        return {
            'tpr_difference': max(tpr_list) - min(tpr_list),
            'fpr_difference': max(fpr_list) - min(fpr_list)
        }
    
    def disparate_impact(self, y_pred: np.ndarray, sensitive_attr: np.ndarray) -> float:
        """
        Calculate disparate impact ratio.
        
        Args:
            y_pred: Model predictions
            sensitive_attr: Sensitive attribute values
            
        Returns:
            Disparate impact ratio (should be close to 1.0 for fairness)
        """
        groups = np.unique(sensitive_attr)
        
        if len(groups) < 2:
            return 1.0
        
        positive_rates = []
        for group in groups:
            mask = sensitive_attr == group
            positive_rate = np.mean(y_pred[mask] > 0.5)
            positive_rates.append(positive_rate)
        
        return min(positive_rates) / max(positive_rates) if max(positive_rates) > 0 else 0
    
    def compute_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                           sensitive_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Compute all fairness metrics for all sensitive attributes.
        
        Args:
            y_true: True labels
            y_pred: Model predictions
            sensitive_data: DataFrame containing sensitive attributes
            
        Returns:
            Dictionary of metrics for each sensitive attribute
        """
        results = {}
        
        for attr in self.sensitive_attributes:
            if attr not in sensitive_data.columns:
                continue
            
            sensitive_attr = sensitive_data[attr].values
            
            results[attr] = {
                'demographic_parity': self.demographic_parity(y_pred, sensitive_attr),
                'disparate_impact': self.disparate_impact(y_pred, sensitive_attr),
                **self.equalized_odds(y_true, y_pred, sensitive_attr)
            }
        
        return results
