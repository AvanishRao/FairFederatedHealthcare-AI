# Results Directory

This directory contains experimental results, visualizations, metrics, and outputs for the Fair Federated Healthcare AI research.

## Purpose

The results folder stores:
- Experimental plots and figures
- Model performance metrics
- Fairness evaluation results
- Statistical analysis outputs
- Manuscript figures and supplementary materials
- Results summaries and reports

## Organization Structure

```
results/
├── exploratory_analysis/
│   ├── data_distributions.png
│   ├── demographic_breakdown.png
│   └── missing_data_patterns.png
├── baseline_models/
│   ├─═ performance_comparison.png
│   ├─═ roc_curves.png
│   ├─═ learning_curves.png
│   └─═ metrics_summary.csv
├── fairness_analysis/
│   ├─═ demographic_parity.png
│   ├─═ equalized_odds_heatmap.png
│   ├─═ disparate_impact_ratio.png
│   ├─═ fairness_metrics_summary.csv
│   └─═ fairness_report.md
├── multimodal_models/
│   ├─═ model_architecture_diagram.png
│   ├─═ attention_weights_visualization.png
│   ├─═ ablation_study_results.png
│   └─═ model_comparison.csv
├── federated_learning/
│   ├─═ convergence_curves.png
│   ├─═ communication_efficiency.png
│   ├─═ privacy_evaluation.png
│   └─═ federated_metrics.csv
├── fairness_aware_fl/
│   ├─═ fairness_performance_tradeoff.png
│   ├─═ fair_aggregation_results.png
│   ├─═ constraint_effectiveness.png
│   └─═ fair_fl_metrics.csv
├── final_evaluation/
│   ├─═ cross_institutional_validation.png
│   ├─═ final_performance_metrics.csv
│   ├─═ fairness_summary.csv
│   └─═ evaluation_report.md
├── manuscript_figures/
│   ├─═ Figure_1_study_design.png
│   ├─═ Figure_2_fairness_framework.png
│   ├─═ Figure_3_model_performance.png
│   ├─═ Figure_4_federated_learning.png
│   ├─═ Figure_5_fairness_results.png
│   └─═ Supplementary_Figures.md
├── metrics_summary.csv
├── experiment_log.md
└── results_summary_for_publication.md
```

## File Naming Conventions

### Images
- Use descriptive names: `{experiment}_{metric}_{date}.png`
- Examples:
  - `fairness_demographic_parity_2025_11_05.png`
  - `baseline_lstm_roc_curve_2025_11_05.png`
  - `federated_convergence_2025_11_05.png`

### Data Files
- Use CSV for tabular results
- Name format: `{experiment}_{metrics}_{date}.csv`
- Examples:
  - `fairness_metrics_summary_2025_11_05.csv`
  - `baseline_model_performance_2025_11_05.csv`
  - `federated_learning_metrics_2025_11_05.csv`

### Reports
- Use Markdown for summaries
- Archive versions with dates
- Examples:
  - `fairness_report_2025_11_05.md`
  - `experiment_log.md` (updated continuously)

## Metrics to Track

### Model Performance Metrics
```python
# Typical metrics for healthcare AI
metrics = {
    'accuracy': float,
    'sensitivity': float,  # True Positive Rate
    'specificity': float,  # True Negative Rate
    'precision': float,
    'f1_score': float,
    'auc_roc': float,
    'auc_pr': float,
    'calibration_error': float,
}
```

### Fairness Metrics
```python
fairness_metrics = {
    'demographic_parity_difference': float,  # Max - Min positive rate
    'equalized_odds_difference': float,      # Max - Min TPR/FPR
    'disparate_impact_ratio': float,         # Min / Max positive rate
    'calibration_by_group': dict,            # By demographic group
    'performance_by_group': dict,            # By demographic group
}
```

### Federated Learning Metrics
```python
federated_metrics = {
    'global_model_accuracy': float,
    'convergence_rounds': int,
    'total_communication_cost': float,
    'privacy_budget_epsilon': float,
    'institutional_variance': float,
    'client_participation_rate': float,
}
```

## Creating Results

### Python Code Template

```python
# experiments/save_results_template.ipynb
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Setup
results_dir = Path('../results')
experiment_name = 'baseline_lstm'
timestamp = datetime.now().strftime('%Y_%m_%d')

def save_metrics(metrics_dict, experiment, timestamp):
    """Save metrics to CSV."""
    df = pd.DataFrame([metrics_dict])
    filepath = results_dir / f'{experiment}_metrics_{timestamp}.csv'
    df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")
    return filepath

def save_figure(fig, experiment, metric, timestamp):
    """Save matplotlib figure."""
    filepath = results_dir / f'{experiment}_{metric}_{timestamp}.png'
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {filepath}")
    return filepath

def save_results_summary(summary_dict, filepath):
    """Save JSON summary."""
    with open(filepath, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Summary saved to {filepath}")

# Example usage
if __name__ == '__main__':
    # Calculate metrics
    metrics = {
        'accuracy': 0.92,
        'auc_roc': 0.89,
        'demographic_parity_difference': 0.08,
    }
    
    # Save metrics
    save_metrics(metrics, experiment_name, timestamp)
    
    # Save figure
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    save_figure(fig, experiment_name, 'roc_curve', timestamp)
    plt.close(fig)
```

## For Nature Medicine Submission

### Figure Requirements

1. **Main Text Figures** (up to 5):
   - High resolution (300 dpi minimum)
   - Clear labeling and legends
   - Colorblind-friendly palettes
   - Figure captions (in main manuscript)

2. **Supplementary Figures**:
   - Detailed methodology diagrams
   - Ablation study results
   - Sensitivity analyses
   - Additional fairness metrics

3. **Table Requirements**:
   - Demographic characteristics (Table 1)
   - Model performance metrics (Table 2)
   - Fairness evaluation results (Table 3)
   - Institutional/demographic breakdowns (Tables 4-N)

### Data to Include

- Summary statistics by demographic group
- Confidence intervals for all metrics
- Statistical test results (p-values)
- Effect sizes where applicable
- Missing data handling documentation

## Reproducibility and Version Control

### Best Practices

1. **Use consistent random seeds**
   ```python
   np.random.seed(42)
   torch.manual_seed(42)
   ```

2. **Track versions of results**
   - Use dates in filenames
   - Include timestamp in code
   - Document changes in experiment_log.md

3. **Version control small results files**
   - Save CSV summaries (< 10MB)
   - Version manuscript figures
   - Commit experiment metadata

4. **Use .gitignore for large files**
   ```
   # Large plot files
   *.png
   *.pdf
   
   # Large data files
   *.parquet
   *.h5
   *.pkl
   
   # Or if committing plots:
   !results/manuscript_figures/*.png
   ```

## Experiment Tracking

Maintain `experiment_log.md`:

```markdown
# Experiment Log

## 2025-11-05: Baseline LSTM Model Training
- **Objective**: Establish baseline performance
- **Parameters**: hidden_size=128, num_layers=2
- **Results**: Accuracy=0.92, AUC-ROC=0.89
- **Fairness**: Demographic Parity Diff=0.08
- **Notes**: Model shows bias against minority groups
- **Next Steps**: Implement fairness constraints

## 2025-11-06: Fair Model Development
- **Objective**: Reduce fairness gap
- **Parameters**: Added fairness_weight=0.3
- **Results**: Accuracy=0.88 (↓0.04), AUC-ROC=0.85 (↓0.04)
- **Fairness**: Demographic Parity Diff=0.03 (↓0.05)
- **Notes**: Fairness-accuracy tradeoff observed
- **Next Steps**: Optimize fairness weight
```

## Manuscript Figure Workflow

1. **Experiment** → Generate results (CSV, plots)
2. **Review** → Check statistical significance, quality
3. **Archive** → Save to `manuscript_figures/`
4. **Document** → Add to manuscript draft with captions
5. **Revise** → Update figures based on reviewer feedback
6. **Final** → Submit high-quality production figures

## Results for Different Stakeholders

### For Internal Discussion
- All experimental variations
- Failed experiments (document lessons learned)
- Raw data and intermediate results
- Development iterations

### For Manuscript Submission
- Polished main figures (5-6 maximum)
- Comprehensive supplementary materials
- Aggregate statistics (no patient-level data)
- Reproducible results (seeds, parameters documented)

### For Public Repository
- Summary statistics (CSV format)
- Synthetic data (if using for examples)
- Figure templates
- Experiment metadata
- NOT: Raw healthcare data, patient-level results

## Quick Tips

1. **Colors**: Use colorblind-friendly palettes
   ```python
   import seaborn as sns
   sns.set_palette('colorblind')
   ```

2. **Fonts**: Ensure readability in figures
   ```python
   plt.rcParams['font.size'] = 12
   plt.rcParams['axes.labelsize'] = 14
   ```

3. **Resolution**: Always use 300 dpi for publications
   ```python
   fig.savefig('plot.png', dpi=300, bbox_inches='tight')
   ```

4. **Documentation**: Include metadata in figure filenames
   - Experiment type
   - Metric evaluated
   - Date generated

## Troubleshooting

### Common Issues

- **Large file sizes**: Use PNG compression or PDF
- **Missing data**: Document in README with explanation
- **Results inconsistency**: Check random seeds and parameters
- **Fairness metrics confusion**: Refer to literature_review/ for definitions

## Last Updated
November 2025

---

**Note**: All results should support our goal of Nature Medicine publication. Maintain rigorous methodology, thorough documentation, and reproducible results. Include fairness evaluation alongside traditional performance metrics.
