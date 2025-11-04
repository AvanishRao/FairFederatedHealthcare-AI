# Data Directory

⚠️ **IMPORTANT: DO NOT UPLOAD SENSITIVE OR PROTECTED HEALTH INFORMATION (PHI) TO THIS REPOSITORY** ⚠️

## Purpose

This directory contains:
- Data handling instructions and documentation
- Data preprocessing scripts and utilities
- Dataset descriptions and metadata
- Links to publicly available datasets
- Synthetic data generation scripts (for testing)

**This directory does NOT contain**:
- Raw patient data
- Protected Health Information (PHI)
- Personally Identifiable Information (PII)
- Any sensitive healthcare data

## Data Sources

All experiments should use publicly available, de-identified healthcare datasets:

### Primary Datasets

1. **MIMIC-III Clinical Database**
   - Website: https://physionet.org/content/mimiciii/1.4/
   - Description: De-identified ICU data from Beth Israel Deaconess Medical Center
   - Access: Requires PhysioNet credentialing and CITI training
   - Size: ~60,000 ICU admissions
   - License: PhysioNet Credentialed Health Data License

2. **eICU Collaborative Research Database**
   - Website: https://physionet.org/content/eicu-crd/2.0/
   - Description: Multi-center ICU database from 200+ hospitals
   - Access: Requires PhysioNet credentialing
   - Size: ~200,000 ICU admissions
   - License: PhysioNet Credentialed Health Data License

3. **UK Biobank** (if applicable)
   - Website: https://www.ukbiobank.ac.uk/
   - Description: Large-scale biomedical database
   - Access: Requires formal application and approval
   - License: Strict usage agreements

### Supplementary Datasets

- **NHANES** (National Health and Nutrition Examination Survey)
- **SEER** (Surveillance, Epidemiology, and End Results Program)
- **Medical Information Mart for Intensive Care (MIMIC-IV)**
- **COVID-19 Open Research Dataset (CORD-19)** (if relevant)

## Data Access Instructions

### PhysioNet Credentialing

1. Create account at https://physionet.org/
2. Complete CITI "Data or Specimens Only Research" course
3. Submit credentialing application
4. Sign data use agreement
5. Download datasets to secure local storage

**⚠️ Never upload credentialed data to GitHub or any cloud repository**

### Local Data Storage

```
# Recommended local directory structure (NOT in this repo)
~/healthcare_data/
├── mimic-iii/
│   ├── raw/
│   └── preprocessed/
├── eicu/
│   ├── raw/
│   └── preprocessed/
└── synthetic/  # Can be version controlled for testing
```

## Data Preprocessing

Data preprocessing scripts should:
1. Load data from local secure storage (outside this repo)
2. Perform de-identification checks
3. Apply preprocessing transformations
4. Save processed data to local secure storage
5. Generate only aggregate statistics for version control

### Example Preprocessing Workflow

```python
# data/preprocess_mimic.py (can be in repo)
import pandas as pd
import numpy as np
from pathlib import Path

# Data paths (user-specific, stored in config or environment variables)
DATA_ROOT = Path.home() / 'healthcare_data'
MIMIC_RAW = DATA_ROOT / 'mimic-iii' / 'raw'
MIMIC_PROCESSED = DATA_ROOT / 'mimic-iii' / 'preprocessed'

def load_raw_data():
    """Load raw MIMIC-III data from local storage."""
    # Implementation here
    pass

def preprocess_data(df):
    """Apply preprocessing transformations."""
    # Implementation here
    pass

def verify_deidentification(df):
    """Verify no PHI remains in processed data."""
    # Check for names, dates, IDs, etc.
    pass

if __name__ == '__main__':
    # Only save aggregate statistics to repo
    df = load_raw_data()
    df_processed = preprocess_data(df)
    verify_deidentification(df_processed)
    
    # Save to local storage (not in repo)
    df_processed.to_parquet(MIMIC_PROCESSED / 'processed.parquet')
    
    # Save only summary statistics to repo
    summary = df_processed.describe()
    summary.to_csv('../results/data_summary_statistics.csv')
```

## Data Privacy and Security

### Compliance Requirements

- **HIPAA**: Health Insurance Portability and Accountability Act (US)
- **GDPR**: General Data Protection Regulation (EU)
- **Data Use Agreements**: Specific to each dataset

### Best Practices

1. **Never commit raw healthcare data to version control**
2. Use `.gitignore` to exclude data directories:
   ```
   # .gitignore entries
   data/raw/
   data/processed/
   *.csv  # if containing patient-level data
   *.parquet
   *.h5
   *.pkl
   ```

3. **Use environment variables or config files for data paths**
   ```python
   # config.yaml (not in repo, or template only)
   data:
     root: /path/to/your/secure/data/storage
     mimic_iii: ${data.root}/mimic-iii
     eicu: ${data.root}/eicu
   ```

4. **Implement access controls**
   - Encrypt data at rest
   - Use secure file permissions
   - Maintain audit logs

5. **De-identification verification**
   - Remove direct identifiers (names, IDs)
   - Generalize quasi-identifiers (dates, locations)
   - Verify k-anonymity if required

## Synthetic Data for Testing

For development and testing without real patient data:

```python
# data/generate_synthetic_data.py (can be in repo)
import numpy as np
import pandas as pd
from faker import Faker

def generate_synthetic_ehr(n_patients=1000, n_timesteps=24):
    """
    Generate synthetic EHR time-series data for testing.
    
    This data has NO relationship to real patients and is safe to version control.
    """
    fake = Faker()
    
    data = {
        'patient_id': range(n_patients),
        'age': np.random.randint(18, 90, n_patients),
        'gender': np.random.choice(['M', 'F'], n_patients),
        # Add synthetic vital signs, lab values, etc.
    }
    
    return pd.DataFrame(data)

if __name__ == '__main__':
    # Generate and save synthetic data (safe to commit)
    synthetic_df = generate_synthetic_ehr()
    synthetic_df.to_csv('synthetic_test_data.csv', index=False)
    print(f"Generated {len(synthetic_df)} synthetic patient records")
```

## Dataset Metadata

Document dataset characteristics (no patient-level data):

### MIMIC-III Summary
- **Patients**: ~46,000 unique patients
- **Admissions**: ~58,000 ICU admissions
- **Time period**: 2001-2012
- **Variables**: Demographics, vital signs, lab results, medications, notes
- **Temporal resolution**: Irregular time-series

### eICU Summary
- **Patients**: ~200,000 unique patients
- **Hospitals**: 200+ across the United States
- **Time period**: 2014-2015
- **Variables**: Similar to MIMIC-III with multi-center data

## Fairness Considerations

When working with healthcare data, consider:

1. **Demographic representation**
   - Document population demographics
   - Identify underrepresented groups
   - Plan strategies for addressing imbalance

2. **Data quality disparities**
   - Assess completeness across demographic groups
   - Document missing data patterns
   - Consider impact on fairness metrics

3. **Proxy variables**
   - Identify potential proxy variables for sensitive attributes
   - Document correlations that might lead to indirect discrimination

## Data Documentation

Maintain comprehensive documentation:

- `data_dictionary.md` - Variable definitions and coding schemes
- `cohort_selection.md` - Inclusion/exclusion criteria
- `preprocessing_pipeline.md` - Step-by-step data transformations
- `quality_checks.md` - Data quality validation procedures

## For Nature Medicine Submission

Ensure data transparency:

1. **Data Availability Statement**
   - Specify how others can access the datasets
   - Include links to PhysioNet, UK Biobank applications
   - Note any restrictions

2. **Reproducibility**
   - Document exact dataset versions used
   - Provide preprocessing scripts
   - Share aggregate statistics and metadata

3. **Ethics Approval**
   - Institutional Review Board (IRB) approval or exemption
   - Data use agreement compliance documentation

## Quick Reference

### ✅ Safe to commit to repository:
- Data documentation and metadata
- Preprocessing scripts
- Synthetic data (clearly labeled)
- Aggregate statistics
- Data loader utilities
- Configuration templates

### ❌ Never commit to repository:
- Raw patient data
- Protected Health Information (PHI)
- Personally Identifiable Information (PII)
- Credentialed dataset files
- Patient-level data of any kind

## Contact

For questions about data access or usage:
- Review main README.md
- Consult PhysioNet documentation
- Open an issue on GitHub (without sharing sensitive information)

## Last Updated
November 2025

---

**Remember**: When in doubt, DON'T commit data files. Our research aims for Nature Medicine publication—maintaining the highest standards of data privacy and ethics is essential.
