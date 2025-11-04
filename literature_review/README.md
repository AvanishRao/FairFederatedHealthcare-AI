# Literature Review

This directory contains organized notes, summaries, and references to key papers relevant to our Fair Federated Healthcare AI research.

## Research Themes

### 1. Federated Learning in Healthcare

#### Key Papers

**Privacy-Preserving Federated Learning**
- [Privacy-preserving Federated Learning in Healthcare: A Systematic Review](https://doi.org/10.1038/s41591-020-1034-2)
  - Nature Medicine, 2020
  - Key insights: Differential privacy mechanisms, secure aggregation protocols
  - Relevance: Foundation for our privacy-preserving approach

- [FedHealth: A Federated Transfer Learning Framework for Wearable Healthcare](https://doi.org/10.1109/JIOT.2020.2992717)
  - IEEE Internet of Things Journal, 2020
  - Key insights: Transfer learning in federated settings, personalization strategies
  - Relevance: Multimodal data handling approaches

**Clinical Applications**
- [The Future of Digital Health with Federated Learning](https://www.nature.com/articles/s41746-020-00323-1)
  - npj Digital Medicine, 2020
  - Key insights: Real-world deployment challenges, regulatory considerations
  - Relevance: Practical implementation guidance

### 2. Fairness in Medical AI

#### Key Papers

**Algorithmic Fairness**
- [Addressing algorithmic bias and the perpetuation of health inequities](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(22)00003-7/fulltext)
  - The Lancet Digital Health, 2022
  - Key insights: Sources of bias in healthcare AI, mitigation strategies
  - Relevance: Fairness metrics selection and implementation

- [Ensuring Fairness in Machine Learning to Advance Health Equity](https://doi.org/10.7326/M18-1990)
  - Annals of Internal Medicine, 2018
  - Key insights: Demographic parity, equalized odds in clinical context
  - Relevance: Foundation for our fairness evaluation framework

**Health Equity**
- [Machine Learning for Healthcare: On the Verge of a Major Shift in Healthcare Epidemiology](https://doi.org/10.1093/cid/ciy712)
  - Clinical Infectious Diseases, 2018
  - Key insights: Disparities across demographic groups, geographic variations
  - Relevance: Global fairness objectives

### 3. Multimodal Healthcare Data Fusion

#### Key Papers

**Deep Learning Architectures**
- [Multimodal Machine Learning for Clinical Decision Support](https://arxiv.org/abs/2107.11229)
  - arXiv, 2021
  - Key insights: CNN-LSTM architectures, attention mechanisms
  - Relevance: Our multimodal model architecture

- [Deep Learning for Healthcare: Review, Opportunities and Challenges](https://doi.org/10.1093/bib/bbx044)
  - Briefings in Bioinformatics, 2019
  - Key insights: Temporal modeling, feature fusion strategies
  - Relevance: Baseline model design

**EHR Time-Series Analysis**
- [Deep EHR: A Survey of Recent Advances in Deep Learning Techniques for Electronic Health Record (EHR) Analysis](https://doi.org/10.1109/JBHI.2017.2767063)
  - IEEE Journal of Biomedical and Health Informatics, 2017
  - Key insights: LSTM for temporal patterns, attention mechanisms
  - Relevance: Time-series modeling approach

### 4. Clinical Validation and Deployment

#### Key Papers

**Model Evaluation**
- [Guidelines for developing and reporting machine learning predictive models in biomedical research](https://doi.org/10.1038/s41591-023-02502-2)
  - Nature Medicine, 2023
  - Key insights: TRIPOD-ML guidelines, reporting standards
  - Relevance: Nature Medicine submission requirements

- [A Roadmap for the Development of Human-Level Artificial General Medical Intelligence](https://doi.org/10.1038/s41591-022-01999-5)
  - Nature Medicine, 2022
  - Key insights: Clinical validation requirements, deployment considerations
  - Relevance: Evaluation methodology

## Additional Resources

### Datasets
- [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/)
- [eICU Collaborative Research Database](https://physionet.org/content/eicu-crd/2.0/)
- [UK Biobank](https://www.ukbiobank.ac.uk/)

### Frameworks and Tools
- [PySyft](https://github.com/OpenMined/PySyft) - Privacy-preserving federated learning
- [TensorFlow Federated](https://www.tensorflow.org/federated) - Federated learning framework
- [Fairlearn](https://fairlearn.org/) - Fairness assessment and mitigation
- [AI Fairness 360](https://aif360.mybluemix.net/) - IBM fairness toolkit

### Regulatory Guidance
- [FDA Guidelines on Software as a Medical Device (SaMD)](https://www.fda.gov/medical-devices/software-medical-device-samd)
- [EU AI Act - High-Risk AI Systems in Healthcare](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)
- [WHO Ethics and Governance of AI for Health](https://www.who.int/publications/i/item/9789240029200)

## Reading Notes

Detailed reading notes and paper summaries are organized in the `notes/` subdirectory:

- `federated_learning_notes.md` - Key insights from federated learning papers
- `fairness_metrics_notes.md` - Fairness definitions and metrics
- `multimodal_architectures_notes.md` - Deep learning architecture notes
- `clinical_validation_notes.md` - Validation and deployment considerations

## Conference Proceedings

### Relevant Conferences
- **NeurIPS** - Machine Learning and Health Workshop
- **ICML** - Healthcare track
- **AAAI** - AI for Social Impact
- **AMIA** - Annual Symposium
- **ML4H** - Machine Learning for Health

## Last Updated
November 2025

---

**Note**: This is a living document. Add new papers and resources as the project progresses. Prioritize papers from Nature Medicine, The Lancet Digital Health, and other top-tier journals for citation purposes.
