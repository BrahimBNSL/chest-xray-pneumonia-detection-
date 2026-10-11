# Pneumonia Detection from Chest X-Ray Images

A deep learning project that classifies chest X-ray images as **normal** or **pneumonia** using a pretrained ResNet-50 model, with a web application that allows users to upload and analyze an image.

> **Disclaimer:** This project is an academic demonstration and a triage-support tool. It is not a medical device and does not replace a radiologist's assessment or a clinical examination. It has not been clinically validated or evaluated on populations outside its dataset.

## Objective

Evaluate what a pretrained neural network can learn from chest X-ray images and honestly measure its performance by comparing it with a baseline without fine-tuning, assessing stability across multiple training runs, and examining the limitations of the dataset.

## Dataset

The project uses the **Chest X-Ray Pneumonia** dataset (Kermany et al., 2018), available on Kaggle. It contains approximately 5,800 chest X-ray images from pediatric patients aged 1–5 years, divided into two classes.

The dataset is not included in this repository. It must be downloaded separately and placed in `data/chest_xray/`.

Points to consider:
- The training set is heavily imbalanced, with 74% pneumonia cases, while the test set has a lower proportion of 62.5%.
- Filenames provide only approximate patient identifiers. Therefore, patient overlap between the training and test sets is possible.

## Methodology

1. **Exploratory data analysis** (`notebooks/01_exploration.ipynb`): class distributions, file integrity, formats, and image brightness.
2. **Data processing** (`src/data.py`): conversion to three-channel grayscale, resizing to 224 × 224, light augmentation on the training set only, stratified group-based validation split, and class weights calculated from the training set.
3. **Baseline model** (`src/features.py`): a fully frozen ResNet-50 followed by logistic regression on its extracted features.
4. **Fine-tuning** (`src/train.py`): full ResNet-50 fine-tuning using AdamW with two learning rates, cosine scheduling, weighted loss, mixed-precision training, and early stopping based on validation loss.
5. **Stability analysis** (`src/ensemble.py`): three training runs with different random seeds, followed by probability averaging.
6. **Application** (`app/`): a Gradio interface that loads the model and analyzes an uploaded image.

The test set was not used to make any decisions, including selecting the regularization parameter C, choosing the classification threshold, or determining when to stop training.

## Results

Test set: 624 images. Decision threshold: 0.5.

### Baseline without fine-tuning

*Frozen ResNet-50 + logistic regression*

| Metric | Value |
|---|---:|
| Balanced accuracy | 0.772 |
| ROC AUC | 0.950 |

### Full fine-tuning

*Seed 42 — model used by the application*

| Metric | Value |
|---|---:|
| Balanced accuracy | 0.856 |
| ROC AUC | 0.961 |
| Accuracy | 0.888 |
| Pneumonia recall | 0.982 (7 missed cases out of 390) |
| Normal-class recall | 0.731 (63 false alarms out of 234) |
| Pneumonia precision | 0.859 |

### Stability across three random seeds

*Seeds: 42, 1, and 2*

| Metric | Value |
|---|---:|
| Balanced accuracy | 0.863 ± 0.013 |
| ROC AUC | 0.965 ± 0.004 |

### Ensemble of three models

The three-model ensemble achieved a balanced accuracy of 0.865, a statistically non-significant improvement over the average performance. The ensemble mainly confirms that the results are reproducible across training runs.

### Confusion Matrix — Seed 42

| Actual class | Predicted normal | Predicted pneumonia |
|---|---:|---:|
| **Normal (234)** | 171 | 63 |
| **Pneumonia (390)** | 7 | 383 |

## Limitations

- **Single dataset:** The data comes from one source and represents children aged 1–5 years. Performance on adults, images from other hospitals, or different imaging devices is unknown.
- **High false-alarm rate:** Approximately one in four healthy cases is classified as pneumonia. The model is tuned to prioritize pneumonia recall.
- **Potentially overoptimistic validation results:** Validation balanced accuracy (approximately 0.97) is substantially higher than test balanced accuracy (approximately 0.86). Therefore, validation performance is not a reliable estimate of real-world performance.
- **Possible patient overlap:** Reliable patient identifiers are unavailable, so patients may appear in both the training and test sets.
- **No explainability yet:** The project does not currently verify whether the model focuses on the lungs rather than medical tubes, image borders, or positioning artifacts.

## Project Structure

```text
chest-xray-pneumonia/
├── app/
│   ├── app.py              # Gradio interface
│   └── predictor.py        # Model loading and inference
├── notebooks/
│   └── 01_exploration.ipynb
├── src/
│   ├── config.py           # Paths and hyperparameters
│   ├── data.py             # Data loading, transformations, and splitting
│   ├── features.py         # Baseline: logistic regression
│   ├── model.py            # ResNet-50 architecture
│   ├── train.py            # Fine-tuning
│   ├── ensemble.py         # Averaging across multiple random seeds
│   └── utils.py            # Random seeds, device management, and logging
├── outputs/                # Checkpoints and metrics (not version-controlled)
└── README.md
```

## References

- Kermany, D. et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. Cell, 172(5).
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- Dataset: [Chest X-Ray Pneumonia — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
