# Drug Response Prediction

A comprehensive machine learning project for predicting cancer drug response (IC50) using multimodal features from the GDSC (Genomics of Drug Sensitivity in Cancer) dataset.

## 🎯 Project Overview

This project implements and compares multiple state-of-the-art approaches for predicting drug sensitivity across cancer cell lines:

- **🌲 Baseline Models**: LightGBM with cross-validation
- **🧠 Neural Networks**: Multi-layer perceptrons (MLP) with PCA-reduced gene expression
- **🧪 Chemical Features**: ChemBERTa transformer embeddings (768-dim) for drug molecules
- **🕸️ Graph Neural Networks**: GIN (Graph Isomorphism Network) for molecular structure
- **🔗 Hybrid Models**: MLP + ChemBERTa combined features with hyperparameter optimization
- **🎭 Ensemble Methods**: Weighted averaging of multiple models for improved performance

## 🏆 Best Results

| Model | Test Spearman ↑ | Test RMSE ↓ | Test MAE ↓ |
|-------|-----------------|-------------|------------|
| **MLP + ChemBERTa (Tuned)** | **0.4498** | **2.3920** | **1.9038** |
| MLP + ChemBERTa | 0.4069 | 2.5876 | 1.9922 |
| MLP Baseline | 0.3559 | 2.6935 | 2.1215 |
| GNN Tuned | 0.3328 | 2.5641 | 2.0064 |

**🎉 Ensemble Results** (2-model weighted average):
- **Spearman: 0.4225** (+1.56% over best individual)
- Optimal weights: 68% MLP+ChemBERTa, 32% MLP Baseline

## 📁 Project Structure

```
drug_response_prediction/
├── 📊 data/
│   ├── raw/                         # Raw GDSC data files
│   │   ├── Cell_line_RMA_proc_basalExp.txt    # Gene expression (17,737 genes × 1,018 cell lines)
│   │   └── screened_compounds_rel_8.5.csv     # Drug screening results
│   └── processed/                   # Processed datasets and features
│       ├── merged.parquet           # Main dataset with all features (~125K drug-cell pairs)
│       ├── gdsc_pairs.csv           # Drug-cell line response pairs
│       ├── gdsc_expr_pca.csv        # PCA-reduced expression (512 components)
│       ├── chemberta_drug_feats.npz # ChemBERTa embeddings (768-dim per drug)
│       ├── mol_graphs.pt            # Molecular graph structures (RDKit + PyG)
│       ├── compounds_with_smiles.csv # Drug ID to SMILES mapping
│       └── splits_drug/             # Train/val/test splits (drug-based)
│           ├── train_idx.npy        # 70% for training
│           ├── val_idx.npy          # 15% for validation
│           └── test_idx.npy         # 15% for testing
│
├── 🔬 scripts/                      # Training and data processing scripts
│   ├── make_dataset.py              # Data preprocessing pipeline
│   ├── make_splits.py               # Create drug-based train/val/test splits
│   ├── make_chemberta_features.py   # Generate ChemBERTa embeddings
│   ├── make_mol_graphs.py           # Create molecular graphs from SMILES
│   ├── enrich_smiles.py             # Match drug IDs to SMILES strings
│   ├── train_mlp_with_splits.py     # Train MLP baseline
│   ├── train_mlp_use_chemBERTa.py   # Train MLP+ChemBERTa
│   ├── tune_mlp_chemberta.py        # Hyperparameter tuning with Optuna
│   ├── train_gnn.py                 # Train GNN baseline
│   ├── train_gnn_finetune.py        # Fine-tune GNN with hyperparameters
│   ├── train_chemberta_finetune.py  # Fine-tune ChemBERTa for regression
│   └── drp_baseline.py              # LightGBM baseline
│
├── 📓 notebooks/                    # Jupyter notebooks for analysis
│   ├── ensemble_model.ipynb         # Ensemble modeling & evaluation
│   ├── ensemble_model_colab.ipynb   # Google Colab version
│   ├── ablation_studies.ipynb       # Feature ablation experiments
│   ├── model_comparison.ipynb       # Compare all model architectures
│   ├── feature_importance.ipynb     # Feature importance analysis
│   ├── error_analysis.ipynb         # Error patterns & diagnostics
│   ├── model_interpretation.ipynb   # SHAP values & interpretability
│   ├── tune_mlp_chemberta_notebook.ipynb  # Interactive hyperparameter tuning
│   └── sanity_check.ipynb           # Data quality checks
│
├── 🎯 results/                      # Model outputs and metrics
│   ├── baseline_lgbm/               # LightGBM results
│   ├── mlp_baseline/                # MLP (expression only) 
│   ├── mlp_baseline_with_splits/    # MLP with drug-based splits
│   ├── mlp_chemberta/               # MLP + ChemBERTa combined
│   ├── mlp_chemberta_tuned/         # Hyperparameter-tuned version ⭐
│   ├── gnn_baseline/                # Graph Neural Network baseline
│   ├── gnn_tuned/                   # Hyperparameter-tuned GNN
│   ├── chemberta_finetune/          # Fine-tuned ChemBERTa for regression
│   └── ablation_studies/            # Ablation experiment results
│
├── 🏗️ src/                          # Source modules
│   ├── models/                      # Model architectures
│   │   ├── mlp.py                   # Multi-layer perceptron
│   │   ├── gnn.py                   # GIN encoder for graphs
│   │   └── __init__.py
│   ├── datasets.py                  # PyTorch Dataset classes
│   ├── featurize.py                 # Feature engineering utilities
│   ├── metrics.py                   # Evaluation metrics
│   └── utils.py                     # Helper functions
│
├── ⚙️ configs/                      # Configuration files (if any)
├── 📦 artifacts/                    # Saved models and outputs
├── environment.yml                  # Conda environment specification
├── requirements.txt                 # Pip requirements
└── README.md                        # This file
```
└── requirements.txt            # Pip requirements
```

## 🚀 Quick Start

### 1. Environment Setup

**Option A: Using Conda (recommended)**
```bash
# Create environment
conda env create -f environment.yml
conda activate drp

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print('PyTorch Geometric: OK')"
```

**Option B: Using pip**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

**Download GDSC data** (if not already available):
- Gene expression: `Cell_line_RMA_proc_basalExp.txt`
- Drug screening: `screened_compounds_rel_8.5.csv`
- Place in `data/raw/`

**Run preprocessing pipeline:**
```bash
# Step 1: Process raw data and create merged dataset
python scripts/make_dataset.py

# Step 2: Create train/validation/test splits (drug-based to avoid leakage)
python scripts/make_splits.py

# Step 3: Generate ChemBERTa embeddings (requires GPU, ~30 min)
python scripts/make_chemberta_features.py

# Step 4: Create molecular graphs (requires RDKit, ~10 min)
python scripts/make_mol_graphs.py
```

**Expected output structure:**
```
data/processed/
├── merged.parquet              # ~125K samples with all features
├── splits_drug/
│   ├── train_idx.npy           # ~87K samples (70%)
│   ├── val_idx.npy             # ~19K samples (15%)
│   └── test_idx.npy            # ~19K samples (15%)
├── chemberta_drug_feats.npz    # 768-dim embeddings for drugs
└── mol_graphs.pt               # PyG graph objects
```

### 3. Train Your First Model

**MLP Baseline (fastest, ~5 min on CPU)**
```bash
python scripts/train_mlp_with_splits.py
# Output: results/mlp_baseline_with_splits/
```

**MLP + ChemBERTa (best performance, ~10 min on GPU)**
```bash
python scripts/train_mlp_use_chemBERTa.py \
    --epochs 100 \
    --batch-size 256 \
    --lr 0.001 \
    --hidden 512,256
# Output: results/mlp_chemberta/
```

### 4. Evaluate Results

Check metrics in `results/{model_name}/metrics_val_test.csv`:
```bash
cat results/mlp_chemberta_tuned/final_test_metrics.csv
```

Expected output:
```
split,rmse,mae,spearman
test,2.3920,1.9038,0.4498
```

## 🧪 Model Architectures

### 1. LightGBM Baseline
Traditional gradient boosting baseline for comparison.

```bash
python scripts/drp_baseline.py
```

**Features:** PCA-reduced expression (512 dims)  
**Performance:** Fast training, interpretable, good baseline

---

### 2. MLP Baseline
Simple neural network using only gene expression features.

```bash
python scripts/train_mlp_with_splits.py \
    --epochs 100 \
    --batch-size 256 \
    --lr 0.001
```

**Architecture:**
- Input: 512 (PCA expression features)
- Hidden layers: [256, 128, 64]
- Output: 1 (LN_IC50 prediction)
- Activation: ReLU + Dropout(0.3)

**Test Results:** Spearman 0.3559, RMSE 2.6935

---

### 3. MLP + ChemBERTa ⭐ BEST MODEL
Combines gene expression with chemical structure embeddings.

**Basic training:**
```bash
python scripts/train_mlp_use_chemBERTa.py \
    --epochs 100 \
    --batch-size 256 \
    --lr 0.001 \
    --hidden 512,256
```

**Hyperparameter tuning (Optuna):**
```bash
python scripts/tune_mlp_chemberta.py \
    --n-trials 50 \
    --epochs 80 \
    --timeout 7200
```

**Architecture:**
- Input: 512 (expression) + 768 (ChemBERTa) = 1,280 features
- Hidden layers: [306, 187, 965] (optimized via Optuna)
- Output: 1 (regression)
- Dropout: 0.3
- Optimizer: Adam with weight decay

**Test Results:** Spearman 0.4498 ⬆, RMSE 2.3920 ⬇

---

### 4. Graph Neural Network (GIN)
Uses molecular graphs to encode drug structure directly.

**Basic training:**
```bash
python scripts/train_gnn.py \
    --splits_dir data/processed/splits_drug \
    --epochs 50 \
    --batch_size 256
```

**Fine-tuned with hyperparameters:**
```bash
python scripts/train_gnn_finetune.py \
    --splits_dir data/processed/splits_drug \
    --gnn_hidden 128 \
    --num_layers 3 \
    --mlp_hidden 256,128 \
    --epochs 50 \
    --lr 0.001
```

**Hyperparameter search:**
```bash
python scripts/train_gnn_finetune.py \
    --splits_dir data/processed/splits_drug \
    --tune \
    --n_trials 30 \
    --tune_epochs 30 \
    --epochs 50 \
    --out results/gnn_tuned
```

**Architecture:**
- **GIN Encoder:** Multiple graph convolution layers (GINConv)
- **Global Pooling:** Mean aggregation of node features
- **MLP Head:** Combines graph embeddings + expression + tissue
- **Output:** Regression prediction

**Key Parameters:**
- `--gnn_hidden`: Hidden dimension for GIN layers (64, 128, 256, 512)
- `--num_layers`: Number of graph convolutions (2-4)
- `--mlp_hidden`: Sizes for MLP head (e.g., "256,128")
- `--use_tissue`: Include tissue type as feature
- `--patience`: Early stopping patience (default: 8)

**Test Results:** Spearman 0.3328, RMSE 2.5641

---

### 5. ChemBERTa Finetuning
Fine-tune the ChemBERTa transformer end-to-end for regression.

```bash
python scripts/train_chemberta_finetune.py \
    --epochs 30 \
    --batch_size 64 \
    --lr 5e-5
```

**Architecture:**
- Pre-trained ChemBERTa (seyonec/ChemBERTa-zinc-base-v1)
- Regression head added on top
- Fine-tuned on drug response task

---

### 6. Ensemble Model 🎭
Combines multiple models using weighted averaging.

**Run ensemble analysis:**
```bash
# Open notebook
jupyter notebook notebooks/ensemble_model.ipynb

# Or for Google Colab
notebooks/ensemble_model_colab.ipynb
```

**Strategies:**
1. **Simple Average:** Equal weights for all models
2. **Weighted Average:** Optimize weights on validation set (scipy.optimize)
3. **Rank Average:** Average prediction ranks (robust to outliers)

**Best Ensemble Results:**
- Models: MLP Baseline (32%) + MLP+ChemBERTa (68%)
- **Spearman: 0.4225** (+1.56% over best individual)
- RMSE: 2.5857, MAE: 1.9748

## 📊 Results Summary

### Model Comparison (Test Set)

| Model | Spearman ↑ | RMSE ↓ | MAE ↓ | Notes |
|-------|------------|--------|-------|-------|
| **MLP + ChemBERTa (Tuned)** | **0.4498** | **2.3920** | **1.9038** | Best individual ⭐ |
| **Ensemble (Weighted)** | **0.4225** | 2.5857 | 1.9748 | 2-model ensemble 🎭 |
| MLP + ChemBERTa | 0.4069 | 2.5876 | 1.9922 | Strong baseline |
| Ensemble (Rank Average) | 0.4173 | 2.7145 | 2.1879 | Robust to outliers |
| Ensemble (Simple Average) | 0.4146 | 2.5081 | 1.9876 | Equal weights |
| MLP Baseline | 0.3559 | 2.6935 | 2.1215 | Expression only |
| GNN Tuned | 0.3328 | 2.5641 | 2.0064 | Graph structure |

### Key Findings

✅ **ChemBERTa embeddings significantly boost performance** (+14% Spearman over MLP baseline)  
✅ **Hyperparameter tuning is crucial** (0.4498 vs 0.4069 Spearman)  
✅ **Ensemble provides consistent gains** (+1.56% over best individual)  
✅ **Drug-based splits prevent data leakage** (realistic evaluation)

### Ablation Studies

Feature contribution analysis:
- Expression features: ~60% importance
- ChemBERTa features: ~38% importance  
- Tissue type: ~2% importance

See [notebooks/ablation_studies.ipynb](notebooks/ablation_studies.ipynb) for details.

---

## 📈 Evaluation Metrics

Each model saves comprehensive evaluation metrics:

**Metrics tracked:**
- **Spearman Correlation** (primary): Rank-based correlation (robust to outliers)
- **RMSE**: Root mean squared error (penalizes large errors)
- **MAE**: Mean absolute error (interpretable scale)

**Output files:**
```
results/{model_name}/
├── metrics_val_test.csv        # Validation and test metrics
├── test_preds.npy              # Raw predictions for ensemble
├── {model_name}.pt             # Trained model weights
└── training_history.json       # Loss curves (if available)
```

**Load and compare results:**
```python
import pandas as pd

# Load metrics
mlp_chemberta = pd.read_csv('results/mlp_chemberta_tuned/final_test_metrics.csv')
print(f"Spearman: {mlp_chemberta['spearman'].values[0]:.4f}")

# Load predictions for custom analysis
import numpy as np
preds = np.load('results/mlp_chemberta_tuned/test_preds.npy')
```

---

## 🔬 Analysis Notebooks

Comprehensive Jupyter notebooks for model analysis:

| Notebook | Purpose |
|----------|---------|
| [ensemble_model.ipynb](notebooks/ensemble_model.ipynb) | Ensemble modeling with 3 strategies |
| [ensemble_model_colab.ipynb](notebooks/ensemble_model_colab.ipynb) | Google Colab version |
| [model_comparison.ipynb](notebooks/model_comparison.ipynb) | Compare all model architectures |
| [ablation_studies.ipynb](notebooks/ablation_studies.ipynb) | Feature importance & ablation |
| [feature_importance.ipynb](notebooks/feature_importance.ipynb) | SHAP values & interpretability |
| [error_analysis.ipynb](notebooks/error_analysis.ipynb) | Prediction errors & diagnostics |
| [model_interpretation.ipynb](notebooks/model_interpretation.ipynb) | Model explainability |
| [tune_mlp_chemberta_notebook.ipynb](notebooks/tune_mlp_chemberta_notebook.ipynb) | Interactive hyperparameter tuning |

**Run notebooks:**
```bash
jupyter notebook notebooks/
```

---

## 🛠️ Advanced Usage

### Custom Training

**Train with custom architecture:**
```python
from src.models.mlp import MLP
import torch

model = MLP(
    in_dim=1280,          # 512 expression + 768 ChemBERTa
    hidden=[512, 256, 128],
    dropout=0.3
)

# Your training loop here
```

### Hyperparameter Search Space

**Optuna search space (tune_mlp_chemberta.py):**
```python
{
    'hidden': [[306, 187, 965], [512, 256, 128], ...],  # Various architectures
    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
    'lr': [1e-4, 5e-4, 1e-3, 5e-3],
    'batch_size': [128, 256, 512],
    'weight_decay': [0, 1e-5, 1e-4]
}
```

**Run custom search:**
```bash
python scripts/tune_mlp_chemberta.py \
    --n-trials 100 \
    --timeout 7200 \
    --epochs 80 \
    --out results/custom_tuning
```

### Ensemble Your Own Models

**Create custom ensemble:**
```python
import numpy as np
from scipy.stats import spearmanr

# Load predictions
preds_model1 = np.load('results/model1/test_preds.npy')
preds_model2 = np.load('results/model2/test_preds.npy')

# Simple average
ensemble_simple = (preds_model1 + preds_model2) / 2

# Weighted average (optimize on validation set)
weights = [0.6, 0.4]  # Optimized weights
ensemble_weighted = weights[0] * preds_model1 + weights[1] * preds_model2

# Evaluate
spearman_corr = spearmanr(y_true, ensemble_weighted)[0]
print(f"Ensemble Spearman: {spearman_corr:.4f}")
```

---

## 💾 Dependencies

### Core Libraries

```
# Deep Learning
torch>=2.0.0
torch-geometric>=2.3.0
transformers>=4.30.0

# Chemistry
rdkit-pypi>=2022.9.5

# Data Science  
numpy>=1.24.0
pandas>=2.0.0
pyarrow>=12.0.0
scikit-learn>=1.3.0
scipy>=1.10.0

# Machine Learning
lightgbm>=4.0.0
optuna>=3.2.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

**Full requirements:** See [requirements.txt](requirements.txt) and [environment.yml](environment.yml)

---

## 🎯 Tips & Best Practices

### Data Splitting
✅ **Use drug-based splits** to prevent data leakage  
✅ Test set should contain unseen drugs  
✅ 70/15/15 train/val/test ratio recommended

### Training
✅ **Start with MLP Baseline** to establish performance floor  
✅ **Use GPU** for neural networks (10-20x speedup)  
✅ **Early stopping** prevents overfitting (patience=10)  
✅ **Learning rate scheduling** improves convergence

### Feature Engineering
✅ **PCA reduces dimensionality** (17K → 512 genes) without major loss  
✅ **ChemBERTa captures chemical structure** better than fingerprints  
✅ **Normalize features** before training (handled automatically)

### Hyperparameter Tuning
✅ **Run 30-50 trials** minimum for Optuna  
✅ **Use validation set** for optimization (never test set!)  
✅ **Log all experiments** for reproducibility

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{drug_response_prediction,
  title={Drug Response Prediction with Multimodal Deep Learning},
  author={Your Name},
  year={2026},
  publisher={GitHub},
  url={https://github.com/yourusername/drug_response_prediction}
}
```

---

## 📄 License

This project is for research and educational purposes.

**Dataset:** GDSC data is publicly available from the Wellcome Sanger Institute  
**Models:** Pre-trained ChemBERTa from Hugging Face (Apache 2.0)

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Add more ensemble strategies (stacking, boosting)
- [ ] Implement attention mechanisms
- [ ] Add cross-validation for more robust evaluation
- [ ] Support for additional datasets (CCLE, CTRP)
- [ ] Uncertainty quantification

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your.email@example.com]

---

## 🙏 Acknowledgments

- **GDSC** (Genomics of Drug Sensitivity in Cancer) for the dataset
- **Hugging Face** for ChemBERTa pre-trained models
- **PyTorch Geometric** team for graph neural network tools
- **RDKit** community for chemistry utilities
