# Drug Response Prediction

A machine learning project for predicting drug response using genomic and chemical features from the GDSC (Genomics of Drug Sensitivity in Cancer) dataset.

## Project Overview

This project implements multiple approaches for predicting cancer cell line drug response:
- **Baseline models**: LightGBM with cross-validation
- **Neural networks**: Multi-layer perceptrons (MLP) with gene expression features
- **Chemical features**: ChemBERTa transformer embeddings for drug molecules
- **Graph neural networks**: GIN (Graph Isomorphism Network) for molecular graphs
- **Hybrid models**: Combined MLP + ChemBERTa features

## Project Structure

```
drug_response_prediction/
├── data/
│   ├── raw/                    # Raw GDSC data files
│   └── processed/              # Processed datasets and features
│       ├── gdsc_pairs.csv      # Drug-cell line pairs
│       ├── gdsc_expr_pca.csv   # PCA-reduced gene expression
│       ├── chemberta_drug_feats.npz  # ChemBERTa embeddings
│       ├── mol_graphs.pt       # Molecular graph structures
│       └── splits_drug/        # Train/val/test splits
├── scripts/
│   ├── make_dataset.py         # Data preprocessing pipeline
│   ├── make_splits.py          # Create train/val/test splits
│   ├── make_chemberta_features.py  # Generate ChemBERTa embeddings
│   ├── make_mol_graphs.py      # Create molecular graphs
│   ├── train_mlp_with_splits.py    # Train MLP baseline
│   ├── train_mlp_use_chemBERTa.py  # Train MLP+ChemBERTa
│   ├── train_gnn.py            # Train GNN baseline
│   ├── train_gnn_finetune.py   # Fine-tune GNN with hyperparameters
│   └── tune_mlp_chemberta.py   # Hyperparameter tuning with Optuna
├── notebooks/
│   ├── mlp_notebook.ipynb      # MLP experiments
│   ├── train_gnn_notebook.ipynb    # GNN experiments
│   └── tune_mlp_chemberta_notebook.ipynb  # Hyperparameter tuning
├── results/                    # Model outputs and metrics
│   ├── baseline_lgbm/
│   ├── mlp_baseline/
│   ├── mlp_chemberta/
│   ├── mlp_chemberta_tuned/
│   └── gnn_tuned/
├── src/                        # Source modules
│   ├── featurize.py
│   ├── interprets.py
│   └── splits.py
├── environment.yml             # Conda environment specification
└── requirements.txt            # Pip requirements
```

## Setup

### 1. Create Environment

**Using Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate drp
```

**Using pip:**
```bash
pip install -r requirements.txt
```

### 2. Data Preparation

Place raw GDSC data files in `data/raw/`:
- `Cell_line_RMA_proc_basalExp.txt` - Gene expression data
- `screened_compounds_rel_8.5.csv` - Drug screening results

Run preprocessing pipeline:
```bash
python scripts/make_dataset.py
python scripts/make_splits.py
```

### 3. Generate Features (Optional)

**ChemBERTa embeddings:**
```bash
python scripts/make_chemberta_features.py
```

**Molecular graphs:**
```bash
python scripts/make_mol_graphs.py
```

## Models

### 1. Baseline (LightGBM)
```bash
python scripts/drp_baseline.py
```

### 2. MLP Baseline
```bash
python scripts/train_mlp_with_splits.py
```

### 3. MLP + ChemBERTa
```bash
python scripts/train_mlp_use_chemBERTa.py
```

### 4. Graph Neural Network (GNN)

The GNN model uses a Graph Isomorphism Network (GIN) to encode molecular structures as graphs, combining them with gene expression features to predict drug response.

**Architecture:**
- GIN encoder with multiple convolutional layers for molecular graphs
- Node feature aggregation with global mean pooling
- MLP head for regression on concatenated features (expression + graph embeddings + optional tissue)

**Basic training:**
```bash
python scripts/train_gnn.py --splits_dir data/processed/splits_drug
```

**Fine-tuned with hyperparameters:**
```bash
python scripts/train_gnn_finetune.py \
    --splits_dir data/processed/splits_drug \
    --gnn_hidden 128 \
    --num_layers 3 \
    --mlp_hidden 256,128 \
    --epochs 50 \
    --batch_size 256 \
    --lr 0.001
```

**Hyperparameter tuning with Optuna:**
```bash
python scripts/train_gnn_finetune.py \
    --splits_dir data/processed/splits_drug \
    --tune \
    --n_trials 30 \
    --tune_epochs 30 \
    --epochs 50 \
    --out results/gnn_tuned
```

**Key parameters:**
- `--gnn_hidden`: Hidden dimension for GIN layers (64, 128, 256, 512)
- `--num_layers`: Number of GIN convolutional layers (2-4)
- `--mlp_hidden`: Hidden layer sizes for MLP head (e.g., "256,128")
- `--use_tissue`: Include tissue type as additional feature
- `--tune`: Enable hyperparameter search with Optuna
- `--patience`: Early stopping patience (default: 8)

### 5. Hyperparameter Tuning
```bash
python scripts/tune_mlp_chemberta.py --n-trials 50
```

## Results

Model performance is tracked in `results/` directory. Each model saves:
- Trained weights (`.pt` files)
- Validation and test metrics (`metrics_val_test.csv`)
- Predictions (`test_preds.npy`)

Key metrics evaluated:
- **Spearman correlation** (primary metric)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)

## Dependencies

Core libraries:
- PyTorch + PyTorch Geometric (neural networks)
- RDKit (molecular processing)
- Transformers (ChemBERTa)
- LightGBM (baseline models)
- Optuna (hyperparameter optimization)
- Pandas, NumPy, scikit-learn (data processing)

## Usage Examples

**Train MLP with ChemBERTa features:**
```bash
python scripts/train_mlp_use_chemBERTa.py \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.001 \
    --hidden-dim 256
```

**Fine-tune GNN:**
```bash
python scripts/train_gnn_finetune.py \
    --hidden-dim 128 \
    --num-layers 3 \
    --epochs 50 \
    --lr 0.0005
```

**Run hyperparameter search:**
```bash
python scripts/tune_mlp_chemberta.py \
    --n-trials 100 \
    --timeout 3600
```

## Notes

- GPU is recommended for neural network training
- Feature generation (ChemBERTa, molecular graphs) can be time-consuming
- Pre-computed features are cached in `data/processed/`
- Use drug-based splits to avoid data leakage

## License

This project is for research purposes.
