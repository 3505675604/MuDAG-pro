## Directory Structure
```
MuDAG-Pro/
├── README.md
├── requirements.txt
├── config/
│   ├── base_config.yaml
│   └── llm_prompt_config.json
├── data/
│   ├── raw/                    # Raw data (Ignored by Git)
│   └── processed/              # Preprocessed intermediate files
├── knowledge_engine/           # Module 1: LLM + RAG Knowledge Engine
├── src/                        # Modules 2-3: Core Model Code
├── pipeline/                   # Module 4: Training & Evaluation Pipeline
├── analysis/                   # Module 5: Evaluation & Visualization
├── outputs/                    # Execution Results
└── main.py                     # Main Entry Point
```
---

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd MuDAG-Pro

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install required packages
pip install -r requirements.txt

# (Optional) Install PyTorch Geometric dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f [https://data.pyg.org/whl/torch-2.0.0+cu118.html](https://data.pyg.org/whl/torch-2.0.0+cu118.html)

```

---

## Data Preparation
### 1. Download Raw Data
Place the following datasets into their respective subdirectories under `data/raw/`:

| Dataset  | Data Types | Source |      
| TCGA-BRCA  | Expression + Mutation + Clinical | [GDC Portal](https://portal.gdc.cancer.gov/) |   
| METABRIC  | Expression + Mutation + Clinical | [cBioPortal](https://www.cbioportal.org/) |   
| SCAN-B  | Expression + Clinical | [GEO GSE96058](https://www.ncbi.nlm.nih.gov/geo/) |   
| GEO GSE2034  | Expression + Clinical | [GEO GSE2034](https://www.ncbi.nlm.nih.gov/geo/) |   
| Reactome |  Pathway Hierarchy | [Reactome Downloads](https://reactome.org/download-data/) |

### 2. Data Format Requirements
--Gene Expression Profile-- (`expression.csv`):
- Rows: Gene Symbols
- Columns: Patient Sample IDs
- Values: Normalized expression levels (TPM/FPKM → log2)
--Mutation Data-- (`mutation.csv`):
- Required Columns: `sample_id`, `gene`, `variant_type`, `protein_change`
--Clinical Data-- (`clinical.csv`):
- Required Columns: `sample_id`, `survival_time` (days), `event_status` (0=censored, 1=event)

### 3. Preprocessing
```bash
python main.py --mode preprocess
```
---

## Usage Guide
### Model Training
```bash
# Perform 5-fold cross-validation on the TCGA-BRCA training set
python main.py --mode train --config config/base_config.yaml
# Specify GPU device
python main.py --mode train --device cuda:0
```

### Inference & Evaluation
```bash
# Evaluate on independent test sets
python main.py --mode evaluate --checkpoint outputs/models/best_model.pt
# Specify target datasets
python main.py --mode evaluate --datasets metabric,scan_b,geo_gse2034
```

### Ablation Studies

```bash
# Run all ablation variants
python main.py --mode ablation --variants all
# Run specific variants
python main.py --mode ablation --variants M_static,M_onehot
```

### Knowledge Engine (Offline)

```bash
# Run LLM + RAG knowledge refinement to generate the regulatory rulebook
python main.py --mode knowledge_refine
# Skip if the rulebook is already generated
```
---

```bibtex
@article{mudagpro2024,
  title={MuDAG-Pro: Multi-modal Directed Acyclic Graph Propagation for Personalized Breast Cancer Prognosis},
  author={...},
  journal={...},
  year={2024}
}

```
---
