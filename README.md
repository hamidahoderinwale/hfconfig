# Config.json Analysis Repository

A comprehensive analysis suite for exploring architectural configurations of AI models from the Hugging Face ecosystem, extracted from `config.json` files.

## Dataset

The underlying dataset is available on Hugging Face:

- [modelbiome/ai_ecosystem](https://huggingface.co/datasets/modelbiome/ai_ecosystem)
- [modelbiome/ai_ecosystem_withmodelcards](https://huggingface.co/datasets/modelbiome/ai_ecosystem_withmodelcards)

## Overview

This repository contains systematic analyses of model architectures, focusing on:
- **Architectural parameters** (hidden size, layers, attention mechanisms)
- **Feature adoption** (MoE, GQA, RoPE, quantization)
- **Family comparisons** (LLaMA, Mistral, GPT, BERT, DeepSeek, etc.)
- **Temporal evolution** (S-curves of feature diffusion)
- **Architectural fitness** (which configs lead to more descendants)
- **Config similarity** (genotype-based similarity graphs)
- **Behavioral integration** (architecture → capability relationships)

## Repository Structure

```
config-analysis-repo/
├── README.md                          # This file
├── CONFIG_SIMILARITY_README.md        # Detailed guide for similarity analyses
│
├── 01_contributor_and_summary_analysis.ipynb
├── 02_parameter_distributions.ipynb
├── 03_model_family_comparisons.ipynb
├── 04_moe_analysis.ipynb
├── 05_attention_and_context.ipynb
├── 06_scurve_architectural_diffusion.ipynb
├── 07_architectural_fitness.ipynb
│
├── 08_config_similarity_basics.ipynb
├── 09_config_drift_analysis.ipynb
├── 10_config_subgraph_analysis.ipynb
├── 11_config_behavioral_integration.ipynb
│
├── 00_config_similarity_graph_FULL_REFERENCE.ipynb  # Original comprehensive notebook
│
├── generate_quick_summary.py         # Quick text summary script
├── figures/                           # Generated visualizations
└── data/                              # Data files (see Setup below)
```

## Quick Start

### 1. Data Setup

Place the following data files in the `data/` directory:

- **`model_configs_expanded.csv`** (required)
  - Parsed and flattened config.json fields
  - Contains ~14,500 models with architectural parameters

- **`ai_ecosystem_graph*.pkl`** (optional, for drift/subgraph analyses)
  - NetworkX graph files with parent-child relationships
  - Available paths: `ai_ecosystem_graph_finetune_fulljson.pkl`, `ai_ecosystem_graph_nomerges.pkl`

- **LMArena data** (optional, for behavioral integration)
  - CSV/JSON with Elo scores and model names
  - See notebook 11 for data format requirements

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib scipy scikit-learn networkx
# Optional for temporal/behavioral analyses:
pip install datasets  # For HuggingFace dataset loading
```

### 3. Run Analyses

Start with the foundational notebooks:

1. **`01_contributor_and_summary_analysis.ipynb`** - Overview and summary statistics
2. **`02_parameter_distributions.ipynb`** - Parameter distributions and patterns
3. **`03_model_family_comparisons.ipynb`** - Family-level comparisons

Then explore specialized analyses:

4. **`04_moe_analysis.ipynb`** - Mixture of Experts deep dive
5. **`05_attention_and_context.ipynb`** - Attention mechanisms and context length
6. **`06_scurve_architectural_diffusion.ipynb`** - Temporal feature adoption
7. **`07_architectural_fitness.ipynb`** - Which architectures are most "fit"

For advanced similarity and drift analyses:

8. **`08_config_similarity_basics.ipynb`** - Core similarity computation
9. **`09_config_drift_analysis.ipynb`** - Config drift along parent-child edges
10. **`10_config_subgraph_analysis.ipynb`** - Subgraph coherence analysis
11. **`11_config_behavioral_integration.ipynb`** - Architecture-behavior integration

## Notebook Descriptions

### Core Analysis Notebooks

#### **01_contributor_and_summary_analysis.ipynb**
- Summary statistics: who contributes config.json files
- Top contributors and organizations
- Model type distributions
- Overall architectural landscape

#### **02_parameter_distributions.ipynb**
- Distributions of key parameters (hidden size, layers, vocab size)
- Parameter relationships and correlations
- Outlier detection and patterns

#### **03_model_family_comparisons.ipynb**
- Architectural comparisons across families (LLaMA, Mistral, GPT, BERT, etc.)
- Feature adoption rates by family
- Parameter medians and distributions
- Temporal adoption trends

#### **04_moe_analysis.ipynb**
- Mixture of Experts (MoE) adoption analysis
- Expert configuration patterns
- MoE vs dense model comparisons
- Leading MoE organizations

#### **05_attention_and_context.ipynb**
- Attention mechanism analysis (GQA, MQA, standard)
- Context length distributions and trends
- RoPE scaling patterns
- Attention head configurations

#### **06_scurve_architectural_diffusion.ipynb**
- S-curve analysis of feature adoption over time
- Diffusion of architectural innovations (dtype, context length, GQA)
- Comparison across task tags and licenses
- Temporal adoption patterns

#### **07_architectural_fitness.ipynb**
- Which architectures lead to more descendants
- Ecosystem success metrics (downloads, likes, spaces)
- Regression analysis: architecture → fitness
- Fitness landscapes by architecture family

### Config Similarity & Drift Notebooks

#### **08_config_similarity_basics.ipynb**
**Foundation notebook** - Core similarity computation
- Gower distance implementation (handles mixed numeric + categorical data)
- Similarity matrix computation
- Similarity graph construction
- Nearest neighbors analysis
- Similarity metrics comparison (Gower, L2, L1, Cosine)

#### **09_config_drift_analysis.ipynb**
- Config drift along parent-child relationships
- Within-family vs between-family drift
- Mutational landscape (which fields change most)
- Family-specific mutation profiles

#### **10_config_subgraph_analysis.ipynb**
- Architectural similarity within descendant clusters
- Subgraph coherence analysis
- Config drift by depth in lineage trees
- Cumulative drift along lineage paths

#### **11_config_behavioral_integration.ipynb**
- LMArena data integration (Elo scores)
- Behavioral drift analysis (config change → capability change)
- Architecture-capability regression (predicting Elo from config)
- Architecture vs behavioral cluster comparison
- Ecosystem fitness vs behavioral fitness

## Key Features

### 1. **Comprehensive Parameter Coverage**
- ~78 config fields analyzed
- Numeric, categorical, and boolean features
- Handles missing data gracefully

### 2. **Multiple Similarity Metrics**
- **Gower distance** (primary): Handles mixed data types
- **L2/Euclidean**: For numeric-only comparisons
- **L1/Manhattan**: More robust to outliers
- **Cosine similarity**: For high-dimensional embeddings

### 3. **Temporal Analysis**
- Feature adoption S-curves
- Parameter evolution over time
- Family adoption timelines

### 4. **Graph-Based Analysis**
- Similarity graphs (architecture-based)
- Family trees (genealogy-based)
- Subgraph coherence analysis
- Drift along lineage paths

### 5. **Behavioral Integration** (when data available)
- Config drift → behavioral drift correlations
- Architecture → capability regression
- Genotype-phenotype relationships

## Data Requirements

### Required
- `model_configs_expanded.csv`: Parsed config.json data
  - Columns: `modelId`, config fields (`config_hidden_size`, `config_num_hidden_layers`, etc.)
  - ~14,500 models

### Optional (for advanced analyses)
- Family graph pickle files: For parent-child relationship analysis
- LMArena data: For behavioral/capability integration
- Temporal metadata: For S-curve and temporal analyses

## Output Files

Each notebook generates:
- **CSV files**: Summary statistics and processed data
- **PNG figures**: Visualizations (saved to `figures/` directory)
- **Console output**: Key statistics and findings

## Methodology

### Similarity Computation
- **Gower distance** for mixed data types (recommended)
- Normalized numeric differences
- Categorical/boolean mismatch indicators
- Missing value handling

### Drift Analysis
- Parent-child pair extraction from graph or dataframe columns
- Config drift = Gower distance between parent and child
- Mutation rates = frequency of field changes

### Behavioral Integration
- Model name mapping (Arena → HuggingFace)
- Behavioral drift = ΔElo (child - parent)
- Regression: capability ~ architecture features

## Research Questions Addressed

1. **Which models are architecturally similar?**
2. **How does architecture drift along parent-child edges?**
3. **Which config fields mutate most frequently?**
4. **Are some families more architecturally stable?**
5. **How coherent are architectural clusters within family trees?**
6. **Does config drift predict behavioral drift?**
7. **Which architectural features predict capability?**
8. **Do architecture clusters align with behavioral clusters?**

## Related Work

This analysis extends the [AI Ecosystem](https://github.com/modelbiome/ai-ecosystem) project by:
- Extracting and analyzing architectural parameters from config.json files
- Computing similarity metrics for architectural comparison
- Analyzing architectural drift and mutation patterns
- Integrating behavioral/capability data (LMArena)

