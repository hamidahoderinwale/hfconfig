#!/usr/bin/env python3
"""
Quick summary script for config.json analysis.
Generates key statistics without requiring Jupyter.

Usage: python generate_quick_summary.py
"""

import pandas as pd
import numpy as np
import os

def main():
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'model_configs_expanded.csv')
    df = pd.read_csv(data_path, low_memory=False)
    
    # Convert boolean columns
    bool_cols = [c for c in df.columns if c.startswith('is_') or c.startswith('uses_')]
    for col in bool_cols:
        df[col] = df[col].map({'True': True, 'False': False, True: True, False: False})
    
    # Extract organization
    df['organization'] = df['modelId'].apply(
        lambda x: str(x).split('/')[0] if pd.notna(x) and '/' in str(x) else 'community'
    )
    
    print("=" * 80)
    print("CONFIG.JSON ECOSYSTEM ANALYSIS - QUICK SUMMARY")
    print("=" * 80)
    
    # === BASIC STATS ===
    print(f"\n📊 DATASET OVERVIEW")
    print(f"   Total models with config.json: {len(df):,}")
    print(f"   Unique organizations: {df['organization'].nunique():,}")
    print(f"   Unique model types: {df['config_model_type'].nunique():,}")
    
    # === TOP CONTRIBUTORS ===
    print(f"\n🏢 TOP 10 CONTRIBUTORS")
    top_orgs = df['organization'].value_counts().head(10)
    for i, (org, count) in enumerate(top_orgs.items(), 1):
        pct = count / len(df) * 100
        print(f"   {i:2d}. {org}: {count:,} models ({pct:.1f}%)")
    
    # === ARCHITECTURE FEATURES ===
    print(f"\n🔧 ARCHITECTURE FEATURE ADOPTION")
    features = {
        'MoE': (df['uses_moe'] == True).sum(),
        'GQA': (df['uses_gqa'] == True).sum(),
        'RoPE': (df['uses_rope'] == True).sum(),
        'Quantization': (df['uses_quantization'] == True).sum(),
        'LoRA': (df['uses_lora'] == True).sum(),
    }
    for feat, count in features.items():
        pct = count / len(df) * 100
        print(f"   {feat}: {count:,} models ({pct:.1f}%)")
    
    # === MODEL FAMILIES ===
    print(f"\n👨‍👩‍👧‍👦 MODEL FAMILIES")
    family_cols = [c for c in df.columns if c.startswith('is_') and 'family' in c]
    families = {}
    for col in family_cols:
        count = (df[col] == True).sum()
        if count > 0:
            name = col.replace('is_', '').replace('_family', '').title()
            families[name] = count
    
    for name, count in sorted(families.items(), key=lambda x: -x[1]):
        pct = count / len(df) * 100
        print(f"   {name}: {count:,} ({pct:.1f}%)")
    
    # === SIZE CATEGORIES ===
    print(f"\n📏 SIZE DISTRIBUTION")
    size_cats = df['size_category'].value_counts()
    for size in ['small', 'medium', 'large', 'xlarge']:
        if size in size_cats.index:
            count = size_cats[size]
            pct = count / size_cats.sum() * 100
            print(f"   {size.title()}: {count:,} ({pct:.1f}%)")
    
    # === CONTEXT LENGTH ===
    print(f"\n📐 CONTEXT LENGTH DISTRIBUTION")
    ctx_cats = df['context_category'].value_counts()
    for ctx in ['short', 'medium', 'long', 'very_long']:
        if ctx in ctx_cats.index:
            count = ctx_cats[ctx]
            pct = count / ctx_cats.sum() * 100
            label = ctx.replace('_', ' ').title()
            print(f"   {label}: {count:,} ({pct:.1f}%)")
    
    # === KEY PARAMETERS ===
    print(f"\n🔢 KEY PARAMETER STATISTICS")
    params = [
        ('Hidden Size', 'config_hidden_size'),
        ('Num Layers', 'config_num_hidden_layers'),
        ('Attention Heads', 'config_num_attention_heads'),
        ('Vocab Size', 'config_vocab_size'),
        ('Context Length', 'config_max_position_embeddings'),
    ]
    
    for name, col in params:
        data = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(data) > 0:
            print(f"   {name}:")
            print(f"      Median: {data.median():,.0f}")
            print(f"      Mean: {data.mean():,.0f}")
            print(f"      Range: [{data.min():,.0f}, {data.max():,.0f}]")
    
    # === MOE DETAILS ===
    moe_df = df[df['uses_moe'] == True]
    if len(moe_df) > 0:
        print(f"\n🧩 MOE MODEL DETAILS")
        experts = pd.to_numeric(moe_df['config_n_routed_experts'], errors='coerce').dropna()
        if len(experts) > 0:
            print(f"   Most common expert count: {int(experts.mode().iloc[0])}")
        per_tok = pd.to_numeric(moe_df['config_num_experts_per_tok'], errors='coerce').dropna()
        if len(per_tok) > 0:
            print(f"   Most common experts per token: {int(per_tok.mode().iloc[0])}")
        print(f"   Top MoE organizations:")
        for org, count in moe_df['organization'].value_counts().head(5).items():
            print(f"      {org}: {count}")
    
    # === GQA DETAILS ===
    gqa_df = df[df['uses_gqa'] == True]
    if len(gqa_df) > 0:
        print(f"\n⚡ GQA MODEL DETAILS")
        gqa_ratio = pd.to_numeric(gqa_df['config_gqa_ratio'], errors='coerce').dropna()
        if len(gqa_ratio) > 0:
            print(f"   Most common GQA ratio: {gqa_ratio.mode().iloc[0]:.0f}:1")
            for ratio, count in gqa_ratio.value_counts().head(5).items():
                print(f"      {ratio:.0f}:1 → {count:,} models")
    
    # === TOP MODEL TYPES ===
    print(f"\n🏗️ TOP 10 MODEL TYPES")
    for model_type, count in df['config_model_type'].value_counts().head(10).items():
        pct = count / len(df) * 100
        print(f"   {model_type}: {count:,} ({pct:.1f}%)")
    
    print("\n" + "=" * 80)
    print("END OF SUMMARY")
    print("=" * 80)

if __name__ == "__main__":
    main()
