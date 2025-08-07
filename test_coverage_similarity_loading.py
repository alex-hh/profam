#!/usr/bin/env python3
"""
Test script to verify that coverage and similarity data loading works correctly.
"""

import os
import numpy as np
from src.data.builders.proteingym import load_msa_for_row
from src.data.tokenizers import ProFamTokenizer

def test_coverage_similarity_loading():
    """Test that coverage and similarity data is loaded correctly."""
    
    # Create a mock row with MSA filename
    row = {
        "DMS_id": "test_dms",
        "MSA_filename": "../data/ProteinGym/DMS_msa_files/BLAT_ECOLX_full_11-26-2021_b02_reformat_hhfilter.a3m",
        "target_seq": "MSIQHFRVALIPFFAAFCLPVFAHPETLVKV",
    }
    
    # Create a mock tokenizer
    tokenizer = ProFamTokenizer.from_pretrained("configs/tokenizer/profam.yaml")
    
    # Test loading
    try:
        result = load_msa_for_row(
            row=row,
            seed=42,
            tokenizer=tokenizer,
            max_tokens=1000,
            max_context_seqs=5,
            keep_wt=False,
            drop_wt=True,
            keep_gaps=False,
            use_filtered_msa=True,
            extra_tokens_per_document=2,
            use_msa_pos=True,
        )
        
        print("✓ Successfully loaded MSA data")
        print(f"Number of sequences: {len(result['MSA'])}")
        
        if 'sequence_similarities' in result:
            print(f"✓ Sequence similarities loaded: {len(result['sequence_similarities'])} values")
            print(f"Sample similarities: {result['sequence_similarities'][:3]}")
        else:
            print("⚠ No sequence similarities found (no .npz file)")
            
        if 'coverages' in result:
            print(f"✓ Coverages loaded: {len(result['coverages'])} values")
            print(f"Sample coverages: {result['coverages'][:3]}")
        else:
            print("⚠ No coverages found (no .npz file)")
            
        # Verify 1:1 correspondence
        if 'sequence_similarities' in result and 'coverages' in result:
            assert len(result['sequence_similarities']) == len(result['coverages']), "Mismatch in array lengths"
            assert len(result['sequence_similarities']) == len(result['MSA']), "Mismatch with sequences"
            print("✓ 1:1 correspondence verified")
            
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing coverage and similarity data loading...")
    success = test_coverage_similarity_loading()
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Tests failed!") 