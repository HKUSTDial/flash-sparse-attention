#!/usr/bin/env python3
"""
Test suite for varlen attention functions in flash_dmattn.

This test suite validates the correctness of the varlen attention functions
after fixing the bug where default mask and bias tensors had incorrect shapes.

Tests cover:
- flash_dmattn_varlen_func
- flash_dmattn_varlen_kvpacked_func  
- flash_dmattn_varlen_qkvpacked_func

Issue #113: RuntimeError: bias must have shape (total_q, num_heads_k, max_seqlen_k)
"""

import torch
import unittest
import sys
from typing import List, Tuple


class TestVarlenFunctions(unittest.TestCase):
    """Test cases for varlen attention functions."""
    
    def setUp(self):
        """Set up test environment."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping tests")
        
        # Import the functions we need to test
        try:
            from flash_dmattn import (
                flash_dmattn_varlen_func,
                flash_dmattn_varlen_kvpacked_func,
                flash_dmattn_varlen_qkvpacked_func,
                CUDA_AVAILABLE
            )
            
            if not CUDA_AVAILABLE:
                self.skipTest("CUDA backend not available")
                
            self.flash_dmattn_varlen_func = flash_dmattn_varlen_func
            self.flash_dmattn_varlen_kvpacked_func = flash_dmattn_varlen_kvpacked_func 
            self.flash_dmattn_varlen_qkvpacked_func = flash_dmattn_varlen_qkvpacked_func
            
        except ImportError as e:
            self.skipTest(f"Could not import flash_dmattn functions: {e}")
    
    def _create_test_tensors(self, 
                           seq_lens: List[int], 
                           num_heads: int = 16, 
                           head_dim: int = 64,
                           dtype: torch.dtype = torch.bfloat16) -> Tuple[torch.Tensor, ...]:
        """Create test tensors for varlen attention."""
        total_tokens = sum(seq_lens)
        max_seqlen = max(seq_lens)
        
        # Create queries, keys, values
        q = torch.randn(total_tokens, num_heads, head_dim, 
                       device=self.device, dtype=dtype)
        k = torch.randn(total_tokens, num_heads, head_dim, 
                       device=self.device, dtype=dtype)
        v = torch.randn(total_tokens, num_heads, head_dim, 
                       device=self.device, dtype=dtype)
        
        # Create cumulative sequence lengths
        cu_seqlens = torch.tensor([0] + seq_lens, device=self.device).cumsum(0)
        
        return q, k, v, cu_seqlens, max_seqlen
    
    def test_flash_dmattn_varlen_func_original_bug_scenario(self):
        """Test the exact scenario from bug report #113."""
        B = 3
        seq_lens = [512, 1024, 768]
        T = sum(seq_lens)  # 2304
        H, D = 16, 64
        
        q = torch.randn(T, H, D, device=self.device, dtype=torch.bfloat16)
        k = torch.randn(T, H, D, device=self.device, dtype=torch.bfloat16)
        v = torch.randn(T, H, D, device=self.device, dtype=torch.bfloat16)
        cu = torch.tensor([0] + seq_lens, device=self.device).cumsum(0)
        
        # This should NOT raise RuntimeError about bias shape
        try:
            output = self.flash_dmattn_varlen_func(
                query=q, key=k, value=v,
                cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=max(seq_lens), max_seqlen_k=max(seq_lens),
                is_causal=True
            )
            
            # Verify output shape
            self.assertEqual(output.shape, (T, H, D))
            self.assertEqual(output.device, self.device)
            self.assertEqual(output.dtype, torch.bfloat16)
            
        except RuntimeError as e:
            if "bias must have shape" in str(e):
                self.fail(f"Bug #113 not fixed: {e}")
            else:
                # Re-raise other RuntimeErrors (e.g., CUDA not available)
                raise
    
    def test_flash_dmattn_varlen_func_none_mask_bias(self):
        """Test flash_dmattn_varlen_func with None mask and bias."""
        seq_lens = [128, 256, 512]
        q, k, v, cu_seqlens, max_seqlen = self._create_test_tensors(seq_lens)
        
        # Test with None mask and bias (should create default tensors with correct shapes)
        output = self.flash_dmattn_varlen_func(
            query=q, key=k, value=v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            attn_mask=None,  # Should create (total_q, num_heads_k, max_seqlen_k)
            attn_bias=None,  # Should create (total_q, num_heads_k, max_seqlen_k)
            is_causal=True
        )
        
        total_tokens = sum(seq_lens)
        expected_shape = (total_tokens, 16, 64)  # (total_q, num_heads, head_dim)
        self.assertEqual(output.shape, expected_shape)
    
    def test_flash_dmattn_varlen_func_different_head_counts(self):
        """Test with different query and key head counts (GQA scenario)."""
        seq_lens = [64, 128]
        total_tokens = sum(seq_lens)
        max_seqlen = max(seq_lens)
        head_dim = 64
        
        num_heads_q = 32
        num_heads_kv = 8  # GQA: fewer key/value heads
        
        q = torch.randn(total_tokens, num_heads_q, head_dim, 
                       device=self.device, dtype=torch.bfloat16)
        k = torch.randn(total_tokens, num_heads_kv, head_dim, 
                       device=self.device, dtype=torch.bfloat16)
        v = torch.randn(total_tokens, num_heads_kv, head_dim, 
                       device=self.device, dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0] + seq_lens, device=self.device).cumsum(0)
        
        # This should work with correct default tensor shapes
        output = self.flash_dmattn_varlen_func(
            query=q, key=k, value=v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            is_causal=False
        )
        
        expected_shape = (total_tokens, num_heads_q, head_dim)
        self.assertEqual(output.shape, expected_shape)
    
    def test_flash_dmattn_varlen_kvpacked_func_none_mask_bias(self):
        """Test flash_dmattn_varlen_kvpacked_func with None mask and bias."""
        seq_lens = [256, 512]
        total_tokens = sum(seq_lens)
        max_seqlen = max(seq_lens)
        num_heads = 16
        head_dim = 64
        
        q = torch.randn(total_tokens, num_heads, head_dim, 
                       device=self.device, dtype=torch.bfloat16)
        # KV packed: (total_tokens, 2, num_heads, head_dim)
        kv = torch.randn(total_tokens, 2, num_heads, head_dim, 
                        device=self.device, dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0] + seq_lens, device=self.device).cumsum(0)
        
        # Test with None mask and bias
        output = self.flash_dmattn_varlen_kvpacked_func(
            q=q, kv=kv,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            attn_mask=None,  # Should create (total_q, num_heads_k, max_seqlen_k) 
            attn_bias=None,  # Should create (total_q, num_heads_k, max_seqlen_k)
            is_causal=True
        )
        
        expected_shape = (total_tokens, num_heads, head_dim)
        self.assertEqual(output.shape, expected_shape)
    
    def test_flash_dmattn_varlen_qkvpacked_func_none_mask_bias(self):
        """Test flash_dmattn_varlen_qkvpacked_func with None mask and bias."""
        seq_lens = [128, 256, 384]
        total_tokens = sum(seq_lens)
        max_seqlen = max(seq_lens)
        num_heads = 8
        head_dim = 128
        
        # QKV packed: (total_tokens, 3, num_heads, head_dim)
        qkv = torch.randn(total_tokens, 3, num_heads, head_dim, 
                         device=self.device, dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0] + seq_lens, device=self.device).cumsum(0)
        
        # Test with None mask and bias
        output = self.flash_dmattn_varlen_qkvpacked_func(
            qkv=qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            attn_mask=None,  # Should create (total_tokens, num_heads, max_seqlen)
            attn_bias=None,  # Should create (total_tokens, num_heads, max_seqlen)
            is_causal=False
        )
        
        expected_shape = (total_tokens, num_heads, head_dim)
        self.assertEqual(output.shape, expected_shape)
    
    def test_all_varlen_functions_edge_cases(self):
        """Test edge cases: single sequence, empty sequences, etc."""
        # Test single sequence
        seq_lens = [1024]
        q, k, v, cu_seqlens, max_seqlen = self._create_test_tensors(seq_lens, num_heads=4, head_dim=32)
        
        output = self.flash_dmattn_varlen_func(
            query=q, key=k, value=v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            is_causal=True
        )
        
        expected_shape = (1024, 4, 32)
        self.assertEqual(output.shape, expected_shape)
        
        # Test very short sequences
        seq_lens = [1, 2, 3]
        q, k, v, cu_seqlens, max_seqlen = self._create_test_tensors(seq_lens, num_heads=2, head_dim=16)
        
        output = self.flash_dmattn_varlen_func(
            query=q, key=k, value=v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            is_causal=False
        )
        
        expected_shape = (6, 2, 16)  # total_tokens=1+2+3=6
        self.assertEqual(output.shape, expected_shape)
    
    def test_tensor_shape_consistency(self):
        """Test that default mask/bias tensors have the correct shapes as documented."""
        seq_lens = [200, 300, 500]
        total_tokens = sum(seq_lens)
        max_seqlen = max(seq_lens)
        
        # Test flash_dmattn_varlen_func shapes
        num_heads_q = 16
        num_heads_k = 8  # Different for GQA
        head_dim = 64
        
        q = torch.randn(total_tokens, num_heads_q, head_dim, device=self.device, dtype=torch.bfloat16)
        k = torch.randn(total_tokens, num_heads_k, head_dim, device=self.device, dtype=torch.bfloat16)
        v = torch.randn(total_tokens, num_heads_k, head_dim, device=self.device, dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0] + seq_lens, device=self.device).cumsum(0)
        
        # Test with explicit mask/bias to verify expected shapes
        expected_mask_shape = (total_tokens, num_heads_k, max_seqlen)
        expected_bias_shape = (total_tokens, num_heads_k, max_seqlen)
        
        # Create explicit mask and bias with expected shapes
        mask = torch.ones(expected_mask_shape, device=self.device, dtype=torch.bfloat16)
        bias = torch.zeros(expected_bias_shape, device=self.device, dtype=torch.bfloat16)
        
        # This should work without any shape errors
        output = self.flash_dmattn_varlen_func(
            query=q, key=k, value=v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            attn_mask=mask,
            attn_bias=bias,
            is_causal=True
        )
        
        expected_output_shape = (total_tokens, num_heads_q, head_dim)
        self.assertEqual(output.shape, expected_output_shape)


def main():
    """Run the test suite."""
    # Set up test runner
    unittest.main(verbosity=2, exit=False)


if __name__ == "__main__":
    main()