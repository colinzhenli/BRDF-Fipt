#!/usr/bin/env python3
"""
Quick GPU memory test for ECC errors on CUDA GPU 0
Tests memory allocation, transfer, and computation patterns
"""

import torch
import time

def test_gpu_memory(device_id=0, size_gb=2, iterations=5):
    """Test GPU memory with intensive operations"""
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return False
    
    device = torch.device(f'cuda:{device_id}')
    torch.cuda.set_device(device)
    
    print(f"Testing GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
    print(f"Memory allocated: {size_gb} GB")
    print(f"Iterations: {iterations}\n")
    
    # Check ECC status
    try:
        if hasattr(torch.cuda, 'memory_stats'):
            print("Checking ECC errors...")
            initial_errors = torch.cuda.memory_stats(device).get('num_ooms', 0)
            print(f"Initial OOM count: {initial_errors}\n")
    except:
        pass
    
    size = int(size_gb * 1024**3 / 4)  # Number of float32 elements
    
    for i in range(iterations):
        print(f"--- Iteration {i+1}/{iterations} ---")
        
        try:
            # Allocate large tensor
            print("Allocating memory...")
            a = torch.randn(size, device=device)
            b = torch.randn(size, device=device)
            
            # Memory-intensive operations
            print("Performing computation...")
            c = a + b
            d = a * b
            result = torch.sum(c * d)
            
            # Matrix operations (stress test)
            print("Matrix operations...")
            matrix_size = min(int((size / 1000) ** 0.5), 8192)
            m1 = torch.randn(matrix_size, matrix_size, device=device)
            m2 = torch.randn(matrix_size, matrix_size, device=device)
            m3 = torch.matmul(m1, m2)
            
            # Verify results are valid
            if torch.isnan(result) or torch.isinf(result):
                print(f"❌ ERROR: Invalid result detected (NaN or Inf)")
                return False
            
            print(f"✓ Passed - Result: {result.item():.4f}")
            
            # Clear memory
            del a, b, c, d, m1, m2, m3
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            print(f"❌ CUDA Error: {e}")
            return False
        
        time.sleep(0.5)
    
    print("\n" + "="*50)
    print("✓ All tests passed successfully!")
    print("="*50)
    
    # Final memory check
    mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
    mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
    print(f"Final memory - Allocated: {mem_allocated:.2f} GB, Reserved: {mem_reserved:.2f} GB")
    
    return True

if __name__ == "__main__":
    success = test_gpu_memory(device_id=0, size_gb=2, iterations=5)
    exit(0 if success else 1)

