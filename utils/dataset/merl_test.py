#!/usr/bin/env python3
"""
Simple test script for MERLInterface - mirrors BRDFRead.cpp functionality.
"""

import torch
import math
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.dataset.MERL import MERLInterface


def main():
    
    brdf_path = "/home/featurize/data/alum-bronze.binary"
    
    # Initialize MERL interface
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    merl = MERLInterface(brdf_path, device=device)
    
    # Print out a 2x8x2x8 table of BRDF values (same as modified C++ with n=2)
    n = 2
    
    for i in range(n):
        if i!=1:
            continue
        theta_in = i * 0.5 * math.pi / n
        
        for j in range(4 * n):
            if j!=4:
                continue
            phi_in = j * 2.0 * math.pi / (4 * n)
            
            for k in range(n):
                theta_out = k * 0.5 * math.pi / n
                
                for l in range(4 * n):
                    phi_out = l * 2.0 * math.pi / (4 * n)
                    
                    # Query BRDF using spherical angles
                    theta_in_t = torch.tensor([theta_in], dtype=torch.float64, device=device)
                    phi_in_t = torch.tensor([phi_in], dtype=torch.float64, device=device)
                    theta_out_t = torch.tensor([theta_out], dtype=torch.float64, device=device)
                    phi_out_t = torch.tensor([phi_out], dtype=torch.float64, device=device)
                    
                    rgb = merl.lookup(theta_in_t, phi_in_t, theta_out_t, phi_out_t)
                    
                    # Print RGB values (same format as C++)
                    print(f"{float(rgb[0,0]):.6f} {float(rgb[0,1]):.6f} {float(rgb[0,2]):.6f}")


if __name__ == "__main__":
    main()

