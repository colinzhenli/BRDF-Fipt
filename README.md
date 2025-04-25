# Neural BRDF Learning

This project implements a neural network-based approach for learning Bidirectional Reflectance Distribution Functions (BRDFs) from rendered images. The system uses a point-based lighting setup for training and can be tested with environment map lighting.

## Introduction

The Neural BRDF Learning project allows you to:
1. Train a neural BRDF model using point light sources
2. Test the trained model with environment map lighting
3. Visualize the results of the learned material properties

The implementation includes:
- Dynamic point light emitters for training
- Environment map lighting for testing
- Path tracing for global illumination
- Neural network-based BRDF representation

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- OpenEXR
- NumPy
- OpenCV
- imageio

### Setup

1. Clone the repository:
