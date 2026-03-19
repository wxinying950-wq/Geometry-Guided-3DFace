# Geometry-Driven Color Consistency and Texture Blending for High-Fidelity 3D Face Reconstruction

**📌 IMPORTANT NOTE:** This repository contains the core implementation code directly related to the manuscript submitted to **The Visual Computer**. 
If you find this codebase helpful for your research, please consider citing our paper (complete citation information will be updated upon formal publication).

## Overview
This repository provides the source code for our geometry-guided framework designed for precise color correction and seamless texture fusion in multi-view 3D face reconstruction. By leveraging dense geometric constraints extracted from 3D meshes, this algorithm establishes robust pixel-level correspondences. It effectively mitigates perspective errors, spatial misalignments, and non-uniform color distortions caused by illumination variations and camera discrepancies, ensuring high-precision 3D face modeling.

## Main Features
1. **Optimal Linear Transformation:** Achieves precise initial RGB channel alignment across multiple views.
2. **Two-Stage CIELAB Refinement:** Eliminates residual luminance inconsistencies in the CIELAB color space.
3. **Distance-Weighted Blending:** Fuses multi-view textures seamlessly to eliminate stitching artifacts and ensure high-fidelity visual smoothness.

## Environmental Requirements
To run this pipeline, please ensure your environment meets the following dependencies:
* C++ 14 or higher / Python 3.8+
* OpenCV (for image processing and color space conversions)
* Eigen3 (for matrix operations and linear transformations)
* *(If using Python)*: `numpy`, `opencv-python`, `trimesh`

## Core Parameters
* `alpha`: Weighting coefficient for the distance-based blending strategy.
* `L_threshold`: Luminance threshold for the CIELAB two-stage refinement process.
* `tolerance`: Spatial alignment tolerance used to filter out outlier pixel correspondences.
* `iter_max`: Maximum number of iterations for the optimal linear transformation matrix estimation.
* `blend_radius`: The effective operational radius for the distance-weighted blending mask.

## Quick Start (Demo)
1. Clone this repository to your local machine.
2. Navigate to the `demo/` directory.
3. Run the main script (e.g., `main.cpp` or `main.py`) to process the provided sample data. The output textured mesh will be saved in the `output/` directory.

## Data Availability & Privacy Statement
To facilitate reproducibility and allow reviewers/researchers to test the algorithmic pipeline, we have provided an anonymized, publicly available standard 3D face white model and sample multi-view texture patches in the `demo/` folder. 

**Privacy Note:** Due to strict institutional ethical guidelines and privacy regulations concerning human biometric facial data at our university, the original "Facial3D" dataset (comprising 74 sets of real 3D face scans used for the quantitative evaluations in the paper) cannot be made publicly available. The provided demo data is entirely sufficient to run the code, observe the execution flow, and verify the color consistency and texture blending effects.
