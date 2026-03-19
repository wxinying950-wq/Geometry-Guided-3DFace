# Geometry-Driven Color Consistency and Texture Blending for High-Fidelity 3D Face Reconstruction

**📌 IMPORTANT NOTE:** This repository contains the core implementation code directly related to the manuscript submitted to **The Visual Computer**. 
If you find this codebase helpful for your research, please consider citing our paper (complete citation information will be updated upon formal publication).

## Overview
This repository provides the source code for our geometry-guided framework designed for precise color correction and seamless texture fusion in multi-view 3D face reconstruction. By leveraging dense geometric constraints extracted from 3D meshes, this algorithm establishes robust pixel-level correspondences. It effectively mitigates perspective errors, spatial misalignments, and non-uniform color distortions caused by illumination variations and camera discrepancies, ensuring high-precision 3D face modeling.

## Main Features
1. **Optimal Linear Transformation:** Achieves precise initial RGB channel alignment across multiple views.
2. **Two-Stage CIELAB Refinement:** Eliminates residual luminance inconsistencies in the CIELAB color space.
3. **Distance-Weighted Blending:** Fuses multi-view textures seamlessly to eliminate stitching artifacts and ensure high-fidelity visual smoothness.

## 🛠️ Environmental Requirements & Dependencies

This project is developed and tested strictly in a **Windows 10/11 (x64)** environment. Due to the stringent dependency chain of 3D point cloud processing and hardware interfacing, please ensure your environment is configured with the exact library versions listed below to guarantee reproducibility.

### 1. Compiler & Build System
* **IDE:** Microsoft Visual Studio 2022
* **Toolchains:** MSVC v143 (with backward compatibility components installed for `vc14` binaries)
* **CMake:** >= 3.20

### 2. Core 3D Processing Library (Strict Versioning)
* **PCL (Point Cloud Library) == 1.12.1**
  * *Crucial Note:* To prevent ABI incompatibilities and CMake linking errors, your PCL 1.12.1 installation MUST be built against the following specific 3rd-party versions:
    * **Boost == 1.78.0**
    * **VTK == 9.1.0**
    * **FLANN == 1.9.1**
    * **Qhull == 2020.2**
    * **Eigen3 == 3.3.9**

### 3. Computer Vision & Machine Learning
* **OpenCV >= 4.5.4** (Pre-built binaries for `vc14`/`vc15`)
* **dlib >= 19.22** (Compiled with AVX instruction set support for accelerated facial landmark extraction)

### 4. Graphics & Visualization
* **GLFW == 3.3.8** (Compiled for `vc2022`)
* **GLUT == 3.7.6**
* **OpenGL** (Native Windows support)

### 5. Hardware Acquisition (Optional)
* **DkamSDK (Release_x64):** Proprietary Depth Camera SDK. 
  * *Reproducibility Note:* This SDK is tightly coupled with our custom binocular hardware rig. Reviewers and researchers focusing on the algorithmic contributions (color correction & texture fusion) can safely bypass this dependency. We have provided an offline execution mode that directly ingests the standard `.obj` and `.png` files provided in the `demo/` folder.

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
