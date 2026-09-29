# FaceStitche C++ reproduction demo

This repository contains a C++17 demo of the image correction, 3D correspondence verification, and mesh colouring stages of FaceStitche. It runs on a recorded capture; camera acquisition, point-cloud registration, and mesh reconstruction have already been performed. The executable entry point is [`Code/cpp/demo.cpp`](Code/cpp/demo.cpp), built as `facestitche_demo`. 

## Requirements

- Windows x64 with Visual Studio 2022 **Desktop development with C++** and a Windows SDK.
- CMake 3.16 or newer, available as `cmake` in PowerShell.
- No Python, Qt, PCL, camera SDK, or separate OpenCV installation is needed for the bundled build. The repository supplies the OpenCV 4.5.0 headers, MSVC import library, and a compressed runtime DLL under `Env/opencv/`. CMake extracts the DLL during configuration.

The bundled OpenCV files are for Windows x64 and MSVC. Other platforms need a compatible OpenCV installation and their own build configuration.

## Build and run

Open PowerShell in the **repository root** (the directory containing `Code/`, `Env/`, `Input/`, and this README):

```powershell
cmake -S Code/cpp -B Code/cpp/build -G "Visual Studio 17 2022" -A x64
cmake --build Code/cpp/build --config Release
.\Code\cpp\build\Release\facestitche_demo.exe --root .
```

CMake selects `Env/opencv` using a path relative to `Code/cpp/CMakeLists.txt` and extracts the bundled `opencv_world450.dll.zip` when the DLL is not already present. The build copies `opencv_world450.dll` beside the executable, so no machine-specific `PATH` edit is required. `--root` names the directory containing `Input/` and `Output/`; if omitted, the current working directory is used. The program prints `wrote 121012 vertices and 237779 faces` for the supplied data.

If you use your own compatible OpenCV, configure an empty build directory with `-DOPENCV_INCLUDE_DIR=...`, `-DOPENCV_LIBRARY_DIR=...`, and `-DOPENCV_LIBRARY=...`. Its runtime DLL must then be available to Windows when the executable starts.

## Repository layout and inputs

```text
Code/cpp/demo.cpp       C++ entry point and fixed capture parameters
Code/cpp/CMakeLists.txt  Build configuration
Env/opencv/              Bundled OpenCV 4.5.0 headers, import library, and compressed runtime DLL
Input/                   Five files from one recorded capture
Output/                  Reference outputs; regenerated when the demo runs
```

The five required files in `Input/` are:

| File | Format and role |
| --- | --- |
| `rgb_face_roi_l.png` | Left colour ROI PNG, 839 × 768 pixels; OpenCV reads it as BGR. |
| `rgb_face_roi_r.png` | Right colour ROI PNG, 839 × 768 pixels; OpenCV reads it as BGR. |
| `point_lift.pcd` | Organized left raw XYZ cloud, binary PCD with float32 `x y z`, 1224 × 1024 points. The original project uses the spelling `lift`. |
| `point_right.pcd` | Organized right raw XYZ cloud in the same format and shape. |
| `mesh.ply` | Reconstructed binary little-endian PLY mesh in the right cloud's coordinate system. |

The separate left and right camera `K/Kc/R/T` values, SAC and ICP transforms, colour matrix `M`, RGB width, and padded ROI offsets are embedded in `recordedParameters()` in `demo.cpp`. The right raw depth ROI is fixed at `u=563, v=123, width=348, height=418`. A different capture requires replacing all five inputs and updating these matching fixed parameters together.

## Expected outputs

| File under `Output/` | Contents |
| --- | --- |
| `corrected_left.png` | Corrected left ROI. |
| `stitched_raw.png`, `stitched_matrix.png`, `stitched_reinhard.png` | Raw, matrix-corrected, and Reinhard-corrected montages. |
| `colorfacemesh.obj`, `colorfacemesh.mtl` | Mesh with `v x y z r g b` vertex colours and normals; no texture map. |
| `depth_consistency.csv` | One row per checked candidate with keep/reject reason, projected pixels, nearest observed right point, 3D distance, and depth difference. |
| `metrics.json` | Embedded parameters, depth-filter counts, projection counts, and colour-difference summaries. |

The executable clears existing regular files in `Output/` before writing new results. Move aside any results you want to preserve before running it. Check `metrics.json` → `depth_consistency` and the `KEEP` rows of `depth_consistency.csv` first. Colour-difference summaries describe these selected samples, not a cross-capture accuracy benchmark.
