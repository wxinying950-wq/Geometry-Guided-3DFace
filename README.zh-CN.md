# FaceStitche C++ 复现 Demo

本仓库包含 FaceStitche 图像校正、三维对应点验证和网格着色阶段的 C++17 Demo。它使用一次已记录的采集数据；相机采集、点云配准和网格重建已在运行前完成。可执行程序入口为 [`Code/cpp/demo.cpp`](Code/cpp/demo.cpp)，编译目标名为 `facestitche_demo`。

## 环境要求

- Windows x64、Visual Studio 2022 的 **“使用 C++ 的桌面开发”** 工作负载，以及 Windows SDK。
- CMake 3.16 或更新版本，并确保 PowerShell 可以调用 `cmake`。
- 使用仓库自带依赖编译时，无需 Python、Qt、PCL、相机 SDK 或单独安装 OpenCV。OpenCV 4.5.0 的头文件、MSVC 导入库和压缩后的运行时 DLL 位于 `Env/opencv/`，CMake 配置时会自动解压 DLL。

自带的 OpenCV 文件面向 Windows x64 和 MSVC。其他平台需要安装兼容版本的 OpenCV 并自行配置构建。

## 编译与运行

在**仓库根目录**（包含 `Code/`、`Env/`、`Input/` 和本 README 的目录）打开 PowerShell：

```powershell
cmake -S Code/cpp -B Code/cpp/build -G "Visual Studio 17 2022" -A x64
cmake --build Code/cpp/build --config Release
.\Code\cpp\build\Release\facestitche_demo.exe --root .
```

CMake 根据 `Code/cpp/CMakeLists.txt` 的相对位置找到 `Env/opencv`；如果 DLL 尚未存在，会在配置时自动解压 `opencv_world450.dll.zip`。构建时会将 `opencv_world450.dll` 复制到可执行文件旁，因此无需修改电脑上的 `PATH`。`--root` 指定包含 `Input/` 和 `Output/` 的目录；省略时使用当前工作目录。使用随仓库提供的数据，程序会输出 `wrote 121012 vertices and 237779 faces`。

如果使用自行安装的兼容 OpenCV，请在空构建目录下配置 `-DOPENCV_INCLUDE_DIR=...`、`-DOPENCV_LIBRARY_DIR=...` 和 `-DOPENCV_LIBRARY=...`。运行时还需让 Windows 能找到相应的 DLL。

## 目录与输入

```text
Code/cpp/demo.cpp       C++ 入口与本次采集的固定参数
Code/cpp/CMakeLists.txt  构建配置
Env/opencv/              随仓库提供的 OpenCV 4.5.0 头文件、导入库和压缩后的运行时 DLL
Input/                   同一次采集的五个输入文件
Output/                  参考输出；运行 Demo 时会重新生成
```

`Input/` 中必须有以下五个文件：

| 文件 | 格式和用途 |
| --- | --- |
| `rgb_face_roi_l.png` | 左侧彩色 ROI PNG，839 × 768 像素；OpenCV 读取为 BGR。 |
| `rgb_face_roi_r.png` | 右侧彩色 ROI PNG，839 × 768 像素；OpenCV 读取为 BGR。 |
| `point_lift.pcd` | 左侧原始有序 XYZ 点云；二进制 PCD，float32 `x y z`，1224 × 1024 点。文件名沿用原工程的 `lift` 拼写。 |
| `point_right.pcd` | 右侧原始有序 XYZ 点云，格式和尺寸同左侧。 |
| `mesh.ply` | 右侧点云坐标系下的重建小端二进制 PLY 网格。 |

左右相机各自的 `K/Kc/R/T`、SAC 与 ICP 矩阵、颜色矩阵 `M`、RGB 宽度和带 padding 的 ROI 偏移已写入 `demo.cpp` 的 `recordedParameters()`。右侧原始深度 ROI 固定为 `u=563, v=123, 宽=348, 高=418`。程序不需要也不会读取 `Input/终端输出.txt`。换一批采集数据时，必须同时更换全部五个输入文件和这些对应的固定数值。

## 预期输出

| `Output/` 下的文件 | 内容 |
| --- | --- |
| `corrected_left.png` | 校正后的左侧 ROI。 |
| `stitched_raw.png`、`stitched_matrix.png`、`stitched_reinhard.png` | 原始、矩阵校正及 Reinhard 校正后的拼接图。 |
| `colorfacemesh.obj`、`colorfacemesh.mtl` | 使用 `v x y z r g b` 顶点颜色、带法向量的网格；不使用纹理贴图。 |
| `depth_consistency.csv` | 每个候选点的通过/剔除原因、投影像素、最近的右侧实测点、三维距离和深度差。 |
| `metrics.json` | 固定参数、深度筛选数量、投影数量和颜色差统计。 |

程序在写入新结果前会清除 `Output/` 里原有的普通文件。运行前请移走需要保留的结果。建议先查看 `metrics.json` 的 `depth_consistency` 字段及 `depth_consistency.csv` 中的 `KEEP` 行。颜色差统计只描述当前样本，不能直接作为不同采集数据之间的精度指标。
