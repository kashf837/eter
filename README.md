# ETER NURBS 渲染器

CUDA + Vulkan 实现的 NURBS 模型实时渲染器（本科毕业设计项目），基于 SIGGRAPH 2023 论文：

> **ETER: Elastic Tessellation for Real-Time Pixel-Accurate Rendering of Large-Scale NURBS Models**
> Ruicheng Xiong, Yang Lu, Cong Chen, Jiaming Zhu, Yajun Zeng, Ligang Liu — ACM Transactions on Graphics 42(4), 2023. DOI: [10.1145/3592419](https://doi.org/10.1145/3592419)（[项目主页](http://staff.ustc.edu.cn/~lgliu/Projects/2023_ETER/default.html)）

## 特性

- **NURBS → 有理 Bézier 分解**：节点插入算法，与文本解析路径数值逐位一致
- **视点相关自适应细分**：每个 patch 按屏幕投影计算细分层级，相机移动时局部重建网格
- **场景级批量求值**：全部曲面一次 CUDA launch（参考场景约 5.7 万 patch / 370 万顶点）
- **.surfb 打包二进制**：3506 曲面场景加载 11.4 s → 35 ms
- **STP/STEP 直接加载**：内置纯标准库 Python 解析器，自动转换
- 调试可视化：Checkerboard 细分密度图、patch 边界显示

## 依赖

| 依赖 | 说明 |
|---|---|
| Windows 10/11 | |
| Visual Studio 2022 + CMake ≥ 3.22 + Ninja | C++20 / CUDA 20 |
| CUDA | 13.0，`sm_89`（RTX 40 系列；其他 GPU 修改 `CMakeLists.txt` 的 `CMAKE_CUDA_ARCHITECTURES`） |
| Vulkan SDK | 仅修改 shader 后重新编译需要 `glslc`（预编译 SPV 已入库） |
| Python 3（标准库即可） | 仅 STP/STEP 转换需要 |

## 构建

本仓库**不包含**第三方库 nvpro_core2（gitignore），需自行获取并打上两处修复补丁：

```bat
:: 1. 获取 nvpro_core2（必须钉住此 commit，补丁按它编写）
git clone https://github.com/nvpro-samples/nvpro_core2.git external/nvpro_core2
git -C external/nvpro_core2 checkout 0097241

:: 2. 打补丁（相机两处 bug 修复，说明见 patches/ 内各文件头部）
git apply patches/nvpro_core2-effectiveup-angle-only.patch patches/nvpro_core2-elemcamera-inverted-y.patch

:: 3. 配置 + 编译
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDACXX=<CUDA 安装目录>\bin\nvcc.exe
cmake --build build
```

生成 `_bin\Release\eter.exe`。仓库内 `configure.bat` / `compile_shaders.bat` 含本机绝对路径，仅作参考（新机器需按自己的安装路径修改）。

## 运行

在**项目根目录**启动（shader 按相对路径加载）：

```bat
_bin\Release\eter.exe
```

模型加载：`File → Open` 选择 `.stp` / `.step`（自动转换）/ `.surf` / `.surfb` / `.surf` 目录；或使用 `--load <路径>` 启动参数直接加载。

| 操作 | 效果 |
|---|---|
| 左键拖 | 轨道旋转 |
| 中键拖 | 平移 |
| 右键拖 / 滚轮 | 缩放（贴近模型时自适应细分层级升高） |
| Shift + 滚轮 | 视野 FOV |
| Ctrl + 左键拖 | 平移 |
| W/S/A/D、方向键 | 前后移动 / 平移 |

界面开关：`Checkerboard`（细分密度棋盘格）、`Show Boundaries`（patch 边界）。

## 数据

`data/` 被 .gitignore 忽略——STEP 模型文件版权归原作者，不入仓库。模型放任意位置后用 `File → Open` 加载即可。

## 目录结构

| 目录 | 内容 |
|---|---|
| `src/` | C++ 主程序 + CUDA 内核（节点插入 / Bézier 求值 / 自适应细分） |
| `shaders/` | Vulkan shader 源码 + 预编译 SPV |
| `scripts/` | `step2surf.py`（STP → .surf/.surfb 转换）、数据生成、批量求值 |
| `patches/` | nvpro_core2 vendor 修复补丁（**构建必打**） |
| `tests/` | `test_packed`：.surfb 读取器控制台回归测试（`cmake --build build --target test_packed`） |
| `cmake/` | `FindNvproCore2.cmake` |

## 性能（参考）

RTX 4070 / `Army Guy.step`（3506 曲面、57484 个 Bézier patch）：

| 阶段 | 耗时 |
|---|---|
| 模型加载（.surfb） | ~35 ms |
| 自适应求值（一次 launch） | ~65–90 ms |
| 上传 + GPU 拷贝 | ~35 ms |
