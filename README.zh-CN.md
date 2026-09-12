# usct-benchlab

[English README](README.md)

`usct-benchlab` 专注于 **二维超声声速重建** 的研究级数值基准测试与运行时集成。
它提供统一输入/输出、数据准备、经典与原生物理模型算法、FWI 适配器、指标和
可复现的评估报告。当前不提供衰减重建 API，也不声明临床有效性。

## USCT 是什么？

**USCT 本质上是一个 PDE 约束反问题。** 换句话说，换能器发射声波，声压场
在人体组织或仿体中按照声学波动方程传播，接收阵列记录时间信号；反问题的
目标是从这些接收信号中恢复介质的空间声学参数。

本仓库的重建目标是声速图 $c(x)$。数据使用统一的 `USCTCase` 格式，
算法输出使用 `ReconstructionResult`。

## 数学形式

USCT 应被理解为 PDE 驱动的反问题，而不是普通的图像重建任务。声源换能器
激发声压场，声场在未知介质中传播，接收器测量这些传播后的信号，再由这些
测量反推介质参数。

$$
\frac{1}{c(x)^2}\partial_{tt}p_s(t,x)-\Delta p_s(t,x)=q_s(t,x).
$$

在频域中，对应的 Helmholtz 形式常写为

$$
\left(\Delta+\omega^2m(x)\right)\hat p_s(\omega,x)=-\hat q_s(\omega,x).
$$

其中 $p_s$ 是声源 $s$ 对应的声压，$q_s$ 是发射源，$c(x)$ 是声速，$m(x)$
是平方慢度：

$$
m(x)=\frac{1}{c(x)^2}.
$$

本仓库中的多数声速重建方法估计的是声速图 $c(x)$，或者慢度图

$$
u(x)=\frac{1}{c(x)}.
$$

接收器 $r$ 通过测量算子观测传播后的声压场：

$$
d_{sr}(t)=\mathcal M_r p_s(t,\cdot)+\eta_{sr}(t).
$$

不同算法的核心区别在于保留了多少波动物理。FWI 在优化中保留声学 PDE 或
Helmholtz 求解，并匹配波形或复数频域压力。travel-time baseline 会先把数据
降维为到时特征，再反演射线或 eikonal 近似；它们更快、更稳定，但舍弃了相位、
幅度、衍射以及大量有限频物理。

直射线 travel-time 模型使用参考声速 $c_0$ 和固定路径 $\gamma_{sr}$：

$$
\Delta t_{sr}\approx\int_{\gamma_{sr}}\delta u(x)d\ell.
$$

慢度扰动为

$$
\delta u(x)=\frac{1}{c(x)}-\frac{1}{c_0}.
$$

像素离散化后得到

$$
A\delta u \approx b.
$$

CGLS、SIRT 和 SART 求解的都是类似下面的代数射线系统：

$$
\min_{\delta u}\|W(A\delta u-b)\|_2^2+\lambda^2\|L\delta u\|_2^2.
$$

上述二次目标对应 CGLS，其中 $W_{ii}=\sqrt{w_i}$。当前 SIRT 的行归一化
实际引入 $w_i/\sum_jA_{ij}$ 权重；固定松弛系数的子集 SART 在不一致数据上还可能循环。
共享前向模型不等于严格最小化同一个目标，可选图像平滑也不保证全局损失单调下降。
详见[反演器数值审计](docs/validation/2026-09-09_inverse_solver_audit_CN.md)。

Bent-ray 方法保留高频 travel-time 模型，路径会随当前介质变化：

$$
|\nabla T_s(x)|=u(x).
$$

接收器 travel time 近似为

$$
t_{sr}\approx T_s(r).
$$

理想化的非线性 travel-time 目标可以写为

$$
\min_c\sum_{s,r}\left|t_{sr}^{\mathrm{obs}}-T_s(r;c)\right|^2+\lambda R(c).
$$

FWI 直接使用波形或频域压力数据：

$$
\min_c
\frac{1}{2}\sum_{\omega,s,r}
\left|
\hat p_s(\omega,r;c)-\hat p_{sr}^{\mathrm{obs}}(\omega)
\right|^2
+\lambda R(c).
$$

其中 $\hat p_s(\omega,r;c)$ 不是任意图像算子，而是候选声速下由声学 PDE 或
Helmholtz solver 预测出来的压力。

| 方法 | 建模假设 | 优化目标 | 适用场景 |
| --- | --- | --- | --- |
| CGLS | 参考介质中的固定直射线；到时差在线性慢度扰动上近似。 | 对 $A\delta u\approx b$ 做加权正则化最小二乘 Krylov 求解。 | 快速、可复现的声速 baseline 和回归测试。 |
| SIRT | 与 CGLS 相同的直射线代数模型，但用同步归一化残差反投影更新。 | 通过 relaxation 和 smoothing 迭代降低 $A\delta u\approx b$ 的加权残差。 | 更重视稳定性的迭代 baseline。 |
| SART | 相同直射线模型，用发射器或射线子集做有序更新。 | 子集 row-action 更新。 | 早期收敛更快，但对排序和 relaxation 更敏感。 |
| Bent-ray | 高频 travel time 满足 eikonal 近似；射线路径随声速或慢度变化。 | 基于 $T_s(r;c)$ 的正则化非线性 travel-time mismatch。 | Fast-marching 折射走时反演，不描述衍射和多次到达。 |
| FWI | 完整声学波或 Helmholtz 传播；数据是波形或复数压力。 | 对声源、接收器和频率上的 PDE-constrained waveform mismatch 做优化。 | 有外部 k-Wave/FWI artifact 或外部 FWI 命令时的高保真汇报。 |

`bent_ray_gn` 现在使用真正的 Eikonal/fast-marching 非线性前向及离散伴随；
`rwave_adapter` 使用复压力与随迭代更新的有限频率 Born 散射算子；配置默认通过
自由空间体积分方程求解完整 Green 背景，Eikonal/WKB 近似保留为显式选项。
两者不再依赖直线投影器。这些数值实现不声称完整复现
上游 r-Wave 的所有功能，也不保证图像质量一定优于直线方法。FWI 路线作为高保真外部
k-Wave/FWI 结果的适配器。更详细的数学说明见
[docs/math_formulation.md](docs/math_formulation.md)。

## 支持的算法

| 算法 | 注册命令 | 数学模型 | 输入要求 | 典型用途 | 配置文件 |
| --- | --- | --- | --- | --- | --- |
| CGLS | `straight_cgls` | 直射线加权最小二乘 | 带环形几何和 travel-time 测量的 `USCTCase` | 快速声速 baseline | `configs/algorithms/cgls.yaml` |
| SIRT | `straight_sirt` | 同步迭代射线层析 | 带环形几何和 travel-time 测量的 `USCTCase` | 稳健的迭代声速 baseline | `configs/algorithms/sirt.yaml` |
| SART | `straight_sart` | 有序/子集代数射线更新 | 带环形几何和 travel-time 测量的 `USCTCase` | 有序更新直射线 baseline | `configs/algorithms/sart.yaml` |
| Bent-ray | `bent_ray_gn` | Eikonal / fast marching 非线性到时反演 | 首波到时或经过校准的到时差 | 折射校正 | `configs/algorithms/bent_ray.yaml` |
| rWave adapter | `rwave_adapter` | 更新背景的有限频率 Ray-Born 散射 | 复压力以及源校准或独立水参考 | 散射敏感反演 | `configs/algorithms/rwave.yaml` |
| WUST FWI | `fwi_wust` | 频域 PDE 全波反演 | 复数总压力、显式约定及掩码 | MATLAB/CUDA 重建 | `configs/algorithms/fwi_wust.yaml` |
| Tiny FWI sanity | `fwi_tiny` | 小型 waveform-inversion sanity model | 小尺寸合成声速样本 | 本地 FWI 管线 sanity check | `configs/algorithms/fwi_tiny.yaml` |

更多算法说明见 [docs/algorithms.md](docs/algorithms.md)。

## 安装

使用 conda：

```bash
conda create -n usctbench python=3.10 -y
conda activate usctbench
pip install -e ".[dev,viz]"
```

或使用 pip：

```bash
pip install -r requirements.txt
pip install -e .
```

检查安装：

```bash
usct --help
usct list-algorithms
pytest -q
```

如果只想快速跑通一个端到端示例，并且把生成文件都写到 `/tmp`，可以运行：

```bash
bash examples/synthetic_quickstart.sh
```

## 环境变量和工作区布局

建议用环境变量管理数据和输出，避免把数据、运行结果或外部工程提交到 Git：

```bash
export USCT_WORKSPACE=/path/to/usct-benchlab
export USCT_DATA_ROOT=$USCT_WORKSPACE/data/openbreastus
export USCT_RUN_ROOT=$USCT_WORKSPACE/runs/usctbench_runs
export USCT_NBP_ZIP_PATH=/path/to/NBPslices2D.zip
```

推荐工作区结构：

```text
<workspace>/
  code/          # 本仓库
  data/          # 本地数据集和转换后的 case
  runs/          # benchmark 输出
  external/      # 可选外部工程
  checkpoints/   # 本地权重或 checkpoint
```

`scripts/setup_workspace.sh` 可以创建这套目录和仓库内的轻量 symlink；它不会
把数据集复制进 Git。

## 准备数据

合成 demo：

```bash
usct data make-synthetic-smoke \
  --out "$USCT_WORKSPACE/data/synthetic_demo" \
  --shape 48 \
  --n-transducers 48
```

OpenBreastUS：

```bash
usct data inspect-openbreastus \
  --root "$USCT_DATA_ROOT" \
  --out "$USCT_RUN_ROOT/openbreastus_index.json"

usct data make-quality \
  --root "$USCT_DATA_ROOT" \
  --out "$USCT_WORKSPACE/data/openbreastus_demo" \
  --cases-per-density 1 \
  --converted-shape 256 \
  --n-transducers 128
```

NBPslice2D：

```bash
usct data inspect-nbpslice2d \
  --zip "$USCT_NBP_ZIP_PATH" \
  --out "$USCT_RUN_ROOT/nbpslice2d_index.json"

usct data make-nbp-quality \
  --zip "$USCT_NBP_ZIP_PATH" \
  --out "$USCT_WORKSPACE/data/nbpslice2d_demo" \
  --cases-per-type 1 \
  --converted-shape 256 \
  --n-transducers 128
```

完整流程见 [docs/usage.md](docs/usage.md) 和
[docs/datasets.md](docs/datasets.md)。

## 运行单个算法

CGLS：

```bash
usct run straight_cgls \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/cgls.yaml \
  --out runs/single_cgls
```

SIRT：

```bash
usct run straight_sirt \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/sirt.yaml \
  --out runs/single_sirt
```

SART：

```bash
usct run straight_sart \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/sart.yaml \
  --out runs/single_sart
```

Bent-ray：

```bash
usct run bent_ray_gn \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/bent_ray.yaml \
  --out runs/single_bent_ray
```

rWave adapter：

```bash
usct run rwave_adapter \
  --case "$USCT_WORKSPACE/data/physics/example/pressure_case.h5" \
  --config configs/algorithms/rwave.yaml \
  --out runs/single_rwave
```

不能把由声速图投影得到的 ToF 当作 rWave 的复压力输入。压力生成和验证流程见
[physics validation](docs/physics_validation.md)，算子按前向/伴随分组的接口见
[operator contracts](docs/operator_contract.md)。无真值时使用独立接收点/频率留出；
原生循环按残差、停滞、时间/算子调用预算等 OR 条件在线停止，并记录停止原因。
默认外部 FWI 结果导入不能控制已经结束的 MATLAB 迭代，不会伪造其停止原因。

WUST FWI：

```bash
export USCT_WUST_ROOT=/path/to/approved/WaveformInversionUST
usct run fwi_wust \
  --case /path/to/frequency_case.h5 \
  --config configs/algorithms/fwi_wust.yaml \
  --out runs/single_fwi
```

该入口使用已有复频域总压力数据，不接受旅行时 demo 作为波场。生产后端为 MATLAB/CUDA，CPU 仅供参考验证。每个频率计划项对应一次完整更新；计划完成不等于收敛。配置中的数值只是使用示例，需明确初始化、声速范围及 PML。详见 [FWI 接入说明](docs/fwi.md)。

## 运行 benchmark

demo benchmark 会读取下面这些可选 case glob：

```bash
export USCT_SYNTHETIC_CASE_GLOB="$USCT_WORKSPACE/data/synthetic_demo/cases/*.h5"
export USCT_NBP_CASE_GLOB="$USCT_WORKSPACE/data/nbpslice2d_demo/cases/*.h5"
export USCT_OPENBREASTUS_CASE_GLOB="$USCT_WORKSPACE/data/openbreastus_demo/cases/*.h5"
```

运行 benchmark：

```bash
usct bench --suite configs/benchmarks/synthetic_demo.yaml
usct bench --suite configs/benchmarks/nbpslice2d_demo.yaml
usct bench --suite configs/benchmarks/openbreastus_demo.yaml
usct bench --suite configs/benchmarks/fwi_wust_demo.yaml
```

## 输出文件

单算法运行会写出：

```text
runs/single_cgls/synthetic_circular_sos/result.h5
runs/single_cgls/synthetic_circular_sos/metrics.json
runs/single_cgls/synthetic_circular_sos/metadata.yaml
runs/single_cgls/synthetic_circular_sos/preview.png
```

benchmark 会写出：

```text
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/straight_cgls/synthetic_circular_sos/result.h5
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/straight_cgls/synthetic_circular_sos/metrics.json
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/straight_cgls/synthetic_circular_sos/metadata.yaml
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/straight_cgls/synthetic_circular_sos/preview.png
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/benchmark_summary.csv
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/benchmark_report.md
```

`metrics.json` 保存每个 case 的图像指标和数据一致性指标；
`metadata.yaml` 记录算法、配置路径、case id、运行时间、状态和测量来源。

新 CLI/benchmark 运行以**去水背景的组织区 RMSE、PSNR、SSIM**作为主要图像指标，
同时单独保留全图指标（`full_image_*`）和水背景 RMSE。GT 掩码仅用于反演结束后的
评价，不参与初始化、更新或停止；没有 GT 时图像指标不可用，仍可报告测量/留出残差。
历史示例图保留原有指标定义，不能直接混用。详见[评价规则](docs/agent_evaluation.md)。

## 示例结果

OpenBreastUS 四类样本对比：

下列两图属于此前主线的历史示例，不能代表本分支新 Eikonal / Ray-Born 实现的验收结果。
当前独立波场测试见[八样本验证报告](docs/validation/2026-09-08_physics.md)。

![OpenBreastUS FWI and baseline comparison](docs/assets/openbreastus_readme_fwi_vs_surrogate.png)

NBPslice2D，2D Acoustic Numerical Breast Phantoms for USCT：

![NBPslice2D FWI and baseline comparison](docs/assets/nbpslice2d_readme_fwi_vs_surrogate.png)

不同算法使用的测量假设不同，结果解读应结合
[docs/algorithms.md](docs/algorithms.md) 和每个 case 的 metadata。

## 常见问题

- `algorithm not found`：运行 `usct list-algorithms`，检查注册命令名。
- 缺少 `.h5` 或 `.mat` 数据：确认数据转换命令已完成，并检查相关环境变量是否指向存在的路径。
- FWI 运行时不可用：设置 `USCT_WUST_ROOT` 指向批准的干净 WUST 版本，
  并检查 MATLAB/CUDA 环境，详见 [部署说明](docs/fwi.md)。
- 输出出现 NaN/Inf：查看 `failure_report.md`，检查 case 单位，并尝试降低迭代次数或 relaxation。
- glob 没有匹配到 case：打印展开后的 `USCT_*_CASE_GLOB`，确认转换后的 case 位于 `data/.../cases/`。
- 缺少 `matplotlib` 或 `scikit-image`：运行 `pip install -e ".[viz]"`。

## 开发

```bash
black src tests scripts
ruff check src tests scripts --fix
python -m compileall src tests
pytest -q
bash scripts/run_smoke.sh
python scripts/audit_release.py
```

更多 release 检查和仓库卫生规则见 [docs/development.md](docs/development.md)。

## 引用 / 数据集

如果在实验中使用了 OpenBreastUS、NBPslice2D、k-Wave 或
WaveformInversionUST，请引用相应数据集和外部工具。参考文献见
[docs/references.bib](docs/references.bib)。

## 许可证

本仓库使用 MIT License 发布。见 [LICENSE](LICENSE)。
