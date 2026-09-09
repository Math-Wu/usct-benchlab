# 数据适配：独立标定、二阶 Eikonal 与有限频率匹配

## 结论与范围

**本轮改善了数值可信度和观测契约，但没有通过数据预处理解决全部乳腺失配，重建图像也没有普遍改善。**
不能把 k-Wave 压力按 GT “修正”为理想直线投影，再当作同一测量下的公平比较。

- 独立均匀介质中，xcorr 时间标尺准确，没有发现能解释乳腺误差的大时间偏移。
- 宽 Gaussian 中几何模型与波形延迟很接近；同一扰动缩小到 2 mm 后差异明显增大。
- 二阶 FMM 减少了离散误差，但没有把有限带宽观测变成几何最早到时。
- Full-Green 经同一个有限频率泛函后，与对应 k-Wave 观测接近；固定水背景的一阶 Born 在较强扰动上仍失败。
- 最终用冻结源码重跑了 10 个完整重建条件，全部保存了组织区 RMSE/PSNR/SSIM、留出残差、停止原因和预览。

分支 `work/physics-agent-validation`，起点 `71e26ec`，main 不变。复用 NBP D 高/低频、
OpenBreastUS HET 低频：**3 次采集、2 个解剖样本**，不是全部八类验收。原始采集不覆盖。
不读乳腺 GT 选择拾取器、校正延迟、限制反演 ROI 或停止；GT 只用于显式事后前向归因和图像评价。

## 1. 独立 k-Wave 标定

在 A100 的 GPU 0/1/2 上分别使用三种既有采集设置。保持网格、实际采样时刻、源脉冲、
元素位置、PML、常密度、无衰减条件。每组 2 TX×64 RX，原始编号 TX=0/32、RX=0/2/.../126。

1. 均匀介质：1500 m/s 水，1460/1540 m/s 用于选择规则，1480/1520 m/s 为保留验证。
2. 宽 Gaussian：标准差为阵列半径的 0.24 倍，峰值为上述声速，背景 1500 m/s。
3. 窄 Gaussian：标准差固定为 2 mm，其余不变。

共 9 组、45 个介质条件、**90 次实际 k-Wave 源传播**，包含各组水参考。
不是数字平移代替物理仿真。初次 MATLAB 调用的整数声速类型错误已改为显式 double；
初次失败目录保留，不计入成功数。源、输入和结果 SHA、CFL、PPW、通道编号均记录。
再生成的水必须重放父采集水波形，脚本以相对残差 `<=1e-5` 检查，不只比较 shape。

固定候选为 xcorr、AIC、包络阈值 0.02/0.05/0.1/0.2。仅用均匀训练介质选择，要求有效比例至少 95%。
三组均选择 **xcorr**。不拟合全局或逐通道延迟补偿；不根据 Gaussian 或乳腺结果重新选择。

### 均匀介质

独立解析真值为

$$
\Delta t_{sr}=\|r-s\|\left(\frac{1}{c}-\frac{1}{1500}\right).
$$

两个保留声速的 RMS 误差范围，单位 ns：

| 采集设置 | xcorr | 包络 0.1 | AIC |
|---|---:|---:|---:|
| NBP D / 500 kHz | 0.00343–0.00358 | 0.0979–0.1177 | 44.60–47.05 |
| NBP D / 200 kHz | 0.01915–0.02004 | 0.3767–0.3966 | 93.25–124.94 |
| OpenBreastUS / 200 kHz | 0.00810–0.00856 | 0.3939–0.4100 | 94.59–126.94 |

这验证了均匀传播中的数据链路和延迟标尺，不证明复杂介质的相关峰等于几何最早到时。
AIC 没有获得替换默认拾取器的证据。

### 空间尺度

宽 Gaussian 中，xcorr 相对二阶 Eikonal 的 RMS 差：

| 采集 | 峰值 1460/1540 m/s | 峰值 1480/1520 m/s |
|---|---:|---:|
| NBP D 高频 | 0.542 / 0.797 ns | 0.267 / 0.399 ns |
| NBP D 低频 | 2.956 / 3.304 ns | 1.447 / 1.665 ns |
| OpenBreastUS 低频 | 1.255 / 2.856 ns | 0.624 / 1.427 ns |

标准差缩小到 2 mm，同样 ±40 m/s 峰值：

| 采集 | xcorr/Eikonal 差，慢/快 | 相对几何延迟信号范数，慢/快 |
|---|---:|---:|
| NBP D 高频 | 9.986 / 9.511 ns | 63.5% / 53.0% |
| NBP D 低频 | 13.644 / 14.711 ns | 85.1% / 79.3% |
| OpenBreastUS 低频 | 10.479 / 12.708 ns | 93.7% / 95.3% |

这是相对 **Eikonal 数值参考**的差异，不是纯衍射误差的严格比例分解，特别是低频 D 仍有离散误差。
但与宽 Gaussian 对照，空间尺度依赖已很清楚。500/200 kHz 水中波长约为 3/7.5 mm，
不能仅凭 256×256 的输出像素数，期待亚波长纹理可靠恢复。

## 2. 二阶 FMM 与离散伴随

新增 `eikonal_order: 2`，类型整数，只允许 1/2；默认仍为 1，保留原有行为。
两个上游节点已接受且时间单调时使用

$$
D^-T\approx\frac{3T-4T_1+T_2}{2h},\qquad
\widetilde T_1=\frac{4T_1-T_2}{3},\quad\widetilde h=\frac{2h}{3}.
$$

否则回退一阶。依据 [Rickett–Fomel 的二阶 FMM 推导](https://sepwww.stanford.edu/sep/sergey/sepsergey/fmsec/paper_html/node2.html)，
并核对 [scikit-fmm 的迎风处理](https://github.com/scikit-fmm/scikit-fmm/blob/master/skfmm/travel_time_marcher.cpp)。
不是复制上游 C++。二阶远邻导数可能为负，因此每节点保存至多四个父节点，切线与伴随均保留这些权重。

通过了一阶回归、二阶点积伴随、中心有限差分、因果依赖、均匀点源和线性声速梯度解析解测试。
**局部二阶不等于全局二阶**：点源奇点、界面和接受模板切换仍可能限制收敛阶。

同一中心插值场、同一物理源/接收位置加密，延迟变化 RMS，单位 ns：

| 采集 | 一阶首次加密 | 二阶首次加密 | 二阶再加密 |
|---|---:|---:|---:|
| NBP D 高频，1024→2047 | 30.284 | 约 5.1 | 未使用未完整验收的 4093 结果 |
| NBP D 低频，256→511→1021 | 27.274 | 32.602 | 10.687 |
| OpenBreastUS，1024→2047→4093 | 93.840 | 10.719 | 3.381 |

二阶不保证每个粗网格更好，低频 D 是反例。OpenBreastUS 离散不确定性降低后，
与原 xcorr 的差仍约 **767 ns**，远大于最后的网格变化。
原先一阶较小的观测残差包含误差抵消；不能据此选择较不准确的前向。

高频 D 的一次 4093 审计被 SSH 中断，保留部分目录，仅使用已完成的级别。
之后长任务采用独立日志，不把终端退出码当作远端数值验收。

## 3. 同一有限频率泛函与模型匹配

参考 [Korta Martiartu 等的有限频率旅行时层析](https://arxiv.org/abs/1908.03302)，
新增小扰动相关延迟导数与现有 Born 算子的组合。**没有新增注册算法，没有替换直线 Siddon 算子。**

正号 Fourier 变换下纯延迟比值为 $\exp(i\omega\tau)$。水背景处相关峰的导数为

$$
\delta\tau_{sr}=
\frac{\sum_k a_k\omega_k |P^0_{ksr}|^2\operatorname{Im}(\delta P_{ksr}/P^0_{ksr})}
{\sum_k a_k\omega_k^2|P^0_{ksr}|^2}.
$$

$a_k$ 是频率积分权重，至少三个足够幅度的频率参与；缺失通道返回 NaN。
`CorrelationTravelTimeJacobian` 输入平方慢度扰动，输出秒，具有完整复数到实数链式伴随。
验证覆盖延迟符号、小扰动展开、点积、复增益不变性、坏频率和通道局部性。
频率必须在特征构造前遵守 hold-out；本次前向归因不是训练/验证重建，不声称留出测试成绩。

4 TX×64 RX、9 个固定频率，对**同一泛函输出**计算 ns RMS：

| 前向预测 | NBP D 高频 | OpenBreastUS HET 低频 |
|---|---:|---:|
| 直线投影，作为不匹配参照 | 95.236 | 1715.131 |
| 水背景一阶 Born + 匹配泛函 | 69.629 | 1743.410 |
| 非线性 Full-Green + 同一泛函 | 1.082 | 3.967 |

对应的完整复数散射比值相对残差分别为 **1.929% / 0.919%**，并非只看一个实值泛函投影掩盖压力误差。
这里 GT 只作为事后前向输入，未拟合乳腺源、偏移或比例。水比值抵消固定乘性源/接收响应，不消除一般模型误差。

重要失败也需保留：固定水背景的一阶 Born 无法覆盖较强扰动 OpenBreastUS。
线性化泛函本身不是大延迟的通用到时估计器，可能失去相位周期信息。
不能把此表当作几何 ToF 已修复或新算法已重建成功。若采用有限频率旅行时，
需要非线性重线性化和对应 Jacobian/伴随，而不是仅把新 sinogram 塞给旧直线矩阵。

## 4. 冻结重建结果

最终重新提取 64×64 xcorr，反演 256×256；每个采集五个条件共享同一个 case、掩码、权重、
1500 m/s 初值和 1300–1700 m/s 边界。不给 GT ROI，组织区指标只在反演后计算。
沿用既有冻结参数：CGLS Laplacian lambda=0.02，其他方法的平滑设置未按本轮 GT 调整。

接收器留出 12.5%，seed=42，排除互易训练泄漏；共同 600 s 时间上限，
迭代上限 120/200/100/20，目标残差/停滞/验证/预算 OR 停止，保留最佳验证 checkpoint。
没有伪造 noise norm，没有 GT 停止。CPU 求解器使用单线程 BLAS/OMP，GPU 用于 k-Wave 标定。
不同算法的迭代和算子调用不等于相同 FLOPs；这些是**冻结参数的受控比较，不是充分调参的最终排行榜**。

| 样本 | 方法 | 组织 RMSE | PSNR | SSIM | 留出相对残差 | 停止 / 选中迭代 |
|---|---|---:|---:|---:|---:|---|
| NBP D 高频 | CGLS | 29.245 | 12.260 | 0.0971 | 0.1709 | 上限 / 120 |
| NBP D 高频 | SIRT | 28.806 | 12.391 | 0.1079 | 0.1809 | 上限 / 200 |
| NBP D 高频 | SART | 28.746 | 12.410 | 0.1088 | 0.1739 | 上限 / 100 |
| NBP D 高频 | Bent 一阶 | 29.134 | 12.293 | 0.1019 | 0.1790 | 验证停滞 / 13 |
| NBP D 高频 | Bent 二阶 | 29.623 | 12.149 | 0.0957 | 0.1575 | 上限 / 19 |
| OpenBreastUS | CGLS | 37.741 | 10.156 | 0.1070 | 0.2040 | 验证停滞 / 48 |
| OpenBreastUS | SIRT | 35.005 | 10.809 | 0.1795 | 0.1925 | 上限 / 200 |
| OpenBreastUS | SART | 35.085 | 10.789 | 0.1801 | 0.1915 | 上限 / 100 |
| OpenBreastUS | Bent 一阶 | 35.469 | 10.695 | 0.1750 | 0.1896 | 验证停滞 / 2 |
| OpenBreastUS | Bent 二阶 | 37.266 | 10.266 | 0.1229 | 0.2016 | 验证停滞 / 5 |

RMSE 单位 m/s，PSNR 单位 dB。留出用于停止和选 checkpoint，是 validation，不是 untouched final test。
Bent 二阶高频完成 20 次外迭代，选择第 19 次；其他完整计数在 JSON 中。
循环时间：NBP D 的 CGLS/SIRT/SART/Bent1/Bent2 为 0.55/0.72/1.91/198.73/329.92 s；
OpenBreastUS 为 0.25/0.71/1.92/92.70/240.62 s，不含全部 I/O/绘图开销。

图像没有普遍改善；“更准确的 Eikonal”不等于“更吻合有限频率 xcorr”。
初轮一项 Bent 运行期间源码校验值变化，因此最终冻结源码重跑全部 10 项；
最终每项 `source_changed_during_run=false`，且每个采集内 split SHA 一致。

## 5. 代码与复现

新增 `CorrelationDelayDerivative`、`CorrelationTravelTimeJacobian` 位于
`src/usctbench/operators/forward/correlation_delay.py`；不属于通用大扰动到时估计器。
原 `EikonalForward(..., spatial_order=2)` 与 Bent YAML 的 `parameters.eikonal_order: 2` 可启用二阶。
`linearize(...).jacobian.forward/adjoint` 使用相应离散导数。

在仓库根目录运行，`ACQUISITION` 为既有压力/仿真输入/manifest 目录，`CASE` 为对应 64 通道 case：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/calibrate_arrival.py --acquisition "$ACQUISITION" \
  --out "$USCT_WORKSPACE/runs/calibration_fresh" --device 0 \
  --matlab "$MATLAB_BIN" --kwave-path "$KWAVE_PATH"
python scripts/calibrate_arrival.py --acquisition "$ACQUISITION" \
  --out "$USCT_WORKSPACE/runs/narrow_fresh" --medium-shape gaussian \
  --gaussian-width-m 0.002 --device 1 --matlab "$MATLAB_BIN" --kwave-path "$KWAVE_PATH"
python scripts/audit_tof_adaptation.py --acquisition "$ACQUISITION" --case "$CASE" \
  --out "$USCT_WORKSPACE/runs/eikonal_fresh" --tx-count 8 --refine 4 --eikonal-order 2
python scripts/audit_finite_frequency.py --acquisition "$ACQUISITION" --case "$CASE" \
  --out "$USCT_WORKSPACE/runs/finite_fresh" --tx-count 4 --full-green
python scripts/compare_adapted_tof.py --acquisition "$ACQUISITION" --case "$CASE" \
  --handoff "$HANDOFF_ROOT" --calibration "$USCT_WORKSPACE/runs/calibration_fresh/summary.json" \
  --out "$USCT_WORKSPACE/runs/comparison_fresh"
```

输出必须在仓库外。`--resume` 核对特征、权重和配置并复用成功输出；不覆盖旧重建。
初始校准报告另存 `summary.initial.json`，后续水重放检查补充到 `summary.json`，可核对旧 SHA。
完整报告数组保留在工作区 `runs/arrival_calibration_20260909/`，不进入 Git。
最终重建为 `comparison_final_high_d/`、`comparison_final_low_ob/`；复数检验为 `finite_verified_*/`。
`render_adaptation_report.py` 从这些实际输出生成汇总图和 GT 对照，没有旋转或后处理增强。

## 6. 验证与剩余工作

本地和 A100 均为 **266 passed, 2 skipped**；跳过外部 MATLAB FWI 测试，不代表生产 FWI 已重跑。
Black、Ruff、compileall、CLI help/list 已通过；本地和 A100 synthetic smoke/release audit 均通过。
两张本地汇总 PNG 已渲染并检查文字、灰度范围和图像方向。本轮不提交原始数据。

已改善的是独立时间标定、数值离散精度、观测与前向/伴随的显式组合及公平对照记录。
尚未解决的是复杂乳腺有限带宽观测到理想几何首达时的可靠、无 GT 转换，也未证明全部算法达到理论极限。

后续应分开两条路线：保留严格几何模型的真实适用尺度；或明确扩展为非线性有限频率旅行时。
两者都从相同原始压力出发，保持水参考、数据划分和不读 GT 的停止规则。
不能按乳腺 GT 调延迟、筛掩码或挑最锐利的图像，把前向模型差异伪装成“数据修好了”。
