# k-Wave 与 surrogate ToF：数据适配核对与剩余失配

## 结论

**本轮没有证据支持“这四种算法已经达到极限”，也没有把更换拾取器包装成已解决失配。**
已排查输入介质、图像轴、网格原点、通道顺序、水参考和旧 surrogate 投影的一致性；
补上可复现的实际仿真介质审计、GT-free 波形平移残差、Modified AIC 候选与观测量契约。
有限带宽特征与几何 ToF 的差异、Eikonal 离散误差仍需继续分离。

当前仍保留 xcorr 为提取工具默认值；高频 D 的历史 envelope 文件不变。
没有替换原始压力、修改已有 case、根据 GT 校正延迟、重新调正则化或增加迭代。
不建议现在用这些结果宣称几何模型的最优性能，也不建议据此删除传统方法。

## 证据范围

- 基于 `work/physics-agent-validation`，先从 `78dea02` 快进到 `d231a59`。
  用户提供的 `9ffe73b` 已包含在其中；后续文档提交报告了 68 个成功重建条件。
  本轮**没有重跑或重新认证那 59/68 个重建条件**。
- 核对 main 的源代码版本：`c330216b18ff6e7c65b93dc8f4ad4f471ff69901`。
- A100 主机上执行三组观测/前向核对：NBP D 高频、同一 NBP D 低频、OpenBreastUS class 1 低频。
  这是 **3 次采集、2 个解剖样本**，不是三个独立患者，也不是全部八类样本验收。
- 在既有 64×64 通道子集中取固定 8 个 TX、全部 64 个 RX。
  原始 128 通道编号为 TX `[0,16,32,48,64,80,96,112]`，RX `[0,2,...,126]`。
  使用原始有效掩码；有效通道分别为 344、344、338。没有重新划分或拟合训练/验证数据。
- 读取 A100 保存的 `simulation_input.mat`、`pressure.mat`、`manifest.json`，
  不再仅依赖网页端数据包中的重采样假设。低频 raw pressure 实际存在，已重新读取。
- 两阶段执行：旧拾取代码下的身份检查；修复 float64 累加并加入 AIC 后的最终完整审计。
  本文表格使用后者，最终产物位于工作区 `runs/tof_adaptation_20260909/final/`。
  原始输入保持不变。每组 JSON 保存输入 SHA、所有包源码 SHA 和数值结果。

## main 和当前数据究竟不同在哪里

main 的 `data/conversion.py` 直接计算

$$
b_{\mathrm{sur}} = A_h(s_{\mathrm{GT}}-s_0),\qquad s=1/c.
$$

生成与反演采用相同的 Siddon 单元交长，不包含有限带宽、衍射、折射、脉冲形变或拾取误差。
这适合求解器闭环验证，但不能证明从原始压力提取的观测量已适配。

当前链路实际是

$$
b_K = \Phi(p_{\mathrm{object}},p_{\mathrm{water}}),
\qquad p=\mathcal F_{\mathrm{kWave}}(c).
$$

直线方法仍预测 $A_h(s-s_0)$；Bent 预测

$$
b_{E,h}(s)=T_h(s)-T_h(s_0).
$$

这里 $\Phi$ 是具体拾取器，不能把它与几何最早到时算子直接视为同一个观测量。
当前 Bent 已正确减去同一离散网格的水参考；没有发现重复减水或正负号颠倒。
减水只能抵消水中的离散偏差，不能消掉非均匀介质的离散误差。

| 项目 | main 的旧样本/代码 | 当前 64 通道诊断 |
|---|---|---|
| 观测生成 | GT 声速直接直线投影 | k-Wave 原始压力与独立水参考，经拾取器生成延迟 |
| 原始通道 | 所查父 case 为 128×128，16256 条非自发自收射线 | 64×64 后还应用几何排除和训练/验证隔离 |
| 反演网格 | 256×256 | 256×256；不能据此认为两者信息量相同 |
| ROI | NBP D 的父 case 有 30229 像素标签 ROI；默认 CGLS 开启 ROI 限制 | 不给求解器 GT ROI |
| Bent 含义 | `StraightRayProjector` 驱动的 surrogate baseline | 非线性离散 Eikonal/FMM |
| 指标口径 | 按原 ROI/全图计算；旧 SSIM 会填充无效 ROI 后算裁剪图 | 主指标为去水组织区，SSIM 要求完整有效窗口 |
| FWI 展示 | 渲染器还可优先读 `kwave_native_psnr/ssim` | 不能与不同掩码、不同范围的数字直接排名 |

因此，截图到当前结果的变化不只来自“surrogate 换成 k-Wave”：通道数量、支持域先验、
指标定义和 Bent 的数学实现也变了。后续比较必须显式固定这些条件。

## 身份与表示误差

三组的文件 SHA、压力导出的 GT、物理元素位置/顺序和格点中心均通过检查。
从图像网格重新生成仿真介质的最大绝对差分别为
`4.58e-10 / 3.04e-10 / 1.96e-11 m/s`，属于浮点算术量级。
未发现这三组中的轴旋转、半像素偏移或毫米/米错用。

两个父 surrogate case 可由当前 Siddon 重新复现：NBP D 延迟 RMS 差
`7.46e-14 ns`，OpenBreastUS 为 `5.30e-13 ns`。父 case 与当前图像 GT 数组相同。
这说明投影迁移没有破坏这两个旧观测的数值闭环，不等于复现了截图全部重建。

单位为 ns，数值均为固定通道上的 RMS：

| 采集 | 图像网格与实际仿真网格的直线表示差 | 图像网格直线预测 vs 保存特征 | 实际仿真网格直线预测 vs 保存特征 |
|---|---:|---:|---:|
| NBP D，500 kHz | 5.582 | 86.100 | 84.585 |
| NBP D，200 kHz | 11.846 | 125.134 | 123.843 |
| OpenBreastUS class 1，200 kHz | 11.022 | 457.277 | 455.966 |

**在这些通道上，输入表示差不是主要误差来源。** 这不是各误差能量可直接相减的比例分解。
两种预测都用相同的 snapped 元素位置，未用 GT 拟合偏移、比例或延迟校正。

## 有限带宽观测与几何模型

必须纠正前一份报告中的统称：**低频 D 和低频 OpenBreastUS 保存的是 xcorr，
高频 D 保存的才是 envelope**。三者原始压力均已重新提取验证。
float64 改动前，保存特征在本次固定通道上可完全复现；改动后低频 xcorr 的 RMS 差
仅 `0.000771 / 0.000973 ns`，envelope 不变。

不读 GT 的检查：给水参考施加已知 ±250 ns 数字平移，再测拾取误差；以及
在真实对象压力上拟合“平移后的水参考 × 非负标量增益”。后者只用于诊断，不将增益写回源校准。

| 采集 | xcorr/envelope 延迟差 RMS（ns） | xcorr 波形拟合残差中位数 | envelope 波形拟合残差中位数 |
|---|---:|---:|---:|
| NBP D 高频 | 64.065（343 个共同有效通道） | 15.54% | 21.05% |
| NBP D 低频 | 53.631 | 8.10% | 9.97% |
| OpenBreastUS 低频 | 407.985 | 16.58% | 22.95% |

xcorr 的数字平移 RMS 误差不超过 `0.014 ns`；envelope 不超过 `0.248 ns`。
这说明数字平移实现准确，但真实对象信号不是纯平移。
xcorr 更擅长解释整段直接脉冲，不代表更准确地测到了 Eikonal 最早到时。
高 SNR、互易性和高相关系数均不能单独证明几何到时正确。

### Modified AIC 候选的实际结果

按 [Javaherian 等，2020，附录 A](https://arxiv.org/html/2005.11204v2#A1) 的思路实现
包络定位小窗、两段方差 AIC、最小值附近加权时间。明确记录方差下限、最小分段和边界拒绝；
这不是逐行复制上游实现。本轮固定阈值 0.25、小窗为已记录的三周期脉冲长度，不用 GT 调整。

| 采集 | 精确介质 Eikonal vs xcorr（ns RMS） | vs envelope | vs AIC 候选 |
|---|---:|---:|---:|
| NBP D 高频 | 123.701 | 113.296 | 129.304 |
| NBP D 低频 | 95.244 | 83.810 | 505.658 |
| OpenBreastUS 低频 | 604.879 | 402.099 | 300.712 |

AIC 在原始水波形数字平移检查中的 RMS 误差约 `3–21 ns`，高于已有两种方法；
在低频 D 还产生明显偏移。因此**没有采用“某一例对 GT 更好”的结果来选它为默认**。
这些差值也不是拾取误差的真值，因为下节所示 Eikonal 自身还未数值收敛。

论文同时讨论了激励频率/带宽对 ToF 拾取的限制。当前 200/500 kHz 的水中波长约为
7.5/3 mm；不能把亚毫米纹理在射线法中的缺失全部归为求解器代码问题。
但这也不能代替独立的模型和离散精度测试。

## Eikonal 数值可信度仍未过关

读取原始仿真节点的声速，然后在同一中心插值场上插入中点加密。
采用 $2N-1$ 个节点保留旧节点和其坐标，而不是把场的范围/中心悄悄移动。
原节点声速保持到 `8e-12 m/s` 内；每条预测都减去相同网格的数值水到时。

| 采集 | 图像网格 E vs K（ns RMS） | 实际仿真网格 E vs K | 加密后 E vs K | 实际网格到加密网格的 E 变化 |
|---|---:|---:|---:|---:|
| NBP D 高频 | 84.540 | 113.296（1024²） | 135.953（2047²） | 30.284 |
| NBP D 低频 | 114.227 | 95.244（256²） | 115.672（511²） | 27.274 |
| OpenBreastUS 低频 | 511.776 | 604.879（1024²） | 668.055（2047²） | 93.840 |

K 为原先保存的观测，不是根据模型选择的最佳拾取器。
加密降低了数值水传播的几何误差，却没有使对象的水相对延迟更接近 K。
**较小 E/K 残差可能含有误差抵消，不能拿它选最粗网格或宣布连续模型已验证。**
一次加密不构成收敛阶证明；本轮没有改 FMM、计算新 FWI 或重新运行 k-Wave。

## 本轮代码交付

| 文件 | 行为 |
|---|---|
| `data/validation_acquisition.py` | 校验实际仿真文件 SHA、介质与坐标对应；仅供事后前向审计，不给求解器 GT |
| `evaluation/arrival_consistency.py` | pair-local 的水参考平移/增益残差，缺失或截断窗返回 NaN，不偷偷改变 valid mask |
| `data/arrival.py` | xcorr 局部 float64 累加，支持显式 `source_onset_s`，新增可选 `aic`，返回明确观测量契约 |
| `core/provenance.py` | 把 ToF 契约传到 run metadata，标明有限带宽观测、正号和水参考定义 |
| `scripts/validate_physics.py` | 暴露 AIC 参数，分离 arrival QC 与 simulation QC 字典，正确写入 simulation QC 状态 |
| `data/waveforms.py` | 澄清“导入阶段不做拾取”而非对已提取 case 仍声称完全未做拾取 |
| `scripts/audit_tof_adaptation.py` | 原始介质、旧 surrogate、三种拾取器、同场网格加密的一键审计，输出 JSON/NPZ/PNG |

新增/改动的参数：

| 参数 | 类型/范围 | 默认 | 意义 |
|---|---|---|---|
| `picker` / `--tof-method` | `xcorr`, `envelope`, `aic` | `xcorr` | 明确选择所测观测量；不在背后融合或切换 |
| `source_onset_s` | 有限实数，秒 | 0 | 发射触发相对记录时间轴的位置；不是用 `time[0]` 猜测的到时偏移 |
| `aic_envelope_fraction` | 浮点，0 到 1 的开区间 | 0.25 | 包络引导的小窗末端阈值 |
| `aic_window_s` | 正有限浮点，秒，或 `None` | 脉冲长度 | AIC 的向前回看窗长；不是成像调优参数 |

契约明确写明 `geometric_first_arrival_certified=false`，以及
`weight_meaning=squared_picker_confidence_not_inverse_noise_variance`。
不能把 confidence 自动转成测量噪声标准差，或把 `K-E(GT)` 当作在线停止的已知 noise norm。
本轮没有把原始 k-Wave `source_signal` 误当成校准后的 Helmholtz 源谱。

### 运行方法

在已安装此分支的仓库根目录运行；`USCT_WORKSPACE` 指向包含 `runs/`、`handoffs/` 的工作区：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/audit_tof_adaptation.py \
  --acquisition "$USCT_WORKSPACE/runs/physics_validation_20260908/high_band_800/D510022534" \
  --case "$USCT_WORKSPACE/handoffs/pro-physics64-20260908/usct_handoff_64/cases/high_band/D510022534/envelope_case.h5" \
  --out "$USCT_WORKSPACE/runs/tof_adaptation_replay/high_d" \
  --tx-count 8 --refine 2
```

另外两组分别将 acquisition 改为 `high_resolution_128/D510022534` 与
`high_resolution_128/breast_train_speed_class_1_000000`，case 对应
`cases/low_band/<case_id>/pressure_case.h5`。每次使用新的输出目录，拒绝覆盖既有审计。
manifest 中的父 property case 路径必须可访问；跨机器迁移须显式重定位，不能换成不同的 GT。
脚本不运行 MATLAB/k-Wave、不做重建，内存按单源 FMM 逐个处理。

产物：`summary.json`（含输入/源码 SHA）、`diagnostics.npz`、`adaptation.png`。
完整压力、GT、数组和图保留在工作区，不进入 Git。

## 验证

- 本地与 A100：`251 passed, 2 skipped`。跳过两项依赖外部 MATLAB 的测试；不能据此声明生产 FWI 已验收。
- Black、Ruff、compileall、CLI help/list 与 synthetic smoke/release audit 已通过。
- 新回归覆盖：精确输入与坐标身份，转置/半格/顺序/SHA 错误拒绝，同节点加密，
  水恒等与已知正负平移、发射时间偏移、AIC 有效性、缺失/截断压力，以及 pair-local 无跨通道拟合。
- 所有原始 case 未覆盖；main 未修改；本轮不报告新的重建 PSNR/SSIM 改善。

## 下一步：先验收适配，再做公平比较

1. **独立标定而非 GT 纠偏。** 固定采集的源、接收链和时间基准，在水及已知均匀/平滑标定体上
   测时间偏差与稳定性。乳腺 GT 只能事后评价，不能用于逐样本拟合延迟偏移或挑拾取参数。
2. **单独收敛 Eikonal。** 固定连续介质表示、源/接收位置，按预声明的到时容差做多级网格检查；
   正确的伴随只证明离散导数一致，不证明前向连续精度足够。现有数十 ns 的变化不能忽略。
3. **固定观测量再谈残差。** xcorr、包络/AIC 首波候选和几何最早到时不能默认为等价。
   若需要有限频率灵敏度，应明确是前向模型的扩展，而不是暗中替换传统直线/Bent 方法。
4. **守住 hold-out。** 当前 pair-local 时间拾取可用于接收通道留出；
   若按频率留出，必须先隔离频率，再做可能读这些频率的特征提取。
   用完整时域脉冲提取的 ToF 不能声称从未使用某个留出频段。
5. **双层验收而非单一总排名。** 一层保留匹配离散观测的求解器测试；另一层使用同一原始
   k-Wave 采集评估端到端成像质量、时间与资源。FWI 可以参与后一层比较，但不能把压力残差
   与 ToF 残差混为一个数据误差排名，也不能把相同迭代次数当成相同计算预算。

这些步骤针对观测定义和数值精度，不是新一轮“把图片调锐”的参数搜索。
