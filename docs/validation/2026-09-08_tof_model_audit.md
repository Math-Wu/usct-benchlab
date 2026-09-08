# ToF适配与反演局限：本轮受控实验

日期2026-09-08。分支work/physics-agent-validation，起点8be7229352588c13522b881b1b8d13e23e9c3de9；诊断代码fe505f3b626a8999333222df534ddacc81dfb23e。以下不是历史测试计数。

## 决策

保留CGLS/SIRT/SART/Bent及共同k-Wave压力基准，但分开“匹配模型的数值认证”与“同源压力的端到端性能”。不通过替换为算法自身生成的ToF来宣称解决了输入问题。CGLS/SIRT/SART共享直线传播算子，主要区别是求解、更新、隐式先验与停止；Bent才改变几何传播模型。

当前证据排除了一部分初等拾取/优化解释，也证明了采样非唯一性；没有完全分离连续模型误差与离散误差，不能宣称已经到达四种方法的无条件质量上限。生产FWI未改动或运行，fwi_tiny/Full Green没有被当作生产FWI。

## 本轮实际执行

CPU Python3.13.5、NumPy2.3.5、SciPy1.17.0、Numba0.65.1。61文件哈希及12 HDF5数组/schema检查通过。起点源树84476fa0463ebe8149158078b4ff75bc4e45c465恢复一致。离线安装成功，终端Git DNS失败，通过GitHub连接器逐批普通快进推送。

完整四基线复跑通过数组、split和stopping门槛；直线三法与附件最大差不超过4.55e-13m/s，Bent相同。开始测试209通过/2 MATLAB跳过；新增9项诊断测试后218通过/2跳过。Black、Ruff、compileall及release audit通过。发布代码fe505f3的CI成功，下载其Actions源快照并校验全部字节/文件模式，tree=fb56e63d0f42578a89f7995bfe4516d54e910193。

36套主重建：high-D三种输入×四法12套，low-D和low-OB三种输入×三个直线法18套，平滑体模4套，Bent暖启动2套。另有独立LSMR和重复执行的等价性验证，不算新样本。仅两张不同乳腺解剖图，加一张解析体模；high/low D共享解剖，不能包装成36例临床验证。

主实验保持原64×64几何、256平方物理网格、有效通道、权重、seed42、receiver_fraction0.125和互易训练排除。GT不参与ROI、初始化、选迭代、停止或offset校正。合成匹配观测明确标记oracle，仅用于诊断；原质量权重保留用于控变量，不意味着其是控制数据的噪声方差。validation参与停止，不是独立test。原组织mask、固定full-GT range、完整组织SSIM窗口及水背景指标均保留。

## 1. 匹配模型与k-Wave交叉实验

S=GT经当前Siddon生成延迟；E=GT经当前离散Eikonal生成延迟；K=原k-Wave包络延迟。S/E有意使用相同离散算子，属于inverse-crime诊断，不是真实压力性能。

每格为组织RMSE(m/s)/SSIM：

| high-D方法 | S直线生成 | E Eikonal生成 | K k-Wave提取 |
|---|---:|---:|---:|
|CGLS|23.32377/0.210493|27.59828/0.127440|27.19321/0.114274|
|SIRT|24.33914/0.194902|27.67864/0.124976|27.00560/0.121096|
|SART|24.45367/0.190073|27.66338/0.124135|26.92774/0.122694|
|Bent|25.73943/0.177183|26.27108/0.139334|27.08777/0.130122|

匹配输入能改善部分结果，但复杂解剖图在自洽数据下仍未恢复精细纹理。Bent在自身E数据上也只有26.27/0.139。不能把S数据上的更好分数解释为S更真实：路径敏感性、离散误差及正则作用不同。

low-D三种直线方法S数据RMSE为23.327/24.333/24.460，K为32.326/31.453/31.453；low-OB的S为24.362/27.286/27.438，K为37.741/35.005/35.085。趋势不局限high-D，但无统计推广保证。low-band本轮没有新Bent控制或时域重拾取。

## 2. 拾取与观测定义

本轮从high-D raw重新提取包络ToF，与存储值逐元素一致。确定选择128个不同无向通道，把水参考数字平移±250ns：包络拾取RMS误差约0.0825ns，xcorr约0.0000354ns，全部有效。采样dt16.13ns。此项验证有限采样下的平移/符号，不验证物理first arrival。

实际object波形仅用平移加标量增益拟合水参考，归一化残差中位数为包络20.55%、xcorr16.73%；xcorr该检查只有127条有效。全采集两种picker在共同2746通道上的延迟差RMS64.23ns；6条xcorr失效未补零或偷换覆盖。

GT下直线/Eikonal与实际提取ToF的RMS差：high-D84.57/87.42ns，low-D120.05/114.31ns，low-OB458.59/499.80ns。high-D分别占观测差分ToF范数43.61%/45.08%。这些是混合失配，不能直接叫picker误差或噪声标准差。

Eikonal在high-D的互易性RMS误差18.60ns，而观测约0.0132ns。离散传播误差仍可见；高SNR、高相关或近互易不能证明正确提取了几何first arrival。

## 3. 网格诊断

仅前向诊断使用固定8个TX、64 RX，共344原有效通道；不改变主重建的64×64采集。将原分片常数速度场最近重复到256/512/1024平方网格，对k-Wave延迟的RMS差为84.54/109.32/137.08ns，级间变化35.95/36.30ns。

重新按记录的order-1速度插值与仿真坐标评估1024/2048平方网格，对k-Wave差113.30/137.27ns，级间31.57ns。原simulation_input.mat未提供，不能声称逐字节恢复了真实仿真场，也不是新k-Wave采集。

细化没有使预测更靠近当前脉冲特征，并且尚不能认证复杂场的连续收敛。粗网格可能含误差抵消。简单Snell/Fermat界面在41/81/161网格的误差1.242/0.813/0.506us虽呈改善，不替代复杂病例收敛认证。

## 4. 固定目标下的独立优化

用独立SciPy LSMR求解与原CGLS同一训练加权Siddon加0.02细网格Laplacian目标，不看GT优化。先2000步触及上限，未称为收敛；随后只把maxiter扩到20000，容差不变，S/K分别在3101/3214步以code2达到规定最小二乘容差，解满足1300–1700m/s边界。

|输入|原CGLS RMSE/SSIM|LSMR RMSE/SSIM|目标值比LSMR/原|
|---|---:|---:|---:|
|S|23.32377/0.210493|23.28540/0.211641|0.95517|
|K|27.19321/0.114274|27.05793/0.117791|0.90989|

因此固定这一目标时，更多求解不能解释纹理差距。这不是改变模型或先验后的上限。LSMR用最终迭代而非原validation选择，只作诊断，不作为公平预算比较。

Bent使用现有CGLS初始化80步：E的RMSE26.271→25.992、SSIM0.139334→0.143518；K的RMSE27.08777→27.11099、SSIM0.130122→0.130303。不存在大幅提升，K的RMSE略差。额外初始化工作已计入，不称为与默认同成本。

## 5. 信息与模型复杂度

high-D有效2752有向观测最多对应1376条无向路径；训练2098条对应1049条无向路径，未知像素65536。维数本身不能给出SSIM上限，但存在很多不可见方向。

实际零空间反例：仅由几何选25mm圆盘，将4mm振荡场移除观测行空间分量，不用GT。正声速图范围1465.11–1536.51m/s，与水图圆盘内RMS相差10.0047m/s；全部有效直线预测延迟最大差仅6.70e-12ns。即使ToF完全准确，不排除该类扰动的模型类也不能唯一恢复所有图像。先验可能排除该反例，Bent或波形数据可能检测到它；不能将其推广成所有算法的上限。

32平方双线性参数空间的训练矩阵在相对奇异值1e-12阈值下秩1004，1e-3下954。GT慢度投影到该空间的全场L2最佳投影，组织RMSE20.81/SSIM0.283；它不是组织RMSE最小值、SSIM最大值或可达重建保证。

同几何/权重/配置下的两块光滑解析慢度体模，使用各自离散模型生成观测：

|方法|RMSE m/s|SSIM|
|---|---:|---:|
|CGLS|1.25558|0.987415|
|SIRT|1.57805|0.983075|
|SART|1.45338|0.984141|
|Bent|0.93149|0.990834|

阳性控制说明当前实现不是普遍坏掉，不能据此宣称乳腺细纹理或全波数据可以同样恢复。

## 6. 开发与agent建议

算法能力必须条件化于观测生成、特征定义、几何、噪声、模型类、先验与预算。数据适配不能只检查数组/单位，而要区分first-arrival、包络阈值、互相关延迟和相位延迟。

分三条轨：匹配算子数值认证；独立几何/细网格ToF能力测试；共同raw压力端到端测试。第三轨不准换成自洽surrogate，前两轨不能冒充临床/full-wave性能。允许每种模型公开、合理的预处理；同源raw不代表每法利用相同信息。可以比较GT图像和实际成本，不直接排列ToF与压力残差，也不强制同一迭代数或不可达质量目标。

直线三方法先保留为快速可复现基线，不优先盲目加迭代减正则；Bent仍值得攻克复杂场离散误差和first-arrival定义。拟合有限频率延迟应另增与该特征相匹配的finite-frequency traveltime模型，而不是改名冒充原Bent。细纹理任务继续以生产FWI为主，但仍需记录初始化、预算、失败和独立验证。

无GT时可用水标定、时间/几何核验、shift对照、波形畸变、多窗口/频率稳定性、独立validation发现风险，不能据此认证真值图像质量。离线GT只用于能力诊断，不能用于修正实际观测offset。本轮K数据CGLS验证相对残差0.240，小于S数据0.340，图像却更差；残差小不能跨观测定义推断图像更准确。

严格分离连续模型误差还需准确实际仿真场、离散传感器位置、源波形/时间记录，几何与全波数值收敛，以及固定场上分别改变带宽和对比度的实验。新k-Wave模拟、A100、MATLAB、原128通道、其余六个low病例及临床数据本轮未运行。没有完整识别全部误差源或证明全球最优质量。

## 7. 软件与执行审计

新增tof_audit_common.py和六个audit_tof_*.py诊断CLI，不改生产默认算法。9个新增回归测试覆盖控制provenance、保留原测量/权重、错误shape/非有限值、HDF5往返以及streaming Eikonal与原forward等价。

全部主数值主体本轮执行。便携版额外复跑prepare（三组所有预测/mask/weight/split数组一致）、invert（三个直线控制数组及停止/split一致）、observable（重拾取与网格数组一致）、information（收敛LSMR）。recorded-grid/targeted的原数值主体已执行，封装CLI仅help/静态检查，未称为另一次全流程运行。36个主重建source hash相同且source_changed_during_run=false。

保留最初两个实验脚本错误：测量替换未同步domain、未注册算法。修复后有效重跑，失败目录不进统计。早期部分控制输出继承父采集provenance，另加audit_context.json明确oracle，未篡改原记录；便携版已原子化正确标签并验证数值等价。

## 复现命令

可移植顺序命令，不声称整段shell又执行了一遍。先进入本分支；OUT使用仓库外的新目录。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
H=/absolute/path/usct_handoff_64
OUT=/absolute/path/new_tof_audit
python -m pip install -e '.[dev,viz,performance]'
python "$H/tools/verify.py" --repo "$PWD"
python "$H/tools/run_baselines.py" --repo "$PWD" --out "$OUT/baseline"
python scripts/check_64_baselines.py --reference "$H/baselines_64" --actual "$OUT/baseline" --out "$OUT/baseline_acceptance.json"
python scripts/audit_tof_prepare.py --handoff "$H" --out "$OUT"
python scripts/audit_tof_invert.py --handoff "$H" --out "$OUT" --group high_d
for G in low_d low_ob; do
  python scripts/audit_tof_invert.py --handoff "$H" --out "$OUT" --group "$G" --algorithms straight_cgls straight_sirt straight_sart
done
python scripts/audit_tof_observable.py --handoff "$H" --out "$OUT"
python scripts/audit_tof_recorded_grid.py --handoff "$H" --out "$OUT"
python scripts/audit_tof_information.py --handoff "$H" --out "$OUT" --lsmr-iterations 20000
python scripts/audit_tof_targeted.py --handoff "$H" --out "$OUT" --mode smooth
python scripts/audit_tof_targeted.py --handoff "$H" --out "$OUT" --mode warm
pytest -q
```

## 外部依据，不是本轮实测

Javaherian, Lucka & Cox (2020), Refraction-corrected ray-based inversion for three-dimensional ultrasound tomography of the breast, arXiv:2005.11204：在包含ToF拾取误差的full-wave乳腺模拟上验证Bent-ray，支持不先验排除全波数据，不证明本仓库复现其3D ray-linking实现。

Korta Martiartu, Boehm & Fichtner (2019), 3D Wave-Equation-Based Finite-Frequency Tomography for Ultrasound Computed Tomography, arXiv:1908.03302：从水标定推导互相关延迟有限频率灵敏度，包含离射线散射/衍射敏感性及实验体模验证。本轮没有实现该论文的完整算法。
