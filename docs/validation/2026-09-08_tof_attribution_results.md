# ToF失配、优化充分性与信息限制：当前会话59套受控实验

日期2026-09-08。本次从fe505f3b626a8999333222df534ddacc81dfb23e接续，未将此前报告和测试计数当作本次执行。新增代码依次发布95ea8e5（协议）、fe6ffb2（诊断核）、d5e0257（实际执行的CLI）、f6f328b（逐比特验证集保护、完整性检查）。始终在work/physics-agent-validation，不改main或生产FWI。

## 决策

保留CGLS/SIRT/SART/Bent及共同k-Wave压力基准，但将数值认证、独立ToF模型能力、同源压力端到端比较分轨。不能把真实压力提取的ToF换成算法自身生成的surrogate，然后声称解决了输入适配。也不能因为当前ToF上图像模糊就断言全部算法到达上限。

本次支持三个不同结论：结构化输入/模型失配显著；固定当前目标的剩余优化收益很小；给定采样和未限制先验的直线问题确有不可辨识方向。复杂病例连续Eikonal与实际模拟场的完整认证仍未完成，因此没有宣称彻底分离所有误差，也没有可部署的GT-free ToF纠错改进。

## 实际执行

CPU Python3.13.5、NumPy2.3.5、SciPy1.17.0、Numba0.65.1，约4核/4GiB。解压、离线editable安装成功，命令行Git DNS失败，通过connector普通快进推送。起始与结束均验证61文件哈希及12 HDF5数组/schema，私有原始输入未修改或提交Git。

完整四基线与附件比较：直线三法最大数组差4.548e-13m/s，Bent完全一致；split、组织评价、停止信息通过验收。开始218 passed/2 MATLAB skipped，新增13个诊断与证据检查用例后231 passed/2 skipped。Black、Ruff、compileall、索引补齐后的release audit通过。f6f328b源快照下载复核，tree f56e042312cf5f5ce982569432a07afb10d0eabe与本地被测源树完全一致；保留并同步了并发新增的文档，没有覆盖它。

最终collector验收59/59套结果：三采集×三类ToF×四方法=36；五种子×三直线方法误差去相关=15；Bent暖启动2；解析光滑体模4；32² CGLS对照2。另有独立Cholesky两解、LSMR两类数据及收敛确认，不算额外病例。仅两张不同乳腺解剖图（high/low D共享解剖），加一张解析体模，不是59例临床或盲测。实际停止41 max_iterations、17 validation_plateau、1 target_residual，没有GT停止；最长记录求解约205.9s，未触及600s。

保持原64TX/RX、256²物理网格、有效mask/权重、water1500初始化、0.125/seed42接收点验证和互易训练排除；不使用GT更新ROI。原质量权重仅为控变量，不被假定为噪声逆方差。全部沿用120/200/100/20上限与原早停政策。validation参与选迭代，不是独立test。

## 1. 同模型/交叉模型/实际ToF

S为GT经当前Siddon生成；E为GT经当前离散Eikonal生成；K为原k-Wave包络差分延迟。S/E是明确标记的oracle同离散算子诊断，具有inverse-crime风险，不是波形性能。每格为同mask组织RMSE(m/s)/SSIM：

| high-D | S | E | K |
|---|---:|---:|---:|
|CGLS|23.32377/.210493|27.59828/.127440|27.19321/.114274|
|SIRT|24.33914/.194902|27.67864/.124976|27.00560/.121096|
|SART|24.45367/.190073|27.66338/.124135|26.92774/.122694|
|Bent|25.73943/.177183|26.27108/.139334|27.08777/.130122|

| low-D | S | E | K |
|---|---:|---:|---:|
|CGLS|23.32692/.209437|27.62052/.127117|32.32602/.083937|
|SIRT|24.33348/.194142|27.66696/.124899|31.45263/.092992|
|SART|24.46028/.188630|27.64218/.123967|31.45345/.093268|
|Bent|25.77269/.176112|26.60061/.135895|33.91362/.068984|

| low-OpenBreastUS | S | E | K |
|---|---:|---:|---:|
|CGLS|24.36204/.340956|35.35927/.182554|37.74073/.107020|
|SIRT|27.28582/.326975|35.48756/.194562|35.00465/.179460|
|SART|27.43759/.324956|35.49797/.194479|35.08502/.180092|
|Bent|31.87637/.280390|27.42169/.269639|35.46875/.174980|

匹配ToF能改善部分结果，但复杂乳腺仍缺细节。Bent在自身E数据也非完美；失配影响依病例而异。S分数好不证明直线物理更真实。另一个相同几何/原参数的光滑解析匹配体模，四法RMSE为1.25558/1.57805/1.45338/.93149m/s，SSIM为.987415/.983075/.984141/.990834，说明当前实现能解低复杂度匹配问题，不支持复杂组织的同等结论。

## 2. Picker与观测量不是一件事

本次high-D raw重提取包络与保存值完全一致。128个不同无向通道的water数字平移±250ns，包络RMS误差约.0825ns，xcorr约.0000354ns；dt16.1317ns。支持时间轴/符号/插值数值正确，不等于量到了几何first arrival。

实际object波形即使允许平移和标量增益，water拟合归一化残差中位数仍为20.55%（包络）/16.73%（xcorr，127条有效）。两picker在共同2746条上差RMS64.23ns，6条xcorr无效未补0。GT预测与K的非加权RMS差：high-D直线/Eikonal84.57/87.42ns；low-D120.05/114.31ns；low-OB458.59/499.80ns。它们是复合失配，不是纯picker误差或噪声方差。low-band没有raw，未重做其时域picker。

## 3. 新增等能量、等均值、同验证集的误差结构实验

e=K-A(s_GT)。只在训练集内部对同组无向对的白化均值置换，去常数投影并归一化能量，恢复原加权均值和反对称误差；不跨分区传递信息。非训练观测逐比特保持，训练误差加权能量/均值、互易缺陷均通过检查。不是iid高斯噪声，不保持完整直方图，更不是可部署的GT纠错。

|方法|原K RMSE/SSIM|五种子均值 RMSE/SSIM|RMSE范围|
|---|---:|---:|---:|
|CGLS|27.19321/.114274|24.26581/.199426|24.08575—24.41881|
|SIRT|27.00560/.121096|24.73788/.187748|24.61182—24.91402|
|SART|26.92774/.122694|24.83586/.183490|24.73540—25.00520|

同量级误差的空间结构会显著影响结果。这是使用GT构造误差的归因证据，不是原K数据上的性能提升。种子0..4固定，示图固定seed0；五种子不是五病例。

初次构造脚本改变validation内部成对项，被不变性断言阻止；改成train-only后完整运行。随后发现两条validation有6.617e-24s减后加舍入，改为原观测直接复制，重跑全部15套，验证集逐比特相同，15幅重建与前一版逐元素相同。旧/失败日志分开保存。

## 4. 精确误差恒等式与互易性下界

r=E-S，g=K-E；K-S=r+g，平方范数必须包含2<r,g>交叉项。高D加权RMS为110.339ns、87.430ns，总差却仅84.579ns，夹角余弦-.65643；low-D/low-OB为-.42575/-.58741。误差抵消明显，不能用RMS相减给picker和折射分配百分比；g仍包含波/射线、特征与离散误差。

任意精确互易预测对每对观测的最小加权SSE为wi*wj/(wi+wj)*(bij-bji)^2。高D该RMS下界仅.00661ns，解释不了84ns差异；近互易并不意味着准确。当前离散Eikonal正反方向差RMS18.60ns，数值误差仍需认证。

## 5. 优化充分性

独立LSMR求同一训练加权A及0.02细网格Laplacian目标：2000步是code7，随后仅增maxiter到20000，S/K分别3101/3214步code2满足容差，解在1300—1700边界内。S图像23.32377/.210493变23.28540/.211641，K由27.19321/.114274变27.05793/.117791；目标值比.95517/.90989。最终LSMR解不按validation选，额外计算已记录，不能作等预算竞赛。

新增32²参数空间直接Cholesky：AB数据项、LB细网格正则，H条件数3056.28、正定。S/K法方程相对残差4.12e-16/4.36e-16，系数盒约束满足；直接目标/同空间CGLS为.99739075/.99975249。固定这一凸二次问题的求解已经充分，但不是最佳先验或所有算法的质量上限。

Bent追加CGLS80步数据初始化：E的RMSE26.27108→25.99238，SSIM.139334→.143518；K为27.08777→27.11099、.130122→.130303。不能靠更多初始化工作宣称解决纹理问题。

## 6. 条件化的信息限制

65536未知像素，2752有效有向ToF最多1376无向路径，训练1049无向路径。维数不是SSIM上限；实际构造了只依赖几何的25mm圆盘零空间扰动，速度1465.11—1536.51m/s，与water圆盘内RMS相差10.00470m/s，所有有效直线ToF最大差6.695e-12ns。

取两预测中点为共同观测，两图仅需各不超过3.348e-21s通道误差即可解释。因此在包含这两图且允许每通道1e-18s误差的模型类内，三角不等式给出任何估计器在两者之一的圆盘RMSE至少5.00235m/s。此为两点最坏情况条件下界，不是当前乳腺27m/s的不可突破证明，不覆盖强先验排除该扰动、Bent或FWI。

补充离散Eikonal对此图响应RMS3.034ns/max22.991ns，说明不同离散模型的不可见方向不相同；含离散差异，未作连续模型认证。32²训练矩阵相对奇异值1e-12/1e-3下秩1004/954；GT慢度全场L2投影的组织RMSE20.8089，不是组织SSIM最优界。

## 7. 未完成的物理分离与后续决策

固定8TX/64RX、344原有效通道的前向网格诊断：原分片常数场重复到256/512/1024后，对K差RMS84.54/109.32/137.08ns，级间35.95/36.30ns。按记录插值的1024/2048场差113.30/137.27ns，级间31.57ns。细化并未更接近pulse特征，也未认证复杂场连续收敛。实际simulation_input.mat缺失，记录插值不是逐字节模拟场。

因此下一阶段需真实模拟输入场、源接收离散映射、源波形/时间零点；low-band还需raw。高D已有raw无需重复索取。独立收敛Eikonal/射线参考、波动模拟网格收敛和固定介质频带对照尚未执行；不能归入本次完成。

继续保留四方法作为agent构件：直线三法是相同前向下不同求解/先验/停止；Bent优先修复杂场离散与first-arrival定义，精细纹理则保留生产FWI及另行明确命名的有限频率方法。共同压力的端到端图像质量可跨模型比较，但ToF/压力残差不能直接排名，迭代单位不等价。agent应看到观测定义、数值认证、失配风险、先验/成本、失败原因和oracle标志；平台或小validation残差不是准确度证书。算法发现必须额外封存病例级test，限制总试验预算。

没有执行A100、MATLAB、生产FWI、完整128通道、临床测量或新的k-Wave仿真；没有训练AI或后期锐化。当前报告用于有限受控诊断，没有泛化统计保证。

## 复现

H为私有usct_handoff_64；O为新的仓库外目录。当前分支仓库内执行（安装在本机实际使用离线方式）：

```bash
export H=/path/to/usct_handoff_64 O=/path/outside/repo/tof_audit
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
python "$H/tools/verify.py" --repo .
python -m pytest -q
python "$H/tools/run_baselines.py" --repo . --out "$O/baseline"
python scripts/check_64_baselines.py --reference "$H/baselines_64" --actual "$O/baseline" --out "$O/baseline_gate.json"
python scripts/audit_tof_prepare.py --handoff "$H" --out "$O"
for g in high_d low_d low_ob; do python scripts/audit_tof_invert.py --handoff "$H" --out "$O" --group "$g"; done
python scripts/audit_tof_observable.py --handoff "$H" --out "$O"
python scripts/audit_tof_recorded_grid.py --handoff "$H" --out "$O"
python scripts/audit_tof_information.py --handoff "$H" --out "$O" --lsmr-iterations 20000
python scripts/audit_tof_attribution.py --handoff "$H" --out "$O"
python scripts/audit_tof_coarse_optimum.py --handoff "$H" --out "$O"
for mode in warm smooth; do python scripts/audit_tof_targeted.py --handoff "$H" --out "$O" --mode "$mode"; done
python scripts/summarize_tof_attribution.py --out "$O"
```

组成命令在本次执行过；实际分组调度，原2000步LSMR结果保留并另目录做20000上限确认。59套结果、25张灰度/曲线图、原始日志、CSV与详细中文报告保留私有交付包，不将数据包提交GitHub。

研究依据（外部论文不是本次数值结果）：Javaherian/Lucka/Cox2020 https://arxiv.org/abs/2005.11204 在全波模拟乳腺上展示Bent-ray；Javaherian/Cox2021 https://arxiv.org/abs/2105.14098 展示超出单ToF的射线Green散射反演；SciPy https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html 给出code2/code7语义。当前FMM不是第一篇射线链接实现的逐行复现，也未在本轮实现第二篇完整算法。
