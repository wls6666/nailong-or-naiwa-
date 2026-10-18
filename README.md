# Nailong or Naiwa · V3 Open-set CNN

> V4 Phase 2：[等预算 Random Cartoon OE / Hard Mining 实验](experiments/v4_phase2/README.md)，研究资产与生产 V3 分离。

> V4 Phase 1 实验报告：[固定 Hard-OOD 与 MobileNet 对照](experiments/v4_phase1/README.md)。结论 C，未替换本页所述的线上 V3。

[网站](https://nailongornaiwa.pages.dev/) · [GitHub](https://github.com/wls6666/nailong-or-naiwa-)

在原项目上增量升级：保留四层自训练 CNN、128×128 输入、原始目标数据划分、网页布局和 Cloudflare Pages。现在先判断图片是否接近已知特征分布，再决定奶龙 / 奶蛙；拒绝时显示“都不是”，不伪造 Other 概率。

```text
图片 → JavaScript RGB/resize/CHW → ONNX Runtime Web → logits [1,2] + embedding [1,256]
                                                     ├→ 稳定 softmax → 奶龙 / 奶蛙
                                                     └→ Mahalanobis + Energy → 已知域检测
任一 OOD 分数超过 validation 校准阈值 → 都不是；否则显示目标类和条件 softmax 概率
```

所有训练、统计、指标、ONNX 和浏览器检查均已实际执行。页面只下载静态文件；没有预测 API、mock、随机分类、图片颜色/文件名规则或加载失败后的默认预测。

## 实际效果与代价

测试集始终保持独立。先在 validation 完成模型、normalization、temperature、OR/AND 和阈值选择，写入 `results/v3/production_selection.json`，之后才提取 test features。没有根据下面的 test 结果反复调参。

测试共 **73 ID + 150 Easy OOD + 149 Hard OOD**。Hard 为 53 个独立 Unicode 角色身份的多来源渲染，149 张不是 149 个独立语义身份。表中 ID accuracy 是**拒识后正确识别数量 / 全部 ID**；误拒也计错。

| System / Method | ID accuracy（含误拒） | Easy FAR ↓ | Hard FAR ↓ | AUROC ↑ | FPR@95TPR ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| V2 旧三分类部署规则 | 93.15% | 2.00% | 74.50% | 78.94% | 38.13% |
| V2 encoder + Max Softmax（两类） | 91.78% | 74.67% | 94.63% | 69.27% | 89.97% |
| V2 encoder + Mahalanobis | 90.41% | 16.00% | 55.03% | 92.67% | 45.82% |
| V2 encoder + Energy | 93.15% | 7.33% | 83.22% | 77.42% | 39.13% |
| V2 encoder + distance + energy | 89.04% | 6.00% | 53.69% | 90.22% | 45.82% |
| V2 encoder + distance + energy + p_other | 89.04% | 0.67% | 50.34% | 87.04% | 45.82% |
| **V3 Custom CNN + OE + distance + energy（生产）** | **89.04%** | **12.00%** | **4.70%** | **95.49%** | **47.49%** |
| V3 OE encoder + Energy only（消融） | 93.15% | 16.67% | 12.75% | 96.52% | 8.70% |
| MobileNetV3-Small frozen + distance + energy（对照） | 90.41% | 28.67% | 39.60% | 90.24% | 42.81% |

**不能把本次描述为所有指标均提高。** 生产系统将 Hard FAR 降为 **7/149**，但 Easy FAR 从 **3/150 增加至 18/150**，ID 最终正确数从 **68/73 降为 65/73**。闭集两类准确率实际为 **70/73 = 95.89%**；有 **5/73 = 6.85%** ID 被拒绝，接受的 ID 中 **65/68 = 95.59%** 分类正确。这批 test 上闭集分类未下降，但拒识造成有效识别率下降。

混合 OOD FAR 为 **25/299 = 8.36%**，ID TAR **93.15%**，AUPR（OOD 为正）**98.31%**。FPR@95TPR 定义为**测试 ROC 上达到至少 95% ID 接受率时的 OOD 接受率**，不是当前 production 阈值的 FAR。少量高 OOD-score ID 尾部样本使该指标达 **47.49%**，说明要求高 ID 保留率时仍不稳健。Energy-only 在 test 的 AUROC/FPR@95 更好，但默认联合方案是 validation 锁定的选择，没有在看完 test 后改选。

完整 15 组消融/对照结果及定义见 [比较表](results/v3/comparison.md)、[逐方法指标](results/v3/method_comparison.json)、[CSV](results/v3/method_comparison.csv)。旧二分类 68/73、旧三分类结果保存在 `results/binary/`、`results/v2/` 和旧根目录报告；没有覆盖它们。

## 为什么原来的 Other 弱

V2 Other 主要来自 CIFAR-100 / Oxford Pets，识别照片域较容易，却没有覆盖很多接近目标角色的卡通。softmax 只比较输出之间的相对大小，不能保证输入属于已知分布。本次新 Hard test 上旧部署规则误接受 **111/149**，直接暴露了这个缺口；不能仅凭旧照片 Other recall=98% 推断开放世界可靠。

V3 不把全部 Other 强迫聚成一个语义 cluster。先验证冻结 encoder 的 post-hoc 方法；validation Hard FAR 仍约 60%，因此才进行 Phase B：两类 CE + energy-margin **Outlier Exposure**。没有采用 SupCon / ViT / CLIP，没有扩大几千张随机 Other。

## 数据与无泄漏检查

原数据：[CristinXie/NaiwaOrNailong-dataset](https://github.com/CristinXie/NaiwaOrNailong-dataset)，固定提交 `900b0419d9c0cfb5ef3c12938a2cbf0ae5eb2abb`；原 split 不移动、不重分。

| Split | Nailong | Naiwa | Easy OOD（原 Other） | Hard OOD |
| --- | ---: | ---: | ---: | ---: |
| train | 184 | 161 | 600 | 143 |
| val | 39 | 34 | 150 | 70 |
| test | 39 | 34 | 150 | 149 |

Easy 复用 V2 两来源、固定 seed=42 的自动下载数据及完整 manifest，不再扩大随机样本。Hard 来源是公开、非 AI 生成的黄色表情、卡通动物、青蛙、龙、恐龙、玩具/怪物角色：

- Google **Noto Emoji**, release v2.047（Apache-2.0）：[来源](https://github.com/googlefonts/noto-emoji/tree/v2.047)。
- **Twemoji**, commit `7407fa31c51be5ab45626b8ab5554d50cc8073f6`（图形 CC BY 4.0）：[来源及署名](https://github.com/jdecked/twemoji/tree/7407fa31c51be5ab45626b8ab5554d50cc8073f6)，源自 Twitter Twemoji / jdecked 项目。
- Microsoft **Fluent UI Emoji 3D**, commit `1ffb34c752ecf5d402f04cfb4b392c77f57c54bc`（MIT）：[来源](https://github.com/microsoft/fluentui-emoji/tree/1ffb34c752ecf5d402f04cfb4b392c77f57c54bc)。

`prepare_hard_ood.py` 按语义类别确定候选、固定 seed、按 Unicode **角色身份分组划分**；同一角色的 Noto/Twemoji/Fluent 渲染不能跨 split。青蛙、龙脸等身份在任何打分前预留 test，没有按模型错误挑选测试图片。检查文件哈希、RGB 像素哈希、来源身份，原始数据与 Hard 间跨 split 精确像素重复为 **0**。Manifest 包含每张 URL、源版本、身份、split、类别、路径、许可、哈希。数据与源许可自动下载到 `data/`，不把图片重复提交至本仓库。

这是**小型卡通图标/3D emoji probe**，不是全面的相似商业卡通/玩偶/二创 benchmark。训练和测试仍共享源画风，精确去重不能排除相似实例。报告中不据此承诺真实世界 Hard FAR=4.70%。CIFAR 图像未明确提供统一图片许可；Oxford Pets CC BY-SA 4.0，原目标图逐图授权未知。研究/演示之外使用需核查原数据权利，细节和引用见 [V2 数据说明](README_V2.md)。

## 模型、训练与校准

模型沿用四层 Conv/BN/ReLU，通道 32/64/128/256，前三块 MaxPool2，GAP 得到 **256D raw embedding**，Dropout(0.35) → Linear(256,2)。forward 输出 raw logits；`forward_outputs` 同时暴露真实 embedding，不在模型中加 softmax。实际映射为 `nailong=0,naiwa=1`，从 checkpoint/config 传递并核验，没有在 JS 猜测顺序。

初始化来自冻结 V2 的本项目 CNN 权重（不是 ImageNet）；两个目标 head 行从原映射读取。实际 CPU 训练 **20 epochs**，最佳第 **20** 轮，seed42、AdamW lr=1e-4、weight_decay=1e-4、ID batch32 + OE batch32、梯度裁剪5。ID 继续轻度翻转、旋转和 ColorJitter，val/test 无随机增强。Python/NumPy/PyTorch 均设种子；跨硬件版本不保证 bitwise 一致。

Loss = ID CrossEntropy + 0.1 × `[relu(E_ID - m_in)^2 + relu(m_out - E_OE)^2]`。训练 Energy T=1；`m_in=-1.268482089` 来自初始化 ID **train** energy 的90分位，`m_out=m_in+2`，这些是训练 objective 的 margin，不是 OOD 决策阈值。OE train 600 easy +143 hard；采样各域一半概率质量，共采样345张/epoch。OE 不参与第三类 CE，也不要求全部 Other 特征聚成一团。最佳 checkpoint 按 validation OOD 和 ID 指标选择，允许相对原 val ID accuracy 最多下降3个百分点、ID TAR≥95%。

`mine_hard_negatives.py` 只对 hard **train** 用冻结 V2 真实预测排序，143 张中 **113 张**为 V2 误接受；本次 OE 使用全部143张，未额外使用报告中的 priority 权重。用户反馈目录继续兼容：

```text
data/hard_negative/       # 新的相似角色等，训练用
data/false_positive/      # 真实线上误判，新实例，训练用
data/hard_ood/train/      # 自动的难负样本曝光
data/hard_ood/val/        # 仅校准
data/hard_ood/test/       # 仅最终评估
```

原反馈目录本次为空。下次训练会自动加入反馈，拒绝与已知 val/test 像素相同的文件。不要将本次测试错误复制进训练再报告同一 test 的改进。反馈不会自动变成独立测试数据；新增独立测试要在训练/校准前预留新身份实例。

距离统计只拟合 **345 个 ID train embedding**：每类中心 + pooled within-class residual 的 **LedoitWolf shared covariance**，用 shrinkage 保证正定，完整逆矩阵256×256，没有把欧氏距离称为 Mahalanobis。计算 `min_c (z-μ_c)^T Σ^-1 (z-μ_c)`。比较 raw/L2，最终选 raw；validation Hard AUROC raw=93.99%、L2=89.47%。最终 shrinkage≈0.01734。

Energy = `-T × logsumexp(logits/T)`，Python/JS 都使用稳定 logsumexp。Phase A 三分类实验只取两个 ID logit，避免 Other logit 很高被误当作低能量 ID；生产本身只有2 logits。

校准只读取 train/val feature 文件，比较两种 embedding、T={0.5,1,2,5}、距离/能量单独及 OR/AND、旧 p_other 信号、92/95/97/98% ID 接受 operating points。默认选 val ID TAR≥95% 中**最小最坏域 FAR**，再最小 pooled FAR、保留正确 ID；同分偏好简单信号。最终 production：

```text
normalize_embedding = false
Mahalanobis threshold = 286.4332033471298
Energy temperature = 5.0
Energy threshold = -2.4278297972582807
reject = (distance > D_threshold) OR (energy > E_threshold)
```

最终 validation ID TAR95.89%、Easy FAR7.33%、Hard FAR8.57%；完整 OR/AND 消融和不同 operating points 在 `phase_*/calibration.json`、`validation_operating_points.csv`。没有证据表明任意新分布都服从这些阈值。

Mahalanobis 单独有区分能力，但 test Hard FAR46.98%仍很弱；OE 后 Energy 单独显著改善，联合在 val 降低最坏域 FAR，并在 test 将 Hard FAR降至4.70%。联合的 AUROC/FPR@95 不优于 Energy-only，这一局限完整保留。

## MobileNet 对照

在自训练 CNN 的 post-hoc 效果不足后，独立测试 torchvision **MobileNetV3-Small IMAGENET1K_V1**：冻结 backbone，ID train PCA576→128，再用 PyTorch CE 真实训练两类 head。PCA/中心仅 fit ID train；ImageNet mean/std 包在 ONNX 图内，外部预处理不变。它的 validation 最坏域 FAR41.43%，不及最终 custom OE 的8.57%，没有用于生产。对照的 test Hard FAR39.60%也如实保留。

真实 Chromium WASM 单线程、3次 warmup+10次真实图片推理：Custom ONNX **1,558,506 bytes / 中位19.10ms**，MobileNet **4,014,459 bytes / 8.45ms**。MobileNet 更快但本次 OOD 指标更差；速度只代表当前 CPU 环境。正式上传浏览器检查中 Custom 推理中位20.65ms，256维完整矩阵 OOD 计算中位0.35ms。没有以模型更大或预训练为由自动替换。

## 项目结构

```text
training/
  model.py / dataset.py / common.py      # 原模型/数据加载，最小兼容修改
  prepare_other_dataset.py               # 保留 V2 自动 easy 数据准备
  prepare_hard_ood.py                    # 固定版本/身份划分的真实 hard 数据
  mine_hard_negatives.py / train_ood.py   # train-only mining、真实 OE
  calibrate_ood.py / export_ood.py / verify_ood_onnx.py
  compare_mobilenet.py
  ood/{core,extract,evaluate,audit,report,select_model,mobilenet}.py
checkpoints/
  binary/ / v2/                         # 冻结旧权重与 ONNX/config
  best_model_ood.pth / mobilenet_ood.pth  # 真实新实验
web/
  index.html / style.css / app.js / preprocess.js / ood.js
  model.onnx / model_config.json / ood_config.json
  assets/vendor/onnxruntime-web/         # 本地固定运行时、许可、哈希
results/
  binary/ / v2/                         # 历史结果
  v3/
    baseline.json / data_audit.json / hard_ood_manifest.json
    production_selection.json / metrics.json
    phase_a/ / phase_b/ / mobilenet/     # 校准、Gaussian参数、独立test消融
    method_comparison.{json,csv} / comparison.md
    validation_operating_points.csv / ood_scores.csv
    mahalanobis_distribution.png / energy_distribution.png / embedding_pca.png
    hard_ood_analysis.json / hard_negative_mining.json
    oe_training_config.json / oe_training_history.json
    onnx_verification.json / browser_verification.json / runtime_comparison.json
scripts/reproduce.sh / verify_ood_browser.py / benchmark_ood.py / check_artifacts.py
```

NPZ feature 缓存和图片被 gitignore，可由相同权重/manifest 重建；保留 feature 行清单与权重哈希。PCA 图只作辅助可视化，不作为成功证据。

## 本地运行（无需训练）

```bash
git clone https://github.com/wls6666/nailong-or-naiwa-.git
cd nailong-or-naiwa-
python -m http.server 8000 --directory web
```

打开 **http://localhost:8000**，不要双击 `file://` HTML。网页只需要静态服务器，不需要安装 torch 或运行 Python 预测服务。JS 验证 ONNX/config 哈希、真实 input/output 名称、维度、数值；模型/OOD 参数缺失或不一致时明确报错，禁用识别，不会 fallback。

输入为128×128 RGB、白底透明合成、EXIF 方向、half-pixel bilinear、float32 NCHW、范围0–1，无 normalization。Canvas 只解码原像素，显式 resize 与 Python 一致。ONNX opset17，输入 `input [N,3,128,128]`，输出 `logits [N,2]`、`embedding [N,256]`，均为真实模型输出。前端不再使用三分类/sigmoid 逻辑。

模型与 OOD 配置通过 SHA-256 文件名发布（`model.<sha256>.onnx` / `ood_config.<sha256>.json`），`model_config.json` 指向这两个不可变资源。前端从配置读取路径；入口配置使用 no-store，避免不同部署的同名资源混用。`model.onnx`、`ood_config.json` 保留为导出/验证原件。仅重新打包现有、校验通过的文件可执行 `python training/publish_web_assets.py`，它不会训练或修改参数。Cloudflare `_headers` 随 web 目录一起部署。

OOD 参数文件约1.72MB JSON，浏览器只做固定矩阵计算，不拟合中心或 covariance。所有阈值/矩阵/温度/映射来自 `ood_config.json`。高级信息显示真实分类概率、距离、Energy、异常 margin；异常分数>0代表拒绝，**不是概率**。OOD 主结果没有百分比；接受后显示的是已知两类条件概率，也不保证实际正确率。

上传限 JPEG/PNG/WebP、10MiB、1600万像素，解码前检查头部尺寸，错误均会显示。没有 Service Worker，不承诺关闭服务器后重载仍可使用。

## 从零复现训练与评估

建议 Python3.12，在根目录激活环境。本次实际使用 CPU 和 requirements 的固定版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
git clone https://github.com/CristinXie/NaiwaOrNailong-dataset.git data/NaiwaOrNailong-dataset
git -C data/NaiwaOrNailong-dataset checkout 900b0419d9c0cfb5ef3c12938a2cbf0ae5eb2abb
bash scripts/reproduce.sh
```

Windows 激活 `.venv\Scripts\activate`，可逐条执行脚本中的 Python 命令。源缓存约需4GB，公开图片下载可能耗时；使用固定版本与 manifest 校验，不在下载失败时生成假数据。首次运行 MobileNet 对照会下载官方 ImageNet 权重。复现起点是仓库中冻结的 V2 best checkpoint；需要从最初 V1/V2 重训时见 [V2 README](README_V2.md) 及历史提交 `1af73a0`。`scripts/reproduce_v2.sh` 是旧流程，会覆盖活动 web 模型，请在独立 checkout 使用。

完整脚本顺序：下载/审计 → frozen post-hoc → train-only mining → OE → train/val extraction/calibration → MobileNet 独立对照 → **冻结选择** → 三实验 test extraction/evaluation → 比较报告 → 导出 → 全 test ONNX 验证。任一步失败非零退出。

仅针对当前提交权重重建最终结果，省略训练与对照的快捷命令：

```bash
python training/prepare_other_dataset.py
python training/prepare_hard_ood.py
python training/ood/audit.py
python training/ood/extract.py --splits train val --checkpoint checkpoints/best_model_ood.pth --output results/v3/phase_b_features
python training/calibrate_ood.py --features results/v3/phase_b_features --output results/v3/phase_b
# 使用当前已提交的 production_selection，不根据 test 改选
python training/ood/extract.py --splits test --checkpoint checkpoints/best_model_ood.pth --output results/v3/phase_b_features
python training/ood/evaluate.py --features results/v3/phase_b_features --calibration results/v3/phase_b/calibration.json --output results/v3/phase_b
python training/export_ood.py
python training/verify_ood_onnx.py
```

## 实际 ONNX / 浏览器检查

checker + ONNX Runtime Python 已通过。全部372张 test（两类/Easy/Hard）对比：最大 logits 误差 **1.049e-5**、embedding **2.861e-6**，均小于1e-4，拒识决策完全一致。距离的最大绝对差约0.00117（高维二次型放大浮点微差），最终 OOD margin 最大差1.754e-5。

```bash
python -m pip install -r requirements-dev.txt
python -m playwright install chromium
python scripts/verify_ood_browser.py --chromium ''
# 有系统 Chromium：python scripts/verify_ood_browser.py --chromium /usr/bin/chromium
node scripts/verify_decision.mjs
python scripts/check_artifacts.py
python scripts/verify_ood_feedback.py
python scripts/verify_asset_delivery.py
# 可选两模型实际浏览器计时（默认系统 Chromium）：
python scripts/benchmark_ood.py
```

真实 UI 操作24次（原文件/无损像素），覆盖奶龙、奶蛙、两个 easy 来源、三来源的青蛙与龙角色，Python/JS logits、embedding、距离、Energy、拒识一致。额外检查透明PNG/WebP/EXIF、非法文件、超大文件、模型与OOD配置下载失败、参数损坏、手机无横向溢出；无 console page error，请求全部为本地 GET。`verify_browser.py` / `check_artifacts.py` 也能识别活动 V3 后分派新检查。没有用数学单元检查替代真实模型推理。

## Cloudflare Pages 部署

继续只发布完整 **`web/`**，必须同时更新 `model.onnx`、`model_config.json`、`ood_config.json` 和 JS，保留本地 WASM/runtime 文件。模型1.56MB、OOD JSON1.72MB，单文件均低于 Pages 25MiB 限制。

现有 Git integration：仓库 production branch `main`、framework None、build command 留空、root 保持仓库根；`wrangler.toml` 已设置 `pages_build_output_dir="./web"`。推送后 Cloudflare 自动部署，无新增 Python/Node 服务。不要将 root 和 output 同时设成 web。原域名仍为 **https://nailongornaiwa.pages.dev/**。

已有 GitHub Pages workflow 也发布同一个 web 目录；Cloudflare/Vercel 等静态托管均可使用。ONNX Runtime Web 本地版本1.22.0，单线程无需 COOP/COEP；不依赖 CDN。重新训练后必须重新导出/验收再一起发布，旧哈希的报告不能作为新模型证据。

## 仍然存在的真实限制

- 难例仍会错：本次7张误接受包括卡通猫/猫脸、山羊、戴墨镜黄脸和3D表情；详情 `hard_ood_analysis.json`。没有玩偶、广泛商业卡通、二创/截图的独立大规模测试，未来应使用真实新错误反馈与预留测试补足。
- Easy FAR 变差、ID 误拒和 FPR@95TPR 偏高是真实问题；本次不承诺全面超过 V2。极小 ID validation/test 使尾部阈值不稳定，不能将单个百分比推广到开放世界。
- 编码器仍可能利用图片风格/背景 shortcut。多来源、白底处理、统一 resize、身份划分只是降低风险，不能证明消除来源识别。
- 未作概率校准；OOD score 只是校准阈值 margin。高两类 softmax 仍可能被错误接受或真实 ID 被拒绝。
- 原数据的近重复/来源细节未知，精确哈希无法证明没有近似场景泄漏；按要求保持原始 split。
- 只实际验证当前 Chromium/CPU，尚未实测所有手机/Safari/Firefox；复杂 ICC/解码器可能产生像素差异。Web 全静态可部署，但不能保证所有设备同样快。
