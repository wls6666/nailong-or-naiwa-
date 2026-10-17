# Naiwa or Nailong · 奶龙 / 奶蛙 / 都不是

在已运行的奶龙/奶蛙项目上增量增加真实 **Other / Neither** 类。保留四层小型 CNN、128×128 输入、原有网页布局、ONNX Runtime Web 和 Cloudflare Pages；没有引入前端框架或预测后端。

**浏览器解码图片 → 统一 RGB/resize/CHW → ONNX Runtime Web → `[1,3]` raw logits → 稳定 softmax → 三类映射 → validation 校准拒识规则 → 结果。**

Other 已真实参与训练。页面没有随机预测、文件名分类、颜色/尺寸分类、mock、`/predict` API 或加载失败后的替代结果。上传文件大小/格式/尺寸检查只会报告错误，不会据此输出分类。

仓库：[wls6666/nailong-or-naiwa-](https://github.com/wls6666/nailong-or-naiwa-)；网站：[nailongornaiwa.pages.dev](https://nailongornaiwa.pages.dev/)。

## 真实结果与升级比较

本次使用 CPU、Python 3.12.14、PyTorch 2.14.1+cpu、seed=42，实际训练 40 轮。最佳 checkpoint 为第 34 轮，按验证集 **macro recall** 优先、加权验证 loss 同分择优；未根据 test 结果调整模型、超参数或阈值。

| 指标 | 旧 binary | 新三分类 argmax | 新三分类 + 拒识（网页） |
| --- | ---: | ---: | ---: |
| 原 73 张奶龙/奶蛙 test accuracy | 93.15% (68/73) | 93.15% (68/73) | 93.15% (68/73) |
| Other test recall | 未建立 Other 测试 | 98.00% (147/150) | 98.00% (147/150) |
| Other false accept rate | 不作未执行的指标声明 | 2.00% (3/150) | 2.00% (3/150) |
| 三类 overall accuracy，223 张 | 不适用 | 96.41% | 96.41% |
| Macro precision | 不适用 | 93.82% | 94.81% |
| Macro recall | 不适用 | 94.56% | 94.56% |
| Macro F1 | 不适用 | 94.08% | 94.64% |

校准阈值在本次 test 上**没有进一步降低 Other FAR**；它把两次原本错误的目标类判断改为低置信度拒识，因此目标类正确数量保持不变、Macro F1 略有改善。不将这种变化描述为 Other recall 的提升。

网页最终规则的每类指标：

| 类别 | Precision | Recall | Support |
| --- | ---: | ---: | ---: |
| Nailong | 92.68% | 97.44% | 39 |
| Naiwa | 93.75% | 88.24% | 34 |
| Other | 98.00% | 98.00% | 150 |

最终混淆矩阵，行是真实、列是预测，顺序为 `[nailong,naiwa,other]`：

```text
             Nailong   Naiwa   Other
Nailong         38       1       0
Naiwa            1      30       3
Other            2       1     147
```

指标来自实际推理：`results/metrics_3class.json`（原始三分类）、`results/metrics_3class_calibrated.json`（网页最终规则）、逐图预测 JSON 和对应混淆矩阵。升级比较另存 `results/upgrade_comparison.json`。

旧 binary 的原始 `results/metrics.json` 等报告未覆盖；另在 `results/binary/` 保存完整快照，权重/ONNX/config 在 `checkpoints/binary/`。原二分类代码可查看 Git 提交 `b96b626`。现有训练脚本默认训练三分类。

## 项目结构

```text
naiwa-or-nailong/
├── README.md
├── requirements.txt / requirements-dev.txt
├── .gitignore
├── wrangler.toml                      # Pages 发布 ./web
├── .github/workflows/pages.yml        # 可选 GitHub Pages
├── data/                              # 数据不提交、不打包
│   ├── NaiwaOrNailong-dataset/data/
│   │   ├── train/{nailong,naiwa,other}/
│   │   ├── val/{nailong,naiwa,other}/
│   │   └── test/{nailong,naiwa,other}/
│   ├── other_manifest.json
│   ├── source_cache/                  # 公开数据下载缓存
│   ├── hard_negative/.gitkeep         # 可选训练负样本
│   └── false_positive/.gitkeep        # 可选误接受反馈
├── training/
│   ├── prepare_other_dataset.py
│   ├── dataset.py / model.py / common.py
│   ├── train.py / inference.py
│   ├── evaluate.py / plot_history.py
│   ├── calibrate_rejection.py
│   ├── export_onnx.py / verify_onnx.py
│   └── evaluate_extra_other.py
├── checkpoints/
│   ├── best_model_3class.pth           # 本次真实训练
│   ├── best_model.pth                 # 保留旧 binary
│   └── binary/                        # 旧权重、ONNX、config
├── web/
│   ├── index.html / style.css
│   ├── app.js / preprocess.js
│   ├── model.onnx / model_config.json
│   └── assets/vendor/onnxruntime-web/  # 固定版本运行时及许可声明
├── scripts/
│   ├── reproduce.sh / download_runtime.py
│   ├── verify_other_data.py
│   ├── verify_browser.py / verify_decision.mjs
│   └── check_artifacts.py
└── results/
    ├── binary/                        # 旧实验快照
    ├── other_manifest.json            # 可提交的构建清单，不含图片
    ├── other_dataset_summary.json / other_data_verification.json
    ├── hard_negative_manifest.json / feedback_verification.json
    ├── dataset_audit_3class.json / class_mapping.json
    ├── training_config_3class.json / training_history_3class.json
    ├── training_curves_3class.png
    ├── metrics_3class.json / metrics_3class_calibrated.json
    ├── test_predictions_3class*.json / confusion_matrix_3class*.png
    ├── rejection_calibration.json / validation_predictions_3class.json
    ├── onnx_export_3class.json / onnx_verification_3class.json
    ├── browser_verification_3class.json / web_*_3class.png
    ├── extra_other_probes.json / upgrade_comparison.json
    └── *.log                          # 实际执行记录
```

## 只运行静态网页

网页已包含训练好的 ONNX 和本地 ONNX Runtime Web，无需先安装训练依赖或下载数据：

```bash
git clone https://github.com/wls6666/nailong-or-naiwa-.git
cd nailong-or-naiwa-
python -m http.server 8000 --directory web
```

打开 **http://localhost:8000**，等待“模型已就绪”，上传 JPEG/PNG/WebP，点击识别。也可 `cd web` 后执行 `python -m http.server 8000`。不要双击 HTML 使用 `file://`。

HTTP server 只提供静态文件，不执行模型推理。网页不访问 CDN、数据库或预测服务；WASM 单线程不需要 COOP/COEP。首次需下载约 13 MB 静态资源。没有 Service Worker，不承诺停止文件服务器后重新打开页面仍可离线访问。

## 从零准备环境

以下命令在项目根目录执行，建议 Python 3.12。CPU 复现：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Windows 激活命令为 `.venv\Scripts\activate`。GPU 使用者应安装与本机 CUDA 匹配的 PyTorch；本次只实际验证 CPU。

## 自动准备 Other，保留原划分

原数据：[CristinXie/NaiwaOrNailong-dataset](https://github.com/CristinXie/NaiwaOrNailong-dataset)，提交 `900b0419d9c0cfb5ef3c12938a2cbf0ae5eb2abb`。已阅读原 README，没有重新划分或移动原图。

```bash
# 若目录已经存在，跳过 clone
git clone https://github.com/CristinXie/NaiwaOrNailong-dataset.git data/NaiwaOrNailong-dataset
git -C data/NaiwaOrNailong-dataset checkout 900b0419d9c0cfb5ef3c12938a2cbf0ae5eb2abb
python training/prepare_other_dataset.py
python scripts/verify_other_data.py
python training/dataset.py
```

`prepare_other_dataset.py` 自动下载、固定 seed=42、按来源类别交错抽样、生成 RGB JPEG（quality=95，subsampling=0，不加黑边或标签文字），再与所有类别共用同一 tensor 预处理。保留原生尺寸，不把所有 Other 统一降成 32×32。

| Split | Nailong（未改动） | Naiwa（未改动） | Other | 总计 |
| --- | ---: | ---: | ---: | ---: |
| train | 184 | 161 | 600 | 945 |
| val | 39 | 34 | 150 | 223 |
| test | 39 | 34 | 150 | 223 |

每个 split 的 Other 一半来自 CIFAR-100、一半来自 Oxford-IIIT Pet：

- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)：物体、食物、交通工具、自然场景、人物等。官方源在本次环境下载较慢，脚本明确使用 fast.ai 的公开 [S3 镜像](https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz)，保留官方 50,000 train / 10,000 test 池。镜像目录按 20 个 coarse 类组织，source ID 中仍保留 fine 类和原图片文件名；本次选中的图片共覆盖 99 个 fine 类。下载的 archive 固定 SHA-256 为 `085ac613ceb0b3659c8072143ae553d5dd146b3c4206c3672a56ed02d0e77d28`，校验通过后才解压。
- [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)：37 种猫狗，提供较高分辨率图片，避免 Other 全部是 CIFAR 风格。由 torchvision 下载并验证官方 MD5。

源 train/trainval 池只用于 train/val，源 test 池只用于 test；train/val 从同一个打乱后的列表依次取不同索引。脚本检查来源身份、文件哈希、解码后的 RGB 像素哈希，并与原始目标图片比对。原始 491 张图片的 SHA-256 也会前后核验。独立验证结果：**900 个 Other，491 张原图，跨 split 字节/像素/来源 ID 重复均为 0**。

Manifest 同时保存 `data/other_manifest.json` 和可提交的 `results/other_manifest.json`，记录来源、官方池、索引、原 ID、源类别、目标 split、最终路径、原生尺寸、源像素/最终像素/文件 SHA-256 及原始目标图快照。再次运行会校验现有文件，避免重复下载。需要主动重建生成的 Other 时：

```bash
python training/prepare_other_dataset.py --rebuild
```

`--rebuild` 只删除并重建三个 split 的 `other/`，不删除目标类。下载缓存保留，预留约 4 GB 磁盘空间。`--train/--val/--test` 可改数量；更改数据后必须重新训练、校准和验收。

CIFAR 页面要求引用 Krizhevsky 2009，但没有为全部原图提供明确的统一授权；不要把它误写成 MIT 图片许可。Oxford Pets 明确使用 CC BY-SA 4.0，版权归原图权利人，引用 Parkhi et al., *Cats and Dogs*, CVPR 2012。原奶龙/奶蛙数据也未附可核实的逐图授权。本项目只保存下载脚本和清单，不向自己的 Git 仓库或压缩包重新分发这些图片；商业使用和图片再分发需核查相应权利。

## 模型与训练

架构沿用原项目，共 **390,147** 个参数，只改变最后一层为 `Linear(256,3)`：

```text
[N,3,128,128]
Conv 3→32, 3×3/pad1 → BN → ReLU → MaxPool2
Conv 32→64, 3×3/pad1 → BN → ReLU → MaxPool2
Conv 64→128, 3×3/pad1 → BN → ReLU → MaxPool2
Conv 128→256, 3×3/pad1 → BN → ReLU
AdaptiveAvgPool(1,1) → Flatten → Dropout(0.35) → Linear 256→3
[N,3] raw logits
```

没有在 forward 中加 softmax，没有换成预训练 ResNet/ViT。默认复用**本项目自己训练的**旧二分类 CNN backbone，并用原 log-odds 初始化两个目标输出；Other 输出行随后与整网一同真实训练。`--from-scratch` 可关闭此初始化；从零模式的指标不等同于本次默认模式。

实际 ImageFolder 映射为 `{"nailong":0,"naiwa":1,"other":2}`，但所有阶段从 Dataset/checkpoint/config 读取并核对映射。映射保存在 `results/class_mapping.json`、权重、ONNX metadata 和网页配置。

使用一种不平衡处理：**class-weighted CrossEntropyLoss**，权重为 `N/(3*class_count)`，本次分别约 `[1.7119565,1.9565217,0.525]`，不再叠加 WeightedRandomSampler。batch=32、AdamW、lr=3e-4、weight_decay=1e-4、Dropout=0.35、ReduceLROnPlateau(patience=4,factor=0.5)、early stopping patience=12、最多 40 轮。

```bash
python training/train.py --epochs 40 --seed 42 --threads 2
python training/evaluate.py
python training/calibrate_rejection.py
python training/evaluate.py --calibrated
python training/plot_history.py
```

记录每轮 train/val loss、accuracy、各类 recall。best checkpoint 按 val macro recall、再按加权 val loss 选择，不能使用 test accuracy 选择。原始三分类 test 指标先保存，阈值校准完成后再独立报告最终规则指标，两者都保留。

随机种子覆盖 Python、NumPy、PyTorch；DataLoader 使用固定 generator、num_workers=0。跨硬件/库版本可能有浮点差异，不承诺 GPU 完全 deterministic。

## 可持续添加 hard negatives

未来把真实图片放入以下任一目录，可以包含子目录：

```text
data/hard_negative/     # 其他黄色卡通、青蛙、恐龙、玩偶、相似画风等
data/false_positive/    # 网页误判为目标类的真实错误案例
```

训练脚本自动读取其中 JPEG/PNG/WebP/BMP，加入 **Other train**；不加入 val/test。目录不存在时正常训练，不报错。本次目录为空，正式训练额外样本数为 0，没有要求用户手工收集负样本。

若反馈图与 val/test 的已知源图片或解码像素完全重复，会明确报错阻止训练；与已有 train 重复则跳过，记录最终使用列表到 `results/hard_negative_manifest.json`。已实际检查“新图片自动加入”和“测试图误放反馈目录被拒绝”，见 `results/feedback_verification.json`。文件名只用于标识数据，不参与模型预测。

添加图片后，重新运行训练及以下校准/导出/浏览器验证阶段，不需要重新下载公共数据。不要把独立测试集错误图直接搬进训练集再宣称同一个 test 的无泄漏成绩；反馈应来自新的真实使用案例。

## 阈值如何选择及 confidence 语义

`calibrate_rejection.py` **只实例化 val Dataset**，不读取 test tensor 或 test predictions。扫描 `[0.50,0.55,...,0.95]`，记录每个候选的目标类准确率、每类 recall、Other FAR 和 Macro F1。

预先固定选择规则：相对原始三分类 val，目标准确率下降不超过 3 个百分点，且任一目标类 recall 下降不超过 5 个百分点；在合格候选里优先最小 Other FAR，再最大 Macro F1，再选较低阈值。若无候选满足约束，先最小化目标损失，并明确记录约束不满足。本次约束满足，选中 **0.55**，val 目标准确率没有下降。

完整扫描保存在 `results/rejection_calibration.json`。导出时会核对 checkpoint SHA-256，并将阈值和校准报告 SHA-256 写入 `web/model_config.json`，不是凭感觉在 JS 中写常量。

最终逻辑：

1. 稳定 softmax：先减最大 logit，再 exp 和归一化。
2. raw argmax 为 Other：输出 **都不是 / Neither**，confidence=`P_other`。
3. raw argmax 为目标类，且 `max(P_nailong,P_naiwa)<reject_threshold`：输出 **都不是 / 无法确定**，显示“最高目标类概率仅 xx%”，该数值不是 `P_other`。
4. 否则输出对应目标类，confidence 为该类别 softmax 概率。

浏览器已实际验证一个低置信度样本：`[P_nailong,P_naiwa,P_other]=[51.08%,24.94%,23.98%]`，页面显示“无法确定”和“最高目标类概率 51.08%”，不会编造更高的 Other confidence。

## 预处理保持原实现

所有类统一为 EXIF 方向处理、RGB（透明图合成白底）、128×128 拉伸、half-pixel bilinear、float32、NCHW、像素范围 0–1，**无 ImageNet 或其他 normalization**。Canvas 只解码原尺寸像素，缩放使用与 Python 相同的显式双线性算法，避免默认缩放滤波差异。

训练对缓存的同一预处理 tensor 做轻度水平翻转、±8° 旋转和 ColorJitter；val/test 无随机增强。保存 Other 时没有标签文字、水印叠加、黑边或 AI 生成内容；多来源和统一处理只能降低来源 shortcut 风险，不能证明模型完全没有学习数据集风格。

## ONNX 导出与验证

```bash
python training/export_onnx.py
python training/verify_onnx.py
```

导出 opset=17，输入 `input`：`[N,3,128,128]`，输出 `logits`：**`[N,3]`**。batch 动态，网页用单张。模型本身输出 raw logits，不包含前端 softmax 或阈值规则。

导出执行 `onnx.checker.check_model(full_check=True)`，随后 ONNX Runtime Python 在真实图片上推理并验证 `[1,3]` 和有限数值。验证默认覆盖 **全部 223 张 test 图片**，包含三类；要求 PyTorch/ONNX 最大 logit 和概率绝对差异均小于 1e-4，原始分类及校准分类一致。`--limit 9` 可进行按类别交错的较小检查。

本次 checker、Python runtime 均 PASS；最大 logit 差异 **6.68e-6**，最大概率差异 **7.00e-7**。完整结果见 `results/onnx_verification_3class.json`。

模型、映射、阈值必须成对发布。重新训练后，旧报告的 checkpoint/model/calibration 哈希不匹配时不能作为新版本验收证据。

## 浏览器实际验证

```bash
python -m pip install -r requirements-dev.txt
python -m playwright install chromium
python scripts/verify_browser.py --chromium ''
# Linux 已安装系统 Chromium 也可以：
# python scripts/verify_browser.py --chromium /usr/bin/chromium
node scripts/verify_decision.mjs
python scripts/check_artifacts.py
```

浏览器脚本使用临时静态 HTTP server 和真实 Chromium，操作文件选择和识别按钮。覆盖每个目标类、两个来源的 Other，以及 validation 中一张真正触发低置信度拒识的图片；每张检查原始 JPEG 和无损 PNG，共 **14 次数值对比**，此外保留透明 PNG、WebP、EXIF、非法文件、10 MB 上限、模型下载失败和手机布局检查。

本次最大浏览器/Python logit 差异 **1.43e-6**，所有检查样本的预处理 tensor 最大差异 **0**。两个实际分支（raw Other argmax / threshold rejection）均有真实图片验证。模型失败时按钮不可用且无预测结果；请求只包含本地 GET 静态资源和本地 blob。

`verify_decision.mjs` 将 223 条真实 validation logits 与 Python softmax 结果比较，同时打乱类别顺序验证 mapping 不依赖固定索引，并测试极大 logits 的数值稳定性和非有限值错误。数值边界单元检查不替代真实模型推理。

前端继续限制 JPEG/PNG/WebP、最大 10 MiB / 1600 万像素，解码前检查图片头尺寸；错误会显示而不会变成默认分类。运行时保持本地 `onnxruntime-web@1.22.0`，包含版本、哈希及 MIT/第三方声明。如需重新获取运行时：`python scripts/download_runtime.py`。

## 额外真实乱入图片

```bash
python training/evaluate_extra_other.py
```

从两个源数据集**未被选入任何 train/val/test 的官方 test 池样本**中，固定 seed 取猫、狗、车辆、食物、建筑、人物、自然各 2 张，共 14 张；本次误接受 **0/14**。不额外爬站、不以 AI 生成图片代替，报告见 `results/extra_other_probes.json`。

这些仍来自已使用的数据源，属于独立样本的定性检查，**不是新的未见数据域 benchmark**，不据此保证所有乱入图片都能拒绝。

## 一次运行完整复现链路

下载原奶龙/奶蛙仓库并激活虚拟环境后：

```bash
bash scripts/reproduce.sh
```

依次执行 Other 准备、数据核验、训练、raw test、val calibration、calibrated test、图表、导出、全 test ONNX 对比和额外图片检查。浏览器验证按上面的可选依赖命令单独执行。所有主要阶段失败均以非零退出码停止后续步骤。

## Cloudflare Pages 与其他静态部署

当前项目继续只发布完整 **`web/`**，保留 `.onnx/.json/.js/.mjs/.wasm`。`wrangler.toml` 已声明 `name="nailongornaiwa"`、`pages_build_output_dir="./web"`。

Cloudflare Pages 设置：连接当前 GitHub 仓库，production branch=`main`、framework=None、build command 留空，root directory 保持仓库根目录；输出目录由配置设为 `web`。不要同时将 root 和 output 都填成 `web`。推送 main 后由已连接的 Cloudflare Git integration 自动部署。

如果部署成功但首页 404，检查日志是否读取了 wrangler 配置、发布目录是否包含 `index.html`，以及生产分支是否为 main。原先已工作的 Cloudflare 托管方式和目录没有改变。

GitHub Pages 也兼容：仓库 Settings → Pages → Source 选择 GitHub Actions，使用已有 workflow。Cloudflare 已连接时，不需要依赖 GitHub Pages workflow 的运行结果。Vercel 等静态服务也只需发布 `web/`。没有 Python/Node 推理服务或数据库。

## 真实限制

- 原目标类训练图只有 345 张，验证/测试各 73 张；此次目标测试正确数量保持 68/73，不等于已经证明泛化能力没有下降。
- Other 只是 600 张有限、多来源负样本。测试 FAR=2% 指这个构建的 Other test，不能推广为现实世界任意图片的 FAR。仍有 3/150 Other 被误接受。
- 多数 Other 是照片，相似黄色卡通、其他青蛙/恐龙、玩偶和二创画风覆盖不足，需靠真实 hard-negative 反馈逐步补充。
- 字节、来源 ID 和精确解码像素去重不保证所有近似场景均被排除；原目标数据的原始划分算法/逐图来源也未知，原 split 按要求保持。
- softmax confidence 未做概率校准；拒识阈值校准不等于 confidence 已校准。高概率仍可能错误，低置信度拒绝也可能拒绝真实目标图。
- 极端长宽比会被拉伸；特殊 ICC、透明度、编解码器和浏览器差异可能导致像素差异。本次已实测 Chromium，未覆盖所有 Safari/Firefox/手机机型。
- 新 ONNX 仍使用经过实际验证的 TorchScript export 路径（`dynamo=False`），当前 PyTorch 会显示弃用提示；checker/Python/WASM 实际执行均已通过。
