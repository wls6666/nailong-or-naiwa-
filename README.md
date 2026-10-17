# Naiwa or Nailong · 奶龙还是奶蛙？

一个已真实训练、真实测试并通过浏览器验证的小型 CNN 图片二分类项目。前端为纯静态 HTML / CSS / JavaScript，图片在设备内处理，不上传到预测服务器。

**推理链路：浏览器图片解码 → JavaScript RGB/resize/CHW → ONNX Runtime Web WASM → `model.onnx` → raw logit → sigmoid → 类别和对应概率。**

没有 `/predict` 接口、Python/Node 推理后端、随机置信度或默认预测。模型或推理失败会明确显示错误。`web/` 自带训练好的模型和固定版本 ONNX Runtime Web，运行网页不需要安装训练依赖、npm 或访问 CDN。

## 交付内容

```text
naiwa-or-nailong/
├── README.md
├── requirements.txt
├── requirements-dev.txt               # 仅可选浏览器验证
├── .gitignore
├── wrangler.toml                      # Cloudflare Pages 发布 web/
├── .github/workflows/pages.yml        # 直接部署 web/ 到 GitHub Pages
├── data/                             # 本地下载，Git 和交付压缩包均排除
│   └── NaiwaOrNailong-dataset/
│       ├── README.md
│       └── data/
│           ├── train/{nailong,naiwa}/
│           ├── val/{nailong,naiwa}/
│           └── test/{nailong,naiwa}/
├── training/
│   ├── dataset.py                    # 发现原划分、审核图片、预处理、Dataset
│   ├── model.py                      # 从零实现的 SmallCNN
│   ├── common.py                     # 随机种子、权重加载、文件哈希
│   ├── train.py
│   ├── evaluate.py
│   ├── plot_history.py
│   ├── export_onnx.py
│   └── verify_onnx.py
├── checkpoints/best_model.pth         # 本次真实训练得到，已保留
├── web/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   ├── preprocess.js
│   ├── model.onnx                    # 1,557,765 bytes，训练权重导出
│   ├── model_config.json             # 类别、预处理、I/O、模型/权重哈希
│   └── assets/vendor/onnxruntime-web/ # JS、MJS、WASM、版本、许可证
├── scripts/
│   ├── download_runtime.py
│   ├── reproduce.sh
│   ├── verify_browser.py
│   └── check_artifacts.py
└── results/
    ├── dataset_audit.json
    ├── class_mapping.json
    ├── environment.json
    ├── training_config.json
    ├── training_history.json
    ├── training_curves.png
    ├── metrics.json
    ├── test_predictions.json
    ├── confusion_matrix.png
    ├── onnx_export.json
    ├── onnx_verification.json
    ├── browser_verification.json
    ├── web_desktop.png
    ├── web_mobile.png
    └── *.log                         # 实际命令输出
```

## 已实际运行的结果

在 Linux、Python 3.12.14、PyTorch 2.14.1+cpu、2 个 CPU 线程上，seed=42，训练 40 个 epoch；未使用 GPU、预训练权重或测试集调参。逐轮记录见 `results/train.log` 和 `results/training_history.json`。

| 项目 | 实际结果 |
| --- | --- |
| 图片解码 | 491 / 491 成功；没有字节完全重复图片 |
| 最佳 checkpoint | epoch 34；val accuracy **94.52%**（69/73），val loss 0.18311 |
| 最佳 epoch 的 train accuracy/loss | 95.65% / 0.11675，使用随机增强和 Dropout 的训练模式 |
| Test accuracy | **93.15%**（68/73） |
| Test precision | **96.77%** |
| Test recall | **88.24%** |
| Test F1 | **92.31%** |
| ONNX checker / Python Runtime | PASS / PASS |
| PyTorch vs ONNX，全部 73 张 test 图 | PASS，最大 logit 差异 **5.72e-6**，最大 sigmoid 概率差异 **2.06e-7** |
| Chromium vs Python，同 tensor，8 次真实 UI 推理 | PASS，最大 logit 差异 **1.91e-6** |
| 浏览器 vs Python 预处理 | 检查的 4 张真实 JPEG 及对应无损 PNG，tensor 最大差异 **0** |

Precision/Recall/F1 为二分类指标，正类是 **奶蛙 / naiwa（1）**。混淆矩阵按 `[nailong, naiwa]` 排序，行是真实类别，列是预测类别：

```text
                  预测奶龙   预测奶蛙
真实奶龙              38         1
真实奶蛙               4        30
```

浏览器验证只验证链路和数值一致性，并不要求每张图分类正确；其中一张选取的奶龙测试图被模型高置信度误判为奶蛙，这个真实错误也保留在报告中。不能把数值一致性 PASS 当成分类准确率 100%。

## 只运行网页（不需要训练环境）

从 GitHub 获取项目：

```bash
git clone https://github.com/wls6666/nailong-or-naiwa-.git
cd nailong-or-naiwa-
```

在项目根目录执行：

```bash
cd web
python -m http.server 8000
```

打开 **http://localhost:8000**，等待“模型已就绪”，上传 JPEG/PNG/WebP，点击“开始识别”。不要直接双击 `index.html` 或使用 `file://`，浏览器需要通过 HTTP 加载 ONNX、ES module 和 WASM。

也可以保持项目根目录：

```bash
python -m http.server 8000 --directory web
```

这个命令只是静态文件服务器，没有预测 API；关闭联网后，服务器仍可从本地文件向浏览器提供全部资源。项目不包含 Service Worker，因此不承诺关闭静态服务器后重新打开页面仍能离线访问。

## 从零复现：环境与数据

以下命令均在项目根目录执行，除非明确写出 `cd web`。建议使用 Python 3.12。CPU 复现方式：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Windows 的激活命令是 `.venv\Scripts\activate`。如需 CUDA，请先按 PyTorch 官方安装说明安装与本机匹配的版本；本次交付仅实际验证了 CPU 训练。

数据来源：[CristinXie/NaiwaOrNailong-dataset](https://github.com/CristinXie/NaiwaOrNailong-dataset)。已阅读其 README 并保留原划分，使用的提交为 `900b0419d9c0cfb5ef3c12938a2cbf0ae5eb2abb`。

```bash
git clone https://github.com/CristinXie/NaiwaOrNailong-dataset.git data/NaiwaOrNailong-dataset
git -C data/NaiwaOrNailong-dataset checkout 900b0419d9c0cfb5ef3c12938a2cbf0ae5eb2abb
python training/dataset.py
```

若目录已存在则跳过 clone。实际目录和数量：

| 原划分 | nailong | naiwa | 总计 |
| --- | ---: | ---: | ---: |
| `data/NaiwaOrNailong-dataset/data/train` | 184 | 161 | 345 |
| `data/NaiwaOrNailong-dataset/data/val` | 39 | 34 | 73 |
| `data/NaiwaOrNailong-dataset/data/test` | 39 | 34 | 73 |

`dataset.py` 会检测传入目录本身、`data/` 或子仓库的 `data/`，适配 `val/validation/valid`；找不到唯一完整划分就报错，不会偷偷重新划分。自定义路径可传 `--data /absolute/path/to/dataset`，训练、测试和验证命令应使用同一路径。

数据集没有附可核实的逐图授权或数据集许可证。本项目不改变原图、不把数据集重新提交或打包，也不为原图片授予新的使用权限；使用或再分发图片前应确认所需授权。

## CNN 架构、预处理与类别

模型从随机初始化开始训练，共 **389,633** 个参数：

```text
[N,3,128,128]
Conv 3→32, 3×3/pad1 → BatchNorm → ReLU → MaxPool2
Conv 32→64, 3×3/pad1 → BatchNorm → ReLU → MaxPool2
Conv 64→128, 3×3/pad1 → BatchNorm → ReLU → MaxPool2
Conv 128→256, 3×3/pad1 → BatchNorm → ReLU
AdaptiveAvgPool(1,1) → Flatten → Dropout(0.35) → Linear 256→1
[N,1] raw logits
```

`forward` 中无 sigmoid，训练用 `BCEWithLogitsLoss`。优化器 AdamW，lr=1e-3、weight_decay=1e-4、batch size=32；验证 loss 连续 4 轮没有改善时 ReduceLROnPlateau 将学习率减半；checkpoint 优先比较 val accuracy，再以 val loss 较小者胜出。连续 12 轮未改善时 early stopping。本次实际执行满 40 轮，保留第 34 轮。

类别映射不是手工猜测，来自实际 ImageFolder 读取，并在所有划分间核对：

```json
{"nailong": 0, "naiwa": 1}
```

映射同时保存在 checkpoint、`results/class_mapping.json`、ONNX metadata 和 `web/model_config.json`；网页读取并验证配置，不根据文件名判断。

Python 测试和 JavaScript 的关键预处理完全相同：

1. 应用 EXIF 方向，解码为 RGB；透明图片合成到白底。
2. 拉伸到 128×128，不做随机裁剪、不保留长宽比。
3. 用显式 half-pixel 双线性插值：`s=max(0,(d+0.5)*sourceSize/128-0.5)`，相邻索引在边缘 clamp；中间不量化为 uint8。
4. RGB 三通道，除以 255，范围 0–1；**不使用 ImageNet 或其他 normalization**。
5. 转为连续 float32，按 R 平面、G 平面、B 平面排列，shape `[1,3,128,128]`。

浏览器 Canvas 只负责原尺寸图片解码与 RGB 提取，缩放由 `pixelsToCHW` 执行，避免 Canvas/Pillow 各自默认缩放滤波差异。训练缓存同一确定性预处理后的 tensor，再做轻度随机水平翻转、±8° 旋转及亮度/对比度/饱和度变化。验证、测试没有随机增强。

## 训练、测试、导出和一致性验证

激活虚拟环境并下载数据后：

```bash
python training/train.py --epochs 40 --batch-size 32 --seed 42 --threads 2
python training/evaluate.py
python training/plot_history.py
python training/export_onnx.py
python training/verify_onnx.py
```

或者顺序执行全部阶段：

```bash
bash scripts/reproduce.sh
```

训练会打印每轮 train/val loss 与 accuracy，保存最佳 `checkpoints/best_model.pth`、类别映射、配置和完整 epoch 历史。测试重新加载最佳权重，对原 test 划分执行真实 inference，产生 `metrics.json`、逐图预测及混淆矩阵。

导出使用 opset 17，输入名 **`input`**，输出名 **`logits`**，batch 维动态；网页使用单张输入 `[1,3,128,128]`，输出 `[1,1]`。导出执行 `onnx.checker.check_model(full_check=True)`，然后用 ONNX Runtime Python 在真实测试图片上运行。`verify_onnx.py` 默认检查完整 test set，同一 tensor 分别交给 PyTorch 与 ONNX，要求 logit 和 sigmoid 概率差异均小于 1e-4，分类也须一致；失败退出码非零。

```bash
# 可只选前 6 张，但完整验收建议保持默认的全部 73 张
python training/verify_onnx.py --limit 6
```

导出会同时更新 `web/model.onnx` 与 `web/model_config.json`。两者必须成对部署。重新训练后要重新导出、验证和浏览器验证；旧报告的模型哈希不匹配时不能继续作为新模型的验收证据。

随机种子覆盖 Python、NumPy、PyTorch。DataLoader 使用固定种子 generator、num_workers=0，并关闭 cuDNN benchmark；不同硬件或库版本仍可能产生不同浮点结果，不承诺 GPU 完全 deterministic。

## 浏览器端真实推理与验证

`app.js` 真实调用 `ort.InferenceSession.create`，对照实际 `session.inputNames/outputNames` 与模型配置，再执行：

```javascript
const tensor = new ort.Tensor('float32', data, [1, 3, 128, 128]);
const outputs = await session.run({ [config.input_name]: tensor });
const logit = Number(outputs[config.output_name].data[0]);
```

`p_positive=sigmoid(logit)`。映射索引 1 的类别概率是 `p_positive`，索引 0 的概率是 `1-p_positive`，选择概率较大的类别，并显示它自己的概率。当前映射下 `p_naiwa=p_positive`。前端检查有限数值和输出 shape，任何异常都会隐藏结果并报告错误，不返回替代预测。

支持点击/拖放、预览、加载状态和重试、真实分析状态、结果和置信度条。文件上限 10 MiB；解码前检查 JPEG/PNG/WebP 文件头和尺寸，上限 1600 万像素、单边 16384 像素，降低超大图片占用风险。上传期间不允许并发推理，切换图片会清除旧结果。Console 中可查看实际 shape、logit、概率、类别。

ONNX Runtime Web 1.22.0 使用 WASM 单线程，不需要 COOP/COEP，也不依赖 WebGPU。所有运行时文件均在 `web/assets/vendor/onnxruntime-web/`，包含版本、SHA-256、MIT License 和第三方声明。重建这些静态依赖（需要联网）可执行：

```bash
python scripts/download_runtime.py
```

该脚本校验 npm tarball 的 SHA-512 integrity，再提取固定文件，并从相同上游版本补齐许可声明；不运行 npm 包中的代码。

可选：复现本次真实 Chromium 自动化验收：

```bash
python -m pip install -r requirements-dev.txt
python -m playwright install chromium
python scripts/verify_browser.py --chromium ''
# 已安装系统 Chromium 的 Linux 可使用：
# python scripts/verify_browser.py --chromium /usr/bin/chromium
python scripts/check_artifacts.py
```

浏览器验证脚本临时启动一个只提供文件的本地 HTTP server，启动真实 Chromium，操作上传控件和识别按钮，对 4 张测试图的 JPEG/PNG 共 8 次输出与 Python ONNX/PyTorch 对比，并比较浏览器与 Python tensor。另检查透明 PNG、WebP、EXIF、非法文件、10 MB 上限、模型加载失败、手机宽度溢出和只访问本地 GET 资源。结束后自动关闭浏览器和服务器，写出报告和截图。若数据、模型或浏览器环境不同，应以重新产生的报告为准。

## 静态部署

只部署完整 **`web/`**，不部署训练代码、数据集或 checkpoint。所有路径均为相对路径，兼容 GitHub Pages 的仓库子路径。保留 `.js/.mjs/.wasm/.onnx/.json`，不要把模型请求重写成 HTML。

**GitHub Pages（已提供配置）**：

1. 把本项目提交到你自己的 GitHub 仓库，保留 `web/model.onnx`、配置与 vendor 文件。`.gitignore` 已排除数据、虚拟环境和缓存，保留约 1.5 MiB 的训练权重和 ONNX。
2. 仓库 Settings → Pages → Source 选择 **GitHub Actions**。
3. 推送到 `main`，或在 Actions 中手动运行 `Deploy static CNN classifier to GitHub Pages`。
4. workflow 直接上传 `web/` 为 Pages artifact，无训练、无 npm 构建、无后端。完成后访问 workflow 返回的 Pages URL。

**Cloudflare Pages**：仓库已包含 `wrangler.toml`，明确设置 `pages_build_output_dir = "./web"`。连接仓库后，生产分支选择 `main`、框架选择 None、构建命令留空、根目录保持仓库根目录（留空或 `/`）。输出目录由配置文件设为 `web`，确保 `web/index.html` 成为线上站点的 `/index.html`。也可使用 Direct Upload 上传 `web/` 的内容。

若部署显示成功但根网址返回 404，请在 Cloudflare 项目 Settings → Builds 中核对根目录仍是仓库根目录，且部署日志已读取 `wrangler.toml`、发布目录为 `web`，然后重新部署最新提交。不要把 `web` 同时填入根目录和输出目录，否则会变成查找 `web/web`。配置文件的 `name` 对应 Pages 项目 `nailongornaiwa`；另建项目时应修改这个名称。

**Vercel**：选择 Other，Root Directory=`web`，关闭框架构建，Output Directory=`.`。也可将 `web/` 直接交给其他能提供静态文件的托管服务。

项目仓库：https://github.com/wls6666/nailong-or-naiwa- 。已通过本地静态 HTTP + Chromium 实测。上传代码不代表已启用线上托管；`pages.yml` 需要先在该仓库 Settings → Pages 中选择 GitHub Actions，才可正常部署。Cloudflare/Vercel 也需要另行配置托管。

## 已知限制与排错

- 训练数据仅 345 张，验证/测试各 73 张。验证/测试准确率不能完全代表真实泛化能力；字节去重不等于感知去重，不排除不同划分有近似场景。
- 仅支持两类，其他角色、多人/多角色、遮挡或背景复杂图片也会被强制二选一。高概率可能错误，概率未经过校准。
- 图片被拉伸到正方形，极端长宽比、特别小的主体和新画风可能降低效果。
- 实测样本的 Python/Chromium 解码与预处理一致；其他浏览器、特殊 ICC 色彩配置、透明度和编解码器版本可能有少量像素差异。当前实现禁用 ImageBitmap 额外色彩转换并明确白底处理，核心缩放和张量布局统一。
- 浏览器必须支持 WebAssembly、ES modules、Canvas、createImageBitmap；本次已实测 Chromium，未在 Safari/Firefox 和所有手机机型上逐一验证。首次要下载约 12.8 MB 的模型及运行时，低端设备可能较慢。
- `模型加载失败`：检查静态服务器、模型/config 是否成对更新、WASM/MJS 是否完整、请求是否返回 200 和正确文件而非 HTML。HTTPS/localhost 下会额外校验模型 SHA-256；普通不安全 HTTP 环境无 Web Crypto 时不做此可选完整性检查，CNN 推理仍由真实 ONNX 执行。
- `识别失败`：页面会明确显示异常且隐藏结果。打开 Console 看实际错误，不存在默认类别或隐藏预测 fallback。
- 导出使用 PyTorch 的 `dynamo=False` 稳定 TorchScript 路径，当前版本会显示弃用提示；本次导出的 opset 17 图已经通过 checker、Python 和浏览器的真实执行。

原始权重与 ONNX 的 SHA-256、逐图预测和各阶段日志均在 `results/` 中，便于核验模型与报告的对应关系。
