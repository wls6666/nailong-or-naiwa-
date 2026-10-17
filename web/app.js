import { SIZE, MAX_BYTES, validateImageBytes, imageToCHW, softmax, classify } from './preprocess.js';

const $ = id => document.getElementById(id);
const input = $('file-input'), button = $('predict-button'), status = $('model-status');
const names = { nailong: { zh: '奶龙', en: 'Nailong' }, naiwa: { zh: '奶蛙', en: 'Naiwa' }, other: { zh: '都不是', en: 'Neither' } };
let session, config, bitmap, objectURL, busy = false, uploadVersion = 0, modelLoading = false;

function showError(message) { $('error').textContent = message; $('error').hidden = !message; }
function updateButton() { button.disabled = !session || !bitmap || busy || modelLoading; }
function clearImage() {
  if (bitmap) bitmap.close();
  bitmap = undefined;
  if (objectURL) URL.revokeObjectURL(objectURL);
  objectURL = undefined;
  $('preview').hidden = true; $('preview').removeAttribute('src');
  $('drop-zone').classList.remove('has-image');
  $('file-name').textContent = '还没有选择图片';
  $('result').hidden = true;
  updateButton();
}

async function loadModel() {
  if (modelLoading) return;
  modelLoading = true; session = undefined; updateButton(); showError('');
  status.textContent = '模型加载中…'; status.dataset.state = 'loading'; $('retry-button').hidden = true;
  try {
    if (!globalThis.ort) throw new Error('ONNX Runtime Web 未加载，请检查静态资源。');
    const response = await fetch(new URL('./model_config.json', import.meta.url), { cache: 'no-cache' });
    if (!response.ok) throw new Error(`模型配置 HTTP ${response.status}`);
    config = await response.json();
    const mapping = config.class_to_idx;
    if (!mapping || Object.keys(mapping).length !== 3 || !['nailong', 'naiwa', 'other'].every(name => Number.isInteger(mapping[name]))
        || new Set(Object.values(mapping)).size !== 3 || !Object.values(mapping).every(i => i >= 0 && i < 3)
        || !Array.isArray(config.classes) || config.classes.length !== 3
        || !config.classes.every((name, i) => mapping[name] === i)) throw new Error('模型三分类映射无效。');
    if (!Number.isFinite(config.reject_threshold) || config.reject_threshold < .5 || config.reject_threshold > .95
        || config.rejection?.calibration_split !== 'val' || config.rejection.other_class !== 'other'
        || !Array.isArray(config.rejection.target_classes) || config.rejection.target_classes.length !== 2
        || new Set(config.rejection.target_classes).size !== 2
        || !config.rejection.target_classes.every(name => name === 'nailong' || name === 'naiwa')
        || JSON.stringify(config.output_shape) !== '[1,3]') throw new Error('模型输出或 validation 拒识配置无效。');
    const prep = config.preprocessing;
    if (prep?.size !== SIZE || prep.channels !== 'RGB' || prep.layout !== 'NCHW' || prep.dtype !== 'float32'
        || prep.normalization !== null || prep.resize !== 'bilinear-half-pixel-clamp-float-no-round'
        || JSON.stringify(prep.range) !== '[0,1]') throw new Error('网页与模型预处理配置不一致。');
    ort.env.wasm.wasmPaths = new URL('./assets/vendor/onnxruntime-web/', import.meta.url).href;
    ort.env.wasm.numThreads = 1; // No cross-origin isolation headers required on GitHub Pages.
    const modelResponse = await fetch(new URL('./model.onnx', import.meta.url), { cache: 'no-cache' });
    if (!modelResponse.ok) throw new Error(`模型文件 HTTP ${modelResponse.status}`);
    const bytes = await modelResponse.arrayBuffer();
    // Verify the exported artifact when the standard secure-context hash API is available.
    if (globalThis.crypto?.subtle) {
      const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(v => v.toString(16).padStart(2, '0')).join('');
      if (hash !== config.model_sha256) throw new Error('模型与配置校验不匹配，请同时更新两个文件。');
    }
    const loaded = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    if (loaded.inputNames.length !== 1 || loaded.inputNames[0] !== config.input_name
        || loaded.outputNames.length !== 1 || loaded.outputNames[0] !== config.output_name) {
      await loaded.release(); throw new Error('ONNX 实际输入输出名称与导出配置不一致。');
    }
    session = loaded;
    status.textContent = '模型已就绪'; status.dataset.state = 'ready';
    console.info('CNN ready', { input: session.inputNames, output: session.outputNames, mapping });
  } catch (error) {
    session = undefined;
    status.textContent = `模型加载失败：${error.message}`; status.dataset.state = 'error';
    showError(`模型加载失败：${error.message}`); $('retry-button').hidden = false;
    console.error(error);
  } finally { modelLoading = false; updateButton(); }
}

async function selectFile(file) {
  if (busy || !file) return;
  const version = ++uploadVersion;
  clearImage(); showError('');
  try {
    if (file.size > MAX_BYTES) throw new Error('图片超过 10 MB，请压缩后再试。');
    validateImageBytes(await file.arrayBuffer(), file.type);
    const decoded = await createImageBitmap(file, { imageOrientation: 'from-image', premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
    if (version !== uploadVersion) { decoded.close(); return; }
    if (decoded.width * decoded.height > 16_000_000) { decoded.close(); throw new Error('图片尺寸过大。'); }
    bitmap = decoded;
    objectURL = URL.createObjectURL(file);
    $('preview').src = objectURL; $('preview').hidden = false;
    $('drop-zone').classList.add('has-image');
    $('file-name').textContent = file.name;
    updateButton();
  } catch (error) {
    if (version === uploadVersion) { clearImage(); showError(`图片读取失败：${error.message}`); }
  }
}

async function predict() {
  if (!session || !bitmap || busy) return;
  busy = true; input.disabled = true; updateButton(); showError(''); $('result').hidden = true;
  button.firstElementChild.textContent = '正在分析…'; button.setAttribute('aria-busy', 'true');
  try {
    const data = imageToCHW(bitmap);
    const tensor = new ort.Tensor('float32', data, [1, 3, SIZE, SIZE]);
    const outputs = await session.run({ [config.input_name]: tensor });
    const output = outputs[config.output_name];
    if (!output || output.data.length !== 3 || output.dims.join(',') !== '1,3') throw new Error('模型必须真实输出 [1,3] logits。');
    const logits = Array.from(output.data, Number), probabilities = softmax(logits);
    const decision = classify(probabilities, config);
    const { category, confidence, thresholdRejected } = decision;
    if (!names[category]) throw new Error('无法匹配模型类别。');
    $('prediction-name').textContent = thresholdRejected ? '都不是 / 无法确定' : `${names[category].zh} / ${names[category].en}`;
    $('confidence-value').textContent = `${(confidence * 100).toFixed(2)}%`;
    $('confidence-label').textContent = thresholdRejected ? '最高目标类概率' : '模型置信度';
    $('result-note').textContent = thresholdRejected ? '低置信度拒识' : '三分类 CNN 预测';
    $('result').dataset.category = category;
    $('confidence-fill').style.width = `${confidence * 100}%`;
    $('confidence-bar').setAttribute('aria-valuenow', String(confidence * 100));
    $('confidence-bar').setAttribute('aria-label', thresholdRejected ? '最高目标类概率，非都不是的概率' : '预测类别置信度');
    const allProbabilities = config.classes.map((name, i) => `${names[name].zh} ${(probabilities[i] * 100).toFixed(2)}%`).join(' · ');
    $('probability-detail').textContent = thresholdRejected
      ? `最高目标类概率仅 ${(confidence * 100).toFixed(2)}%，低于验证集选定的 ${(config.reject_threshold * 100).toFixed(0)}% 阈值。${allProbabilities}`
      : allProbabilities;
    $('result').hidden = false;
    const detail = { shape: tensor.dims, outputShape: output.dims, logits, probabilities, ...decision };
    console.debug('CNN inference', detail);
    document.dispatchEvent(new CustomEvent('inference-complete', { detail }));
  } catch (error) {
    $('result').hidden = true;
    showError(`识别失败：${error.message} 请重试或选择其他图片。`);
    console.error(error);
  } finally {
    busy = false; input.disabled = false; button.firstElementChild.textContent = '开始识别';
    button.removeAttribute('aria-busy'); updateButton();
  }
}

input.addEventListener('change', () => { const file = input.files[0]; input.value = ''; selectFile(file); });
button.addEventListener('click', predict);
$('retry-button').addEventListener('click', loadModel);
for (const name of ['dragenter', 'dragover']) $('drop-zone').addEventListener(name, event => {
  event.preventDefault(); if (!busy) $('drop-zone').classList.add('dragging');
});
for (const name of ['dragleave', 'drop']) $('drop-zone').addEventListener(name, event => {
  event.preventDefault(); $('drop-zone').classList.remove('dragging');
});
$('drop-zone').addEventListener('drop', event => { if (!busy) selectFile(event.dataTransfer.files[0]); });
window.addEventListener('pagehide', () => { if (objectURL) URL.revokeObjectURL(objectURL); bitmap?.close(); });
loadModel();
