// Identical half-pixel bilinear arithmetic to training/dataset.py.
export const SIZE = 128;
export const MAX_BYTES = 10 * 1024 * 1024;
const MAX_PIXELS = 16_000_000;
const MIME_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);

export function validateImageBytes(buffer, declaredType) {
  if (declaredType && !MIME_TYPES.has(declaredType)) throw new Error('请选择 JPEG、PNG 或 WebP 图片。');
  const bytes = new Uint8Array(buffer), view = new DataView(buffer);
  if (bytes.length < 24) throw new Error('图片内容不完整，无法读取。');
  const text = (start, n) => String.fromCharCode(...bytes.subarray(start, start + n));
  let width, height, type;
  if (text(0, 8) === '\x89PNG\r\n\x1a\n' && text(12, 4) === 'IHDR') {
    type = 'image/png'; width = view.getUint32(16); height = view.getUint32(20);
  } else if (bytes[0] === 255 && bytes[1] === 216) {
    type = 'image/jpeg'; let i = 2;
    const sof = new Set([192, 193, 194, 195, 197, 198, 199, 201, 202, 203, 205, 206, 207]);
    while (i + 4 < bytes.length) {
      if (bytes[i] !== 255) break;
      while (bytes[i] === 255) i++;
      const marker = bytes[i++];
      if (marker === 217 || marker === 218) break;
      if (marker === 1 || (marker >= 208 && marker <= 215)) continue;
      if (i + 2 > bytes.length) break;
      const length = view.getUint16(i);
      if (length < 2 || i + length > bytes.length) break;
      if (sof.has(marker) && length >= 8) { height = view.getUint16(i + 3); width = view.getUint16(i + 5); break; }
      i += length;
    }
  } else if (text(0, 4) === 'RIFF' && text(8, 4) === 'WEBP') {
    type = 'image/webp';
    for (let i = 12; i + 8 <= bytes.length;) {
      const kind = text(i, 4), length = view.getUint32(i + 4, true), d = i + 8;
      if (d + length > bytes.length) break;
      const u24 = p => bytes[p] + bytes[p + 1] * 256 + bytes[p + 2] * 65536;
      if (kind === 'VP8X' && length >= 10) { width = u24(d + 4) + 1; height = u24(d + 7) + 1; break; }
      if (kind === 'VP8 ' && length >= 10 && text(d + 3, 3) === '\x9d\x01\x2a') {
        width = view.getUint16(d + 6, true) & 16383; height = view.getUint16(d + 8, true) & 16383; break;
      }
      if (kind === 'VP8L' && length >= 5 && bytes[d] === 47) {
        const bits = view.getUint32(d + 1, true); width = (bits & 16383) + 1; height = ((bits >>> 14) & 16383) + 1; break;
      }
      i = d + length + (length & 1);
    }
  }
  if (!type || !width || !height) throw new Error('无法读取图片格式或尺寸，请换一张有效图片。');
  if (declaredType && declaredType !== type) throw new Error('图片类型与实际内容不一致。');
  if (width * height > MAX_PIXELS || width > 16384 || height > 16384) {
    throw new Error('图片尺寸过大，请缩小到 1600 万像素以内再试。');
  }
  return { width, height, type };
}

export function pixelsToCHW(rgba, width, height) {
  const output = new Float32Array(3 * SIZE * SIZE), plane = SIZE * SIZE;
  for (let y = 0; y < SIZE; y++) {
    const sy = Math.max(0, (y + .5) * height / SIZE - .5);
    const y0 = Math.floor(sy), y1 = Math.min(y0 + 1, height - 1), wy = sy - y0;
    for (let x = 0; x < SIZE; x++) {
      const sx = Math.max(0, (x + .5) * width / SIZE - .5);
      const x0 = Math.floor(sx), x1 = Math.min(x0 + 1, width - 1), wx = sx - x0;
      for (let channel = 0; channel < 3; channel++) {
        const at = (px, py) => rgba[(py * width + px) * 4 + channel];
        const top = at(x0, y0) * (1 - wx) + at(x1, y0) * wx;
        const bottom = at(x0, y1) * (1 - wx) + at(x1, y1) * wx;
        output[channel * plane + y * SIZE + x] = (top * (1 - wy) + bottom * wy) / 255;
      }
    }
  }
  return output;
}

export function imageToCHW(image) {
  const width = image.naturalWidth || image.width, height = image.naturalHeight || image.height;
  if (!width || !height || width * height > MAX_PIXELS) throw new Error('图片尺寸无效或过大。');
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true, colorSpace: 'srgb' });
  if (!ctx) throw new Error('浏览器不支持 Canvas 图片处理。');
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, width, height);
  ctx.drawImage(image, 0, 0); // Decode only; perform resize explicitly below.
  const data = ctx.getImageData(0, 0, width, height).data;
  const tensor = pixelsToCHW(data, width, height);
  canvas.width = canvas.height = 1; // Release the original-size backing store.
  return tensor;
}

export function softmax(logits) {
  const values = Array.from(logits, Number);
  if (values.length !== 3 || !values.every(Number.isFinite)) throw new Error('模型必须输出三个有限 logits。');
  const max = Math.max(...values);
  const exponentials = values.map(value => Math.exp(value - max));
  const sum = exponentials.reduce((a, b) => a + b, 0);
  return exponentials.map(value => value / sum);
}

export function classify(probabilities, config) {
  const mapping = config.class_to_idx;
  let rawIndex = 0;
  probabilities.forEach((p, i) => { if (p > probabilities[rawIndex]) rawIndex = i; });
  const highestTargetProbability = Math.max(...config.rejection.target_classes.map(name => probabilities[mapping[name]]));
  const thresholdRejected = rawIndex !== mapping.other && highestTargetProbability < config.reject_threshold;
  const rawCategory = config.classes[rawIndex];
  return {
    rawCategory, category: thresholdRejected ? 'other' : rawCategory, thresholdRejected,
    highestTargetProbability,
    confidence: thresholdRejected ? highestTargetProbability : probabilities[rawIndex],
    confidenceKind: thresholdRejected ? 'highest-target-probability' : 'predicted-class-probability',
  };
}
