// Check stable math, actual validation decisions, and class-order independence.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { softmax, classify } from '../web/preprocess.js';
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const config = read('../web/model_config.json');
const rows = read('../results/validation_predictions_3class.json');
for (const row of rows) {
  const probabilities = softmax(row.logits);
  row.probabilities.forEach((p, i) => assert(Math.abs(p - probabilities[i]) < 1e-12));
  const result = classify(probabilities, config);
  const permutation = [2, 0, 1];
  const classes = permutation.map(i => config.classes[i]);
  const permutedConfig = { ...config, classes, class_to_idx: Object.fromEntries(classes.map((name, i) => [name, i])) };
  const permuted = classify(permutation.map(i => probabilities[i]), permutedConfig);
  assert.equal(permuted.category, result.category);
  assert.equal(permuted.thresholdRejected, result.thresholdRejected);
  assert.equal(permuted.confidence, result.confidence);
  if (result.thresholdRejected) {
    assert.equal(result.confidenceKind, 'highest-target-probability');
    assert.equal(result.confidence, result.highestTargetProbability);
  } else assert.equal(result.confidence, probabilities[config.class_to_idx[result.rawCategory]]);
}
// Numerical edge cases are unit checks, not substitutes for model predictions.
const extremes = softmax([1000, 999, -1000]);
assert(extremes.every(Number.isFinite));
assert(Math.abs(extremes.reduce((a,b) => a+b) - 1) < 1e-12);
assert.throws(() => softmax([NaN, 1, 2]));
assert.throws(() => softmax([1]));
console.log(`PASS: ${rows.length} actual validation logits, permuted mapping, confidence semantics, stable softmax`);
