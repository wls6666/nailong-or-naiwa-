// Verify pure OOD math against saved REAL browser/Python comparisons.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { inferOOD, softmax, validateOOD } from '../web/ood.js';
const read = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const config = read('../web/ood_config.json'), model = read('../web/model_config.json');
validateOOD(config, model);
const rows = read('../results/v3/browser_verification.json').samples;
for (const { browser: actual } of rows) {
  const result = inferOOD(actual.logits, actual.embedding, config);
  assert.equal(result.category, actual.category);
  assert.equal(result.accepted, actual.accepted);
  assert(Math.abs(result.oodScore - actual.oodScore) < 1e-9);
  // Rename/reorder logit columns; fitted class-center meaning is unchanged.
  const permuted = { ...config, class_mapping: { nailong: 1, naiwa: 0 }, class_indices: [1, 0], energy_logit_indices: [1, 0] };
  const alternative = inferOOD([actual.logits[1], actual.logits[0]], actual.embedding, permuted);
  assert.equal(alternative.category, result.category);
  assert.equal(alternative.confidence, result.confidence);
  if (!result.accepted) assert.equal(result.confidence, null);
}
// Numerical edge cases are math checks, never model predictions.
assert(softmax([10000, 9999]).every(Number.isFinite));
assert.throws(() => softmax([NaN, 1]));
assert.throws(() => validateOOD({}, model));
console.log(`PASS: ${rows.length} real output pairs, mapping permutation, score semantics, stable softmax`);
