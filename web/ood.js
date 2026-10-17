// Post-hoc statistics fitted on ID TRAIN; all rejection parameters chosen on VAL.
// OOD margins are scores, never Other probabilities.
export function softmax(logits) {
  const values = Array.from(logits, Number);
  if (!values.length || !values.every(Number.isFinite)) throw new Error('模型 logits 无效。');
  const m = Math.max(...values), exp = values.map(v => Math.exp(v - m));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(v => v / sum);
}

export function validateOOD(config, model) {
  const d = config.embedding_dim, finite = Number.isFinite;
  if (config.version !== 'v3' || config.calibration_split !== 'val' || config.model_sha256 !== model.model_sha256
      || !Number.isInteger(d) || d < 1 || d > 512 || model.embedding_shape?.join(',') !== `1,${d}`
      || typeof config.normalize_embedding !== 'boolean'
      || !['or', 'and'].includes(config.reject_logic)
      || !['mahalanobis', 'energy', 'distance_energy', 'distance_energy_p_other'].includes(config.method)) throw new Error('OOD 配置版本、模型或维度不一致。');
  const matrix = config.inverse_covariance, centers = config.class_centers;
  if (!Array.isArray(centers) || centers.length !== 2 || !centers.every(a => a.length === d && a.every(finite))
      || !Array.isArray(matrix) || matrix.length !== d || !matrix.every(a => a.length === d && a.every(finite))
      || !matrix.every((row, i) => row[i] > 0)) throw new Error('OOD 中心或逆协方差矩阵无效。');
  if (![config.mahalanobis_threshold, config.energy_threshold].every(finite)
      || ![config.energy_temperature, config.distance_scale, config.energy_scale].every(v => finite(v) && v > 0)) throw new Error('OOD 校准参数无效。');
  const indices = ['nailong', 'naiwa'].map(name => config.class_mapping?.[name]);
  if (indices.some((i, n) => !Number.isInteger(i) || i < 0 || i >= model.logits_shape?.[1]
      || model.class_to_idx?.[['nailong', 'naiwa'][n]] !== i)
      || new Set(indices).size !== 2
      || JSON.stringify(config.class_indices) !== JSON.stringify(indices)
      || JSON.stringify(config.energy_logit_indices) !== JSON.stringify(indices)) throw new Error('OOD 类别映射不一致。');
  if (config.method === 'distance_energy_p_other' && (!Number.isInteger(config.other_logit_index)
      || config.other_logit_index < 0 || config.other_logit_index >= model.logits_shape[1]
      || !finite(config.p_other_threshold) || config.p_other_threshold <= 0 || config.p_other_threshold > 1)) throw new Error('p_other 校准参数无效。');
}

export function inferOOD(logits, embedding, config) {
  if (embedding.length !== config.embedding_dim || !Array.from(embedding).every(Number.isFinite)) throw new Error('模型 embedding 无效。');
  const z = Float64Array.from(embedding);
  if (config.normalize_embedding) {
    const norm = Math.max(Math.sqrt(z.reduce((s, v) => s + v * v, 0)), 1e-12);
    for (let i = 0; i < z.length; i++) z[i] /= norm;
  }
  // A real full shared-covariance quadratic form, NOT Euclidean distance.
  let distance = Infinity;
  for (const center of config.class_centers) {
    const delta = z.map((v, i) => v - center[i]); let quadratic = 0;
    for (let i = 0; i < z.length; i++) {
      let row = 0;
      for (let j = 0; j < z.length; j++) row += config.inverse_covariance[i][j] * delta[j];
      quadratic += delta[i] * row;
    }
    distance = Math.min(distance, Math.max(quadratic, 0));
  }
  const indices = config.energy_logit_indices, temperature = config.energy_temperature;
  const values = indices.map(i => logits[i] / temperature), maximum = Math.max(...values);
  const energy = -temperature * (maximum + Math.log(values.reduce((s, v) => s + Math.exp(v - maximum), 0)));
  const dm = (distance - config.mahalanobis_threshold) / config.distance_scale;
  const em = (energy - config.energy_threshold) / config.energy_scale;
  let score = config.method === 'mahalanobis' ? dm : config.method === 'energy' ? em
    : config.reject_logic === 'or' ? Math.max(dm, em) : Math.min(dm, em);
  const probabilities = softmax(indices.map(i => logits[i]));
  if (config.method === 'distance_energy_p_other') score = Math.max(score,
    (softmax(logits)[config.other_logit_index] - config.p_other_threshold) / config.p_other_threshold);
  if (![distance, energy, score].every(Number.isFinite)) throw new Error('OOD 分数不是有限数值。');
  const targetIndex = probabilities[0] >= probabilities[1] ? 0 : 1;
  const targetName = Object.keys(config.class_mapping).find(name => config.class_mapping[name] === indices[targetIndex]);
  return { category: score > 0 ? 'other' : targetName, accepted: score <= 0,
    confidence: score > 0 ? null : probabilities[targetIndex], confidenceKind: score > 0 ? 'no-ood-probability' : 'conditional-id-softmax',
    probabilities, distance, energy, oodScore: score };
}
