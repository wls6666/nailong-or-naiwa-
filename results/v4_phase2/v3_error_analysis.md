# Frozen V3 diagnostic (historical test only)

321 Emoji:52 accepted. Humanoid/fantasy31/61; animals9/122; yellow faces12/138. Noto23/121, Twemoji23/121, Fluent3D6/79.

At deployed OR thresholds, Energy uniquely rejects149 Emoji; distance uniquely rejects19; both reject101. Energy is the dominant effective rejection signal, with complementary distance. ID rejects5/73: distance-only3,energy-only1,both1. All3 closed-set mistakes are accepted.

Hypothesis: character/domain exposure could improve representations and energy rejection. These historical test errors are not training examples, calibration data, or a source of matching-image collection. New data role assignments precede all new model scores.

Detailed sample-level rejected-ID scores are in v3_error_analysis.json. Pixel-cause attribution requires visual review; the metrics alone cannot identify a causal reason.

Visual review of the five rejected ID images (v3_id_rejected_review.jpg) shows text overlays, a multi-character cooking scene, elaborate costume/photographed character, and low-resolution screen capture. These are appearance-coverage hypotheses, not proof of a causal failure mechanism. Original labels and splits are retained.
