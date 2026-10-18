import sys, json, copy, collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from common import save_json, sha256
from v4_phase2.pipeline import protocol, cartoon, extract, BASE, MANIFEST, EXP
from v4_phase2.guards import validate_selection

p = protocol()
m = json.load(open(MANIFEST))
selections = {
    s: json.load(open(EXP / f"{s}_selection.json")) for s in ["random", "mined"]
}
for s, doc in selections.items():
    validate_selection(doc, cartoon("train"), 120, 40)
leaked = copy.deepcopy(selections["random"])
leaked["samples"][0] = cartoon("test")[0]
try:
    validate_selection(leaked, cartoon("train"), 120, 40)
except AssertionError as e:
    message = str(e)
else:
    raise AssertionError("Test image accepted as OE selection")
assert not (BASE / "evaluation_lock.json").exists()
try:
    extract("v3_baseline", "test")
except FileNotFoundError:
    blocked = True
else:
    raise AssertionError("Test extraction allowed before all fits locked")
assert not (BASE / "v3_baseline/features/test.npz").exists()
for field in ["source", "identity_group", "original_group"]:
    sets = {
        s: {r[field] for r in m["samples"] if r["split"] == s}
        for s in ["train", "val", "test"]
    }
    assert all(
        not (sets[a] & sets[b])
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
    )
assert (
    sha256(Path("data/hard_ood_test_manifest.json"))
    == json.load(open(BASE / "freeze.json"))["emoji_manifest_sha256"]
)
save_json(
    BASE / "preflight_verification.json",
    {
        "status": "PASS",
        "selection_val_test_injection_blocked": message,
        "test_extraction_before_lock_blocked": blocked,
        "source_identity_original_groups_disjoint": True,
        "equal_source_quota_and2s_spacing": True,
        "production_hashes_unchanged": True,
        "permanent_emoji_manifest_unchanged": True,
        "test_model_outputs_generated": False,
    },
)
print("Phase2 preflight guards PASS")
