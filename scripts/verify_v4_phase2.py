"""Check actual experiment artifacts, paired budget and holdout feedback guard.

Run after run_v4_phase2.py. Test probes are never passed to any model/trainer.
"""

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from common import ROOT, save_json, sha256, load_model
from dataset import datasets
from v4.common import Images, check_production
from v4_phase2.pipeline import BASE, EXP, MANIFEST, protocol
from v4_phase2.guards import validate_selection


def main():
    torch.set_num_threads(1)
    p = protocol()
    lock = json.loads((BASE / "evaluation_lock.json").read_text())
    manifest = json.loads(MANIFEST.read_text())
    for row in manifest["samples"]:
        assert sha256(ROOT / row["path"]) == row["sha256"]
    for row in json.loads((ROOT / "data/hard_ood_test_manifest.json").read_text())[
        "samples"
    ]:
        assert sha256(ROOT / row["path"]) == row["file_sha256"]
    for record in lock["models"].values():
        for path, digest in record.items():
            assert sha256(ROOT / path) == digest
    baseline, _ = load_model(ROOT / "checkpoints/best_model_ood.pth")
    shapes = {k: list(v.shape) for k, v in baseline.state_dict().items()}
    configs = {}
    verified = []
    for strategy in ["random", "mined"]:
        selection = json.loads((EXP / f"{strategy}_selection.json").read_text())
        validate_selection(
            selection,
            [r for r in manifest["samples"] if r["split"] == "train"],
            120,
            40,
        )
        for seed in p["training_seeds"]:
            run = f"{strategy}_seed{seed}"
            folder = BASE / run
            config = json.loads((folder / "training_config.json").read_text())
            configs[run] = config
            done = json.loads((folder / "complete.json").read_text())
            history = json.loads((folder / "training_history.json").read_text())
            assert (
                done["total_steps"] == 132
                and done["epochs_executed"] == len(history) == 12
            )
            assert history[-1]["steps"] == 132
            assert (
                config["new_cartoon"] == 120
                and config["original_easy"] == 600
                and config["original_hard_and_feedback"] == 143
            )

            def key(row):
                feasible = (
                    row["val_id_accept"] >= 0.95
                    and row["val_id_accuracy"]
                    >= config["initial_val_id_accuracy"] - 0.03
                )
                far = list(row["val_far"].values())
                return (
                    feasible,
                    -max(far),
                    -float(np.mean(far)),
                    row["val_id_end_to_end_accuracy"],
                    row["val_id_accuracy"],
                )

            assert done["best_epoch"] == max(history, key=key)["epoch"]
            checkpoint = ROOT / f"checkpoints/v4_phase2/{run}.pth"
            assert sha256(checkpoint) == done["checkpoint_sha256"]
            model, saved = load_model(checkpoint)
            assert {k: list(v.shape) for k, v in model.state_dict().items()} == shapes
            assert any(
                not torch.equal(value, baseline.state_dict()[name])
                for name, value in model.state_dict().items()
            )
            data = np.load(folder / "features/train.npz")
            tm = json.loads((folder / "features/train.json").read_text())
            assert data["embedding"].shape == (345, 256) and data["logits"].shape == (
                345,
                2,
            )
            assert tm["checkpoint_sha256"] == done["checkpoint_sha256"]
            ds = Images(tm["rows"][:1], 128)
            with torch.inference_mode():
                logits, embedding = model.forward_outputs(ds[0][0].unsqueeze(0))
            ld = float(np.abs(logits.numpy()[0] - data["logits"][0]).max())
            ed = float(np.abs(embedding.numpy()[0] - data["embedding"][0]).max())
            assert ld < 1e-4 and ed < 1e-4
            ood = json.loads((folder / "ood_config.json").read_text())
            assert ood["fit_id_samples"] == 345 and ood["embedding_dim"] == 256
            assert np.linalg.eigvalsh(np.array(ood["inverse_covariance"])).min() > 0
            assert (
                ood["calibration_split"] == "val"
                and ood["checkpoint_sha256"] == done["checkpoint_sha256"]
            )
            cal = json.loads((folder / "calibration.json").read_text())
            assert (
                cal["test_loaded"] is False
                and cal["selected"]["validation_id_accept"] >= 0.95
            )
            rows = list(csv.DictReader((folder / "predictions.csv").open()))
            metrics = json.loads((folder / "metrics.json").read_text())
            assert len(rows) == 692
            for domain, count in [("easy", 150), ("hard", 321), ("cartoon", 148)]:
                rr = [r for r in rows if r["domain"] == domain]
                assert len(rr) == count
                assert (
                    sum(r["accepted"] == "True" for r in rr)
                    == metrics["per_domain"][domain]["accepted"]
                )
            verified.append(
                {
                    "model": run,
                    "status": "PASS",
                    "epochs": 12,
                    "steps": 132,
                    "best_epoch": done["best_epoch"],
                    "actual_pytorch_cache_logit_difference": ld,
                    "actual_pytorch_cache_embedding_difference": ed,
                }
            )
    paired_keys = [
        "epochs",
        "batch_size_id",
        "batch_size_oe",
        "optimizer",
        "lr",
        "weight_decay",
        "oe_weight",
        "oe_objective",
        "oe_sampling_mass",
        "oe_draws_per_epoch",
        "initial_checkpoint_sha256",
        "margin_in",
        "margin_out",
        "train_set_sha256",
        "val_set_sha256",
        "augmentation",
        "checkpoint_selection",
    ]
    for seed in p["training_seeds"]:
        for key in paired_keys:
            assert (
                configs[f"random_seed{seed}"][key] == configs[f"mined_seed{seed}"][key]
            ), key
    assert len({c["train_set_sha256"] for c in configs.values()}) == 1
    assert len({c["val_set_sha256"] for c in configs.values()}) == 1
    # All training is over before creating these temporary leakage probes.
    probes = []
    for split in ["val", "test"]:
        source = (
            ROOT / next(r for r in manifest["samples"] if r["split"] == split)["path"]
        )
        dest = ROOT / "data/false_positive/.phase2_holdout_guard_probe.jpg"
        assert not dest.exists()
        shutil.copyfile(source, dest)
        try:
            try:
                datasets(training=True)
            except ValueError as exc:
                assert "leaks validation/test" in str(exc)
                probes.append({"split": split, "status": "PASS", "error": str(exc)})
            else:
                raise AssertionError("Phase2 holdout entered feedback loader")
        finally:
            dest.unlink()
    old = json.loads((ROOT / "results/v4_phase1/v3_baseline/metrics.json").read_text())
    new = json.loads((BASE / "v3_baseline/metrics.json").read_text())
    assert (
        new["per_domain"]["hard"]["accepted"] == 52
        and new["per_domain"]["easy"]["accepted"] == 18
    )
    assert new["closed_id_accuracy"] == old["closed_set_id_accuracy"]
    assert new["id_accuracy_including_rejection"] == old["id_end_to_end_accuracy"]
    assert (
        json.loads((BASE / "browser_regression.json").read_text())["status"] == "PASS"
    )
    check_production()
    save_json(
        BASE / "verification.json",
        {
            "status": "PASS",
            "paired_training_budget_and_inputs_equal": True,
            "actual_trained_models": verified,
            "feedback_leakage_probes": probes,
            "new_test_count": 148,
            "permanent_emoji_count": 321,
            "frozen_v3_historical_results_reproduced": True,
            "test_params_from_validation_only": True,
            "production_files_unchanged": True,
            "candidate_onnx_not_claimed": "Only production browser regression was executed; new export requires promotion evidence.",
        },
    )
    print("Phase2 artifact/control/feedback guards PASS")


if __name__ == "__main__":
    main()
