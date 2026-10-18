"""Locked-test summaries; never selects training parameters or thresholds."""

import csv, json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from common import ROOT, save_json, sha256
from v4_phase2.pipeline import BASE, EXP, protocol, write_csv


def report():
    p = protocol()
    names = ["v3_baseline", "v3_recalibrated"] + [
        f"{s}_seed{seed}" for s in ["random", "mined"] for seed in p["training_seeds"]
    ]
    metrics = {
        name: json.loads((BASE / name / "metrics.json").read_text()) for name in names
    }
    assert len({r["test_set_sha256"] for r in metrics.values()}) == 1
    predictions = {
        name: list(csv.DictReader((BASE / name / "predictions.csv").open()))
        for name in names
    }
    flat = []
    for name, m in metrics.items():
        flat.append(
            {
                "model": name,
                "closed_id_accuracy": m["closed_id_accuracy"],
                "id_accuracy_including_rejection": m["id_accuracy_including_rejection"],
                "id_false_reject_rate": m["id_false_reject_rate"],
                "id_rejected": m["id_rejected"],
                "id_closed_misclassified": m["id_closed_misclassified"],
                "id_accepted_misclassified": m["id_accepted_misclassified"],
                "easy_far": m["per_domain"]["easy"]["far"],
                "emoji_far": m["per_domain"]["hard"]["far"],
                "real_cartoon_far": m["per_domain"]["cartoon"]["far"],
                "auroc": m["auroc"],
                "aupr_ood": m["aupr_ood"],
                "fpr_at_95_tpr_id": m["fpr_at_95_tpr_id"],
                "method": m["method"],
                "reject_logic": m["reject_logic"],
            }
        )
    write_csv(BASE / "comparison.csv", flat)
    source_rows = []
    for name, prediction_rows in predictions.items():
        groups = sorted(
            {(r["domain"], r["source"]) for r in prediction_rows if r["domain"] != "id"}
        )
        for domain, source in groups:
            selected_rows = [
                r
                for r in prediction_rows
                if r["domain"] == domain and r["source"] == source
            ]
            accepted_count = sum(r["accepted"] == "True" for r in selected_rows)
            source_rows.append(
                {
                    "model": name,
                    "domain": domain,
                    "source": source or "legacy_easy_photos",
                    "n": len(selected_rows),
                    "accepted": accepted_count,
                    "far": accepted_count / len(selected_rows),
                }
            )
    write_csv(BASE / "source_breakdown.csv", source_rows)
    keys = [
        "closed_id_accuracy",
        "id_accuracy_including_rejection",
        "id_false_reject_rate",
        "easy_far",
        "emoji_far",
        "real_cartoon_far",
        "auroc",
        "aupr_ood",
        "fpr_at_95_tpr_id",
    ]
    aggregate = {}
    for strategy in ["random", "mined"]:
        rr = [r for r in flat if r["model"].startswith(strategy + "_")]
        aggregate[strategy] = {
            key: {
                "mean": float(np.mean([r[key] for r in rr])),
                "seed_sd": float(np.std([r[key] for r in rr], ddof=1)),
            }
            for key in keys
        }
    # Paired uncertainty: same test rows and seed-averaged accept decisions. Resample
    # entire movie/cast groups for real cartoons; Unicode identities for Emoji.
    rng = np.random.default_rng(20261018)
    uncertainty = {}
    reference = predictions["v3_baseline"]

    def accepted(name, domain):
        return np.array(
            [
                r["accepted"] == "True"
                for r in predictions[name]
                if r["domain"] == domain
            ],
            dtype=float,
        )

    def averaged(strategy, domain):
        return np.mean(
            [
                accepted(f"{strategy}_seed{seed}", domain)
                for seed in p["training_seeds"]
            ],
            axis=0,
        )

    for domain in ["hard", "cartoon"]:
        rr = [r for r in reference if r["domain"] == domain]
        groups = np.array([r["identity_group"] for r in rr])
        unique = sorted(set(groups))
        masks = [groups == g for g in unique]
        pairs = {}
        for label, delta in [
            (
                "mined_minus_random",
                averaged("mined", domain) - averaged("random", domain),
            ),
            (
                "mined_minus_v3",
                averaged("mined", domain) - accepted("v3_baseline", domain),
            ),
        ]:
            sums = np.array([delta[m].sum() for m in masks])
            counts = np.array([m.sum() for m in masks])
            draws = rng.integers(0, len(unique), size=(5000, len(unique)))
            values = sums[draws].sum(1) / counts[draws].sum(1)
            pairs[label] = {
                "far_difference": float(delta.mean()),
                "group_bootstrap_ci95": np.quantile(values, [0.025, 0.975]).tolist(),
                "resampling_unit": (
                    "entire movie/cast"
                    if domain == "cartoon"
                    else "Unicode character identity"
                ),
                "groups": len(unique),
                "caution": (
                    "Only two independent films; interval is conditional and cannot establish broad character/source generalization."
                    if domain == "cartoon"
                    else "Conditional on fixed benchmark identities; separate from training-seed variability."
                ),
            }
        uncertainty[domain] = pairs
    save_json(
        BASE / "uncertainty.json",
        {
            "across_seed": aggregate,
            "fixed_test_group_resampling": uncertainty,
            "paired_seed_differences": [
                {key: flat[2 + 3 + i][key] - flat[2 + i][key] for key in keys}
                for i in range(3)
            ],
            "seed_selection": "No seed selected using test. All three reported.",
        },
    )
    # Model-blind adoption gate; all test runs retained, no subsequent retraining.
    a = flat[0]
    b = aggregate["random"]
    c = aggregate["mined"]
    gate = {
        "real_better_than_b_and_a": c["real_cartoon_far"]["mean"]
        < min(b["real_cartoon_far"]["mean"], a["real_cartoon_far"]),
        "emoji_better_than_b_and_a": c["emoji_far"]["mean"]
        < min(b["emoji_far"]["mean"], a["emoji_far"]),
        "id_end_to_end_within3pp": c["id_accuracy_including_rejection"]["mean"]
        >= a["id_accuracy_including_rejection"] - 0.03,
        "id_frr_within3pp": c["id_false_reject_rate"]["mean"]
        <= a["id_false_reject_rate"] + 0.03,
        "easy_far_within3pp": c["easy_far"]["mean"] <= a["easy_far"] + 0.03,
        "paired_real_group_ci_below0": uncertainty["cartoon"]["mined_minus_random"][
            "group_bootstrap_ci95"
        ][1]
        < 0,
        "sufficient_independent_final_groups": len(
            {r["identity_group"] for r in reference if r["domain"] == "cartoon"}
        )
        >= 5,
    }
    gate["promote"] = all(gate.values())
    gate["production_changed"] = False
    gate["onnx_exported"] = False
    if gate["promote"]:
        gate["note"] = (
            "Eligible for isolated candidate export; production never replaced automatically."
        )
    else:
        gate["note"] = (
            "No credible promotion gate pass; no candidate ONNX fabricated/exported. Production V3 remains intact."
        )
    save_json(BASE / "decision.json", gate)
    pct = lambda x: f"{100*x:.2f}%"
    lines = [
        "# V4 Phase 2 — controlled cartoon OE experiment",
        "",
        "All thresholds/checkpoints were locked before the new final-test inference. Lower FAR is better.",
        "",
        "| Model | Real cartoon FAR | Emoji FAR | Easy FAR | ID closed | ID incl. reject | ID rejected /73 | closed / accepted errors | AUROC | FPR@95TPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in flat:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["model"],
                    pct(r["real_cartoon_far"]),
                    pct(r["emoji_far"]),
                    pct(r["easy_far"]),
                    pct(r["closed_id_accuracy"]),
                    pct(r["id_accuracy_including_rejection"]),
                    str(r["id_rejected"]),
                    f"{r['id_closed_misclassified']} / {r['id_accepted_misclassified']}",
                    pct(r["auroc"]),
                    pct(r["fpr_at_95_tpr_id"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Across-seed mean ± sample SD (three actual training seeds; not test sampling uncertainty):",
        "",
    ]
    for strategy, stats in aggregate.items():
        lines.append(
            f"- {strategy}: Real FAR {pct(stats['real_cartoon_far']['mean'])} ± {pct(stats['real_cartoon_far']['seed_sd'])}; Emoji FAR {pct(stats['emoji_far']['mean'])} ± {pct(stats['emoji_far']['seed_sd'])}; ID incl. reject {pct(stats['id_accuracy_including_rejection']['mean'])} ± {pct(stats['id_accuracy_including_rejection']['seed_sd'])}."
        )
    lines += [
        "",
        "Group-bootstrap intervals, paired per fixed test image and averaged over seeds (negative is improvement):",
        "",
    ]
    for domain, pairs in uncertainty.items():
        for label, r in pairs.items():
            lines.append(
                f"- {domain} {label}: {pct(r['far_difference'])}, 95% CI [{pct(r['group_bootstrap_ci95'][0])}, {pct(r['group_bootstrap_ci95'][1])}], groups={r['groups']}."
            )
    lines += [
        "",
        "Real-cartoon test has only two held-out film/cast groups. This is a genuine source/identity holdout, but does **not** establish robust performance across cartoon brands or art styles. Per-image sample size overstates independent evidence. No seed or model was selected using test. A_recalibrated isolates validation expansion from training effects.",
        "",
        f"Production replacement recommended: **{gate['promote']}**. See decision.json and the full experiment README. No production assets changed.",
    ]
    (BASE / "comparison.md").write_text("\n".join(lines) + "\n")
    # Training curves and score histograms. All runs shown, no best-test cherry-pick.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for strategy, color in [("random", "#3268ab"), ("mined", "#c25334")]:
        for seed in p["training_seeds"]:
            history = json.loads(
                (BASE / f"{strategy}_seed{seed}/training_history.json").read_text()
            )
            axes[0].plot(
                [r["epoch"] for r in history],
                [r["train_loss"] for r in history],
                color=color,
                alpha=0.65,
                label=strategy if seed == 42 else None,
            )
            axes[1].plot(
                [r["epoch"] for r in history],
                [r["val_id_accuracy"] for r in history],
                color=color,
                alpha=0.65,
            )
    axes[0].set(title="Training CE + OE loss", xlabel="Epoch")
    axes[0].legend()
    axes[1].set(title="Validation ID closed accuracy", xlabel="Epoch")
    fig.tight_layout()
    fig.savefig(BASE / "training_curves.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, name in zip(axes, ["v3_baseline", "random_seed42", "mined_seed42"]):
        rr = predictions[name]
        for dom in ["id", "easy", "hard", "cartoon"]:
            x = [float(r["ood_score"]) for r in rr if r["domain"] == dom]
            ax.hist(x, bins=22, density=True, alpha=0.4, label=dom)
        ax.axvline(0, c="black", ls="--")
        ax.set_title(name)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(BASE / "score_distributions.png", dpi=140)
    plt.close(fig)
    # PCA fits ID training embeddings only. Seed42 is preselected for illustration.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, name in zip(axes, ["v3_recalibrated", "random_seed42", "mined_seed42"]):
        train = np.load(BASE / name / "features/train.npz")
        test = np.load(BASE / name / "features/test.npz")
        rr = json.loads((BASE / name / "features/test.json").read_text())["rows"]
        pca = PCA(n_components=2).fit(train["embedding"])
        xy = pca.transform(test["embedding"])
        for group in ["nailong", "naiwa", "easy", "hard", "cartoon"]:
            mask = np.array(
                [
                    (
                        r.get("class") == group
                        if group in ["nailong", "naiwa"]
                        else r["domain"] == group
                    )
                    for r in rr
                ]
            )
            ax.scatter(xy[mask, 0], xy[mask, 1], s=7, alpha=0.5, label=group)
        ax.set_title(name)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(BASE / "embedding_pca.png", dpi=140)
    plt.close(fig)
    print("\n".join(lines), flush=True)
