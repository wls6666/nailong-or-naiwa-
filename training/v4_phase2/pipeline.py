"""Random vs mined cartoon OE. Test features are inaccessible until all fits lock."""

import csv, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
from common import ROOT, load_model, save_json, sha256, seed_all
from v4.common import rows as old_rows, Images, features, check_production, set_hash
from v4.detector import scores
from ood.core import (
    fit_gaussian,
    distance,
    energy,
    combined_score,
    probabilities,
    rank_ood,
)
from calibrate_ood import quantiles
from v4_phase2.guards import validate_selection

BASE = ROOT / "results/v4_phase2"
EXP = ROOT / "experiments/v4_phase2"
MANIFEST = ROOT / "data/v4_phase2/cartoon_manifest.json"
CKPT = ROOT / "checkpoints/best_model_ood.pth"


def protocol():
    p = json.loads((EXP / "protocol.json").read_text())
    f = json.loads((BASE / "freeze.json").read_text())
    assert sha256(EXP / "protocol.json") == f["protocol_sha256"]
    assert sha256(CKPT) == p["initial_checkpoint_sha256"]
    check_production()
    addendum = json.loads((EXP / "data_protocol_addendum.json").read_text())
    p["negative_selection"].update(
        {
            k: v
            for k, v in addendum["selection_budget_revision"].items()
            if k in ["count", "quota_per_train_film"]
        }
    )
    if MANIFEST.exists():
        assert json.loads(MANIFEST.read_text())["data_addendum_sha256"] == sha256(
            EXP / "data_protocol_addendum.json"
        )
    return p


def cartoon(role):
    m = json.loads(MANIFEST.read_text())
    a = json.loads((BASE / "data_audit.json").read_text())
    assert sha256(MANIFEST) == a["manifest_sha256"]
    out = [r for r in m["samples"] if r["split"] == role]
    for r in out:
        assert sha256(ROOT / r["path"]) == r["sha256"]
    return out


def items(split):
    return old_rows(split) + ([] if split == "train" else cartoon(split))


def original_oe():
    from dataset import datasets

    ds = datasets(training=True)["train"]
    hard = json.loads((ROOT / "data/hard_ood_manifest.json").read_text())
    a = [
        {"path": str(Path(p).relative_to(ROOT)), "label": 2, "domain": "easy"}
        for p, y in ds.samples
        if y == 2
    ]
    h = [
        {**r, "label": 2, "domain": "old_hard"}
        for r in hard["samples"]
        if r["split"] == "train"
    ]
    feedback = [
        r
        for r in a
        if "/hard_negative/" in r["path"] or "/false_positive/" in r["path"]
    ]
    a = [r for r in a if r not in feedback]
    h += feedback
    assert len({r["path"] for r in a + h}) == len(a + h)
    return a, h


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def spaced_selection(priority, count):
    """Honor ranking and 2s source spacing without greedy cardinality dead ends."""
    chosen = []

    def compatible(r, chosen):
        return all(
            abs(r["timestamp_seconds"] - s["timestamp_seconds"]) >= 2 for s in chosen
        )

    def capacity(rr):
        n = 0
        last = -float("inf")
        for r in sorted(rr, key=lambda r: r["timestamp_seconds"]):
            if r["timestamp_seconds"] - last >= 2:
                n += 1
                last = r["timestamp_seconds"]
        return n

    assert capacity(priority) >= count, (
        "Insufficient temporally separated candidates",
        capacity(priority),
        count,
    )
    for r in priority:
        if len(chosen) == count:
            break
        if not compatible(r, chosen):
            continue
        trial = chosen + [r]
        remaining = [
            s
            for s in priority
            if s["image_id"] not in {t["image_id"] for t in trial}
            and compatible(s, trial)
        ]
        if len(trial) + capacity(remaining) >= count:
            chosen = trial
    assert len(chosen) == count
    return chosen


def mine():
    p = protocol()
    candidates = cartoon("train")  # No val/test rows or scores loaded here.
    model, _ = load_model(CKPT)
    cfg = json.loads((ROOT / "web/ood_config.json").read_text())
    logits, emb = features(model, Images(candidates, 128))
    d, e, s = scores(logits, emb, cfg)
    prob = probabilities(logits)
    enriched = []
    for i, r in enumerate(candidates):
        enriched.append(
            {
                **r,
                "v3_prediction": int(logits[i].argmax()),
                "logits": logits[i].tolist(),
                "softmax": prob[i].tolist(),
                "energy": float(e[i]),
                "mahalanobis": float(d[i]),
                "accepted": bool(s[i] <= 0),
                "acceptance_margin": float(-s[i]),
                "distance_margin": float(
                    (cfg["mahalanobis_threshold"] - d[i]) / cfg["distance_scale"]
                ),
                "energy_margin": float(
                    (cfg["energy_threshold"] - e[i]) / cfg["energy_scale"]
                ),
            }
        )
    count = p["negative_selection"]["quota_per_train_film"]
    rng = np.random.default_rng(p["selection_seed"])
    random = []
    mined = []
    stats = {}
    for source in sorted({r["source"] for r in enriched}):
        group = [r for r in enriched if r["source"] == source]
        assert len(group) >= count, (source, len(group), count)
        random.extend(
            spaced_selection([group[i] for i in rng.permutation(len(group))], count)
        )
        accepted = sorted(
            [r for r in group if r["accepted"]], key=lambda r: -r["acceptance_margin"]
        )
        primary = accepted[: int(count * 0.8)]
        chosen = {r["image_id"] for r in primary}
        # Fill with samples near the actual combined rejection boundary (both sides).
        boundary = sorted(
            [r for r in group if r["image_id"] not in chosen],
            key=lambda r: abs(r["acceptance_margin"]),
        )
        picked = spaced_selection(primary + boundary, count)
        mined.extend(picked)
        stats[source] = {
            "candidates": len(group),
            "v3_false_accepts": len(accepted),
            "selected_mined_accepted": sum(r["accepted"] for r in picked),
            "selected_mined_semi_hard": sum(not r["accepted"] for r in picked),
            "random_accepted": sum(
                r["accepted"] for r in random if r["source"] == source
            ),
        }
    assert len(random) == len(mined) == p["negative_selection"]["count"]
    for name, sample in [("random", random), ("mined", mined)]:
        # Put lists in identical source/time ordering; paired RNG streams map positions equally.
        sample.sort(key=lambda r: (r["source"], r["timestamp_seconds"]))
        save_json(
            EXP / f"{name}_selection.json",
            {
                "manifest_sha256": sha256(MANIFEST),
                "initial_checkpoint_sha256": sha256(CKPT),
                "selection_seed": p["selection_seed"],
                "role": "train only",
                "samples": sample,
            },
        )
    save_json(
        BASE / "mining.json",
        {
            "candidate_count": len(enriched),
            "false_accepts": sum(r["accepted"] for r in enriched),
            "per_source": stats,
            "selection_overlap": len(
                {r["image_id"] for r in random} & {r["image_id"] for r in mined}
            ),
            "mined_count": len(mined),
            "random_count": len(random),
            "reviewed_before_scoring": True,
            "validation_or_test_scored": False,
        },
    )
    write_csv(
        BASE / "mining.csv",
        [
            {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()}
            for r in enriched
        ],
    )
    print("MINED", json.loads((BASE / "mining.json").read_text()), flush=True)


def train(strategy, seed):
    """Same initialization, paired RNG streams and full fixed budget for B/C."""
    p = protocol()
    run = f"{strategy}_seed{seed}"
    folder = BASE / run
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / f"checkpoints/v4_phase2/{run}.pth"
    selected = json.loads((EXP / f"{strategy}_selection.json").read_text())
    assert selected["manifest_sha256"] == sha256(MANIFEST)
    assert selected["initial_checkpoint_sha256"] == sha256(CKPT)
    assert selected["selection_seed"] == p["selection_seed"]
    validate_selection(
        selected,
        cartoon("train"),
        p["negative_selection"]["count"],
        p["negative_selection"]["quota_per_train_film"],
    )
    if (folder / "complete.json").exists():
        completed = json.loads((folder / "complete.json").read_text())
        config = json.loads((folder / "training_config.json").read_text())
        assert completed["checkpoint_sha256"] == sha256(checkpoint)
        assert completed["protocol_sha256"] == sha256(EXP / "protocol.json")
        assert config["selection_sha256"] == sha256(EXP / f"{strategy}_selection.json")
        assert config["manifest_sha256"] == sha256(MANIFEST)
        return
    seed_all(seed)
    model, saved = load_model(CKPT)
    cfg = p["training"]
    trainrows = items("train")
    valrows = items("val")
    easy, hard = original_oe()
    oe = easy + hard + selected["samples"]
    n = len(trainrows)
    assert n == 345
    idds = Images(trainrows, 128, True)
    oeds = Images(oe, 128, True)
    ideval = Images(trainrows, 128)
    vds = Images(valrows, 128)
    weights = (
        [0.25 / len(easy)] * len(easy)
        + [0.25 / len(hard)] * len(hard)
        + [0.5 / len(selected["samples"])] * len(selected["samples"])
    )
    sampler = WeightedRandomSampler(
        weights,
        n,
        replacement=True,
        generator=torch.Generator().manual_seed(seed + 1000),
    )
    iloader = DataLoader(
        idds,
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 2000),
    )
    oloader = DataLoader(oeds, batch_size=32, sampler=sampler)
    yl = np.array([r["label"] for r in trainrows])
    vl = np.array([r["label"] for r in valrows])
    domains = np.array([r["domain"] for r in valrows])
    vid = domains == "id"
    model.eval()
    initial, _ = features(model, ideval)
    initialv, _ = features(model, vds)
    m_in = float(np.quantile(energy(initial, [0, 1], 1), 0.9))
    m_out = m_in + 2
    initialacc = float((initialv[vid].argmax(1) == vl[vid]).mean())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    best = None
    history = []
    steps = 0
    config = {
        **cfg,
        "strategy": strategy,
        "seed": seed,
        "initial_checkpoint_sha256": sha256(CKPT),
        "protocol_sha256": sha256(EXP / "protocol.json"),
        "selection_sha256": sha256(EXP / f"{strategy}_selection.json"),
        "manifest_sha256": sha256(MANIFEST),
        "margin_in": m_in,
        "margin_out": m_out,
        "original_easy": len(easy),
        "original_hard_and_feedback": len(hard),
        "new_cartoon": len(selected["samples"]),
        "train_set_sha256": set_hash(trainrows),
        "val_set_sha256": set_hash(valrows),
        "initial_val_id_accuracy": initialacc,
    }
    save_json(folder / "training_config.json", config)
    for epoch in range(1, cfg["epochs"] + 1):
        start = time.perf_counter()
        model.train()
        loss_sum = 0
        ce_sum = 0
        oe_sum = 0
        correct = 0
        seen = 0
        for (x, y), (ox, _) in zip(iloader, oloader):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model.forward_outputs(torch.cat([x, ox]))
            il = logits[: len(x)]
            ol = logits[len(x) :]
            ce = torch.nn.functional.cross_entropy(il, y)
            ei = -torch.logsumexp(il, 1)
            eo = -torch.logsumexp(ol, 1)
            oel = (
                torch.relu(ei - m_in).square().mean()
                + torch.relu(m_out - eo).square().mean()
            )
            loss = ce + cfg["oe_weight"] * oel
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            steps += 1
            loss_sum += loss.item() * len(y)
            ce_sum += ce.item() * len(y)
            oe_sum += oel.item() * len(y)
            correct += (il.argmax(1) == y).sum().item()
            seen += len(y)
        model.eval()
        tl, te = features(model, ideval)
        ll, ee = features(model, vds)
        gp = fit_gaussian(te, yl, [0, 1], False)
        d = distance(ee, gp)
        e = energy(ll, [0, 1], 1)
        ac = (d <= np.quantile(d[vid], 0.99)) & (e <= np.quantile(e[vid], 0.99))
        tar = float(ac[vid].mean())
        acc = float((ll[vid].argmax(1) == vl[vid]).mean())
        e2e = float(((ll[vid].argmax(1) == vl[vid]) & ac[vid]).mean())
        fars = {
            domain: float(ac[domains == domain].mean())
            for domain in ["easy", "hard", "cartoon"]
        }
        key = (
            tar >= 0.95 and acc >= initialacc - 0.03,
            -max(fars.values()),
            -float(np.mean(list(fars.values()))),
            e2e,
            acc,
        )
        row = {
            "epoch": epoch,
            "steps": steps,
            "train_loss": loss_sum / seen,
            "train_ce": ce_sum / seen,
            "train_oe": oe_sum / seen,
            "train_accuracy": correct / seen,
            "val_loss_id": float(
                torch.nn.functional.cross_entropy(
                    torch.from_numpy(ll[vid]), torch.from_numpy(vl[vid])
                )
            ),
            "val_id_accuracy": acc,
            "val_id_accept": tar,
            "val_id_end_to_end_accuracy": e2e,
            "val_far": fars,
            "seconds": time.perf_counter() - start,
        }
        history.append(row)
        if best is None or key > best:
            best = key
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "class_to_idx": {"nailong": 0, "naiwa": 1},
                    "preprocessing": saved["preprocessing"],
                    "config": config,
                },
                checkpoint,
            )
        save_json(folder / "training_history.json", history)
        print(run, row, flush=True)
    save_json(
        folder / "complete.json",
        {
            "checkpoint_sha256": sha256(checkpoint),
            "best_epoch": torch.load(checkpoint, weights_only=True)["epoch"],
            "total_steps": steps,
            "epochs_executed": len(history),
            "protocol_sha256": sha256(EXP / "protocol.json"),
        },
    )
    check_production()


def extract(run, split):
    """Real model outputs; test requires the all-model evaluation lock."""
    p = protocol()
    folder = BASE / run / "features"
    folder.mkdir(parents=True, exist_ok=True)
    if split == "test":
        lock = json.loads((BASE / "evaluation_lock.json").read_text())
        assert lock["manifest_sha256"] == sha256(MANIFEST)
        for name, digests in lock["models"].items():
            for path, digest in digests.items():
                assert sha256(ROOT / path) == digest
    checkpoint = (
        CKPT
        if run in ["v3_baseline", "v3_recalibrated"]
        else ROOT / f"checkpoints/v4_phase2/{run}.pth"
    )
    model, _ = load_model(checkpoint)
    rr = items(split)
    l, e = features(model, Images(rr, 128))
    np.savez_compressed(
        folder / f"{split}.npz",
        logits=l,
        embedding=e,
        labels=np.array([r["label"] for r in rr]),
    )
    save_json(
        folder / f"{split}.json",
        {
            "rows": rr,
            "class_to_idx": {"nailong": 0, "naiwa": 1},
            "checkpoint_sha256": sha256(checkpoint),
            "set_sha256": set_hash(rr),
            "input_size": 128,
        },
    )


def calibrate(run):
    """Train-ID Gaussian fit, then shared validation-only detector search."""
    folder = BASE / run
    train = np.load(folder / "features/train.npz")
    val = np.load(folder / "features/val.npz")
    meta = json.loads((folder / "features/val.json").read_text())
    train_meta = json.loads((folder / "features/train.json").read_text())
    assert train_meta["checkpoint_sha256"] == meta["checkpoint_sha256"]
    rows = meta["rows"]
    domains = np.array([r["domain"] for r in rows])
    vid = domains == "id"
    label = val["labels"]
    pred = val["logits"].argmax(1)
    candidates = []
    gps = {}

    def key(ac):
        return (
            max(float(ac[domains == d].mean()) for d in ["easy", "hard", "cartoon"]),
            float(ac[~vid].mean()),
            -float(((pred == label) & ac)[vid].mean()),
            -float(ac[vid].mean()),
        )

    def record(method, ac, score, params):
        return {
            "method": method,
            "rank": list(key(ac)),
            "validation_id_accept": float(ac[vid].mean()),
            "validation_fars": {
                d: float(ac[domains == d].mean()) for d in ["easy", "hard", "cartoon"]
            },
            "parameters": params,
            **rank_ood(score, vid),
        }

    for normalized in [False, True]:
        gp = fit_gaussian(train["embedding"], train["labels"], [0, 1], normalized)
        gps[normalized] = gp
        d = distance(val["embedding"], gp)
        ds = max(float(d[vid].std()), 1e-6)
        dgrid = quantiles(d[vid])
        best = None
        for dt in dgrid:
            ac = d <= dt
            if ac[vid].mean() < 0.95:
                continue
            if best is None or key(ac) < best[0]:
                best = (key(ac), ac, dt)
        candidates.append(
            record(
                "mahalanobis",
                best[1],
                (d - best[2]) / ds,
                {
                    "normalize_embedding": normalized,
                    "distance_threshold": float(best[2]),
                    "distance_scale": ds,
                },
            )
        )
        for temp in [0.5, 1.0, 2.0, 5.0]:
            e = energy(val["logits"], [0, 1], temp)
            es = max(float(e[vid].std()), 1e-6)
            egrid = quantiles(e[vid])
            best = None
            for et in egrid:
                ac = e <= et
                if ac[vid].mean() < 0.95:
                    continue
                if best is None or key(ac) < best[0]:
                    best = (key(ac), ac, et)
            if not normalized:
                candidates.append(
                    record(
                        "energy",
                        best[1],
                        (e - best[2]) / es,
                        {
                            "energy_temperature": temp,
                            "energy_threshold": float(best[2]),
                            "energy_scale": es,
                        },
                    )
                )
            for logic in ["or", "and"]:
                best = None
                for dt in dgrid:
                    for et in egrid:
                        ac = (
                            ((d <= dt) & (e <= et))
                            if logic == "or"
                            else ((d <= dt) | (e <= et))
                        )
                        if ac[vid].mean() < 0.95:
                            continue
                        k = key(ac)
                        if best is None or k < best[0]:
                            best = (k, ac, float(dt), float(et))
                _, ac, dt, et = best
                params = {
                    "normalize_embedding": normalized,
                    "distance_threshold": dt,
                    "energy_threshold": et,
                    "energy_temperature": temp,
                    "distance_scale": ds,
                    "energy_scale": es,
                    "reject_logic": logic,
                }
                s = combined_score(d, e, dt, et, logic, ds, es)
                candidates.append(record("distance_energy", ac, s, params))
    simplicity = {"mahalanobis": 0, "energy": 1, "distance_energy": 2}
    chosen = min(candidates, key=lambda r: (*r["rank"], simplicity[r["method"]]))
    par = chosen["parameters"]
    gp = gps[par.get("normalize_embedding", False)]
    er = min(
        [r for r in candidates if r["method"] == "energy"], key=lambda r: r["rank"]
    )["parameters"]
    d = distance(val["embedding"], gp)
    config = {
        **gp,
        "version": "v4-phase2",
        "method": chosen["method"],
        "class_mapping": {"nailong": 0, "naiwa": 1},
        "energy_logit_indices": [0, 1],
        "mahalanobis_threshold": par.get(
            "distance_threshold", float(np.quantile(d[vid], 0.95))
        ),
        "distance_scale": par.get("distance_scale", max(float(d[vid].std()), 1e-6)),
        "energy_temperature": par.get("energy_temperature", er["energy_temperature"]),
        "energy_threshold": par.get("energy_threshold", er["energy_threshold"]),
        "energy_scale": par.get("energy_scale", er["energy_scale"]),
        "reject_logic": par.get("reject_logic", "or"),
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "calibration_split": "val",
        "manifest_sha256": sha256(MANIFEST),
        "protocol_sha256": sha256(EXP / "protocol.json"),
    }
    save_json(
        folder / "calibration.json",
        {
            "selected": chosen,
            "candidates": candidates,
            "fit_split": "train ID only",
            "threshold_split": "val only",
            "test_loaded": False,
        },
    )
    save_json(folder / "ood_config.json", config)
    print("CALIBRATED", run, chosen["method"], chosen["validation_fars"], flush=True)


def freeze_evaluation():
    p = protocol()
    runs = [
        f"{s}_seed{seed}" for s in ["random", "mined"] for seed in p["training_seeds"]
    ]
    models = {}
    for run in runs:
        paths = [
            ROOT / f"checkpoints/v4_phase2/{run}.pth",
            BASE / run / "ood_config.json",
            BASE / run / "complete.json",
        ]
        models[run] = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    models["v3_recalibrated"] = {
        str(CKPT.relative_to(ROOT)): sha256(CKPT),
        str((BASE / "v3_recalibrated/ood_config.json").relative_to(ROOT)): sha256(
            BASE / "v3_recalibrated/ood_config.json"
        ),
    }
    models["v3_baseline"] = {
        str(CKPT.relative_to(ROOT)): sha256(CKPT),
        "web/ood_config.json": sha256(ROOT / "web/ood_config.json"),
    }
    lock = {
        "models": models,
        "manifest_sha256": sha256(MANIFEST),
        "protocol_sha256": sha256(EXP / "protocol.json"),
        "new_test_model_outputs_seen_before_lock": False,
    }
    target = BASE / "evaluation_lock.json"
    if target.exists():
        assert (
            json.loads(target.read_text()) == lock
        ), "Evaluation already locked with different parameters"
    else:
        save_json(target, lock)


def evaluate(run):
    """One locked test protocol, with decisions and sample-level audit trail."""
    folder = BASE / run
    data = np.load(folder / "features/test.npz")
    meta = json.loads((folder / "features/test.json").read_text())
    rows = meta["rows"]
    cfg = json.loads(
        (
            ROOT / "web/ood_config.json"
            if run == "v3_baseline"
            else folder / "ood_config.json"
        ).read_text()
    )
    d, e, s = scores(data["logits"], data["embedding"], cfg)
    ac = s <= 0
    pred = data["logits"].argmax(1)
    labels = data["labels"]
    domains = np.array([r["domain"] for r in rows])
    vid = domains == "id"
    final = np.where(ac, pred, 2)
    stats = {
        "model": run,
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "test_set_sha256": meta["set_sha256"],
        "manifest_sha256": sha256(MANIFEST),
        "method": cfg["method"],
        "reject_logic": cfg["reject_logic"],
        "closed_id_accuracy": float((pred[vid] == labels[vid]).mean()),
        "id_accuracy_including_rejection": float((final[vid] == labels[vid]).mean()),
        "id_false_reject_rate": float((~ac[vid]).mean()),
        "id_rejected": int((~ac[vid]).sum()),
        "id_closed_misclassified": int((pred[vid] != labels[vid]).sum()),
        "id_accepted_misclassified": int(((pred != labels) & ac & vid).sum()),
        "id_macro_f1_including_rejection": float(
            f1_score(
                labels[vid], final[vid], labels=[0, 1], average="macro", zero_division=0
            )
        ),
        **rank_ood(s, vid),
        "per_domain": {},
        "subtypes": {},
        "test_used_to_tune": False,
    }
    for i, cl in enumerate(["nailong", "naiwa"]):
        mask = vid & (labels == i)
        stats[cl] = {
            "n": int(mask.sum()),
            "closed_accuracy": float((pred[mask] == i).mean()),
            "recall_including_rejection": float((final[mask] == i).mean()),
            "rejected": int((~ac[mask]).sum()),
        }
    for domain in ["easy", "hard", "cartoon"]:
        m = domains == domain
        stats["per_domain"][domain] = {
            "n": int(m.sum()),
            "accepted": int(ac[m].sum()),
            "far": float(ac[m].mean()),
            **rank_ood(s[m | vid], vid[m | vid]),
        }
    for category in sorted(
        {r.get("category", "") for r in rows if r["domain"] != "id"}
    ):
        mask = np.array(
            [r.get("category", "") == category and r["domain"] != "id" for r in rows]
        )
        stats["subtypes"][category or "easy_photos"] = {
            "n": int(mask.sum()),
            "accepted": int(ac[mask].sum()),
            "far": float(ac[mask].mean()),
        }
    # Same selected detector thresholds, signal removal diagnostics (not re-tuned on test).
    stats["signal_rejects"] = {
        domain: {
            "distance_only": int(
                (
                    (d > cfg["mahalanobis_threshold"])
                    & (e <= cfg["energy_threshold"])
                    & (domains == domain)
                ).sum()
            ),
            "energy_only": int(
                (
                    (e > cfg["energy_threshold"])
                    & (d <= cfg["mahalanobis_threshold"])
                    & (domains == domain)
                ).sum()
            ),
            "both": int(
                (
                    (e > cfg["energy_threshold"])
                    & (d > cfg["mahalanobis_threshold"])
                    & (domains == domain)
                ).sum()
            ),
        }
        for domain in ["id", "easy", "hard", "cartoon"]
    }
    out = []
    for i, r in enumerate(rows):
        out.append(
            {
                **{
                    k: r.get(k, "")
                    for k in [
                        "path",
                        "image_id",
                        "domain",
                        "source",
                        "identity_group",
                        "original_group",
                        "shot_group",
                        "category",
                    ]
                },
                "true_label": int(labels[i]),
                "predicted_id": int(pred[i]),
                "accepted": bool(ac[i]),
                "correct": bool(final[i] == labels[i]),
                "mahalanobis": float(d[i]),
                "energy": float(e[i]),
                "ood_score": float(s[i]),
                "id_confidence": float(probabilities(data["logits"][i : i + 1]).max()),
            }
        )
    write_csv(folder / "predictions.csv", out)
    save_json(folder / "metrics.json", stats)
    print("EVALUATED", run, stats["per_domain"], stats["id_rejected"], flush=True)
    return stats
