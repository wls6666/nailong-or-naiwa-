"""Explicit membership checks: a matching manifest hash alone is insufficient."""

from collections import Counter


def validate_selection(document, training_rows, count, quota):
    allowed = {row["image_id"]: row for row in training_rows}
    sample = document["samples"]
    assert len(sample) == count
    assert len({row["image_id"] for row in sample}) == count
    for row in sample:
        assert row["image_id"] in allowed, "Non-training image in OE selection"
        reference = allowed[row["image_id"]]
        for key in ["path", "split", "label", "source", "identity_group", "sha256"]:
            assert row[key] == reference[key], ("Selection membership mismatch", key)
        assert row["split"] == "train" and row["label"] == 2
    counts = Counter(row["source"] for row in sample)
    assert set(counts) == {row["source"] for row in training_rows}
    assert all(value == quota for value in counts.values())
    for source in counts:
        times = sorted(
            row["timestamp_seconds"] for row in sample if row["source"] == source
        )
        assert all(b - a >= 2 for a, b in zip(times, times[1:]))
    return document
