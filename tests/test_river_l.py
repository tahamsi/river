import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from river import build_baseline
from river.baselines import BaselineUnavailable
from river.learned.checkpoint import load_checkpoint, save_checkpoint
from river.learned.data import collate_examples, load_jsonl, permute_evidence
from river.learned.features import PROHIBITED_TRUTH_FEATURES, RiverLFeatureConfig, RiverLFeatureExtractor
from river.learned.losses import invariance_loss, update_loss
from river.learned.model import DeepSetsRIVERL, RiverLModelConfig
from river.learned.train import set_seed, train_from_config


FIXTURE = Path("river_bench/processed/river_bench_unit.jsonl")


def _examples(limit=4):
    return load_jsonl(FIXTURE)[:limit]


def _model_and_batch(examples=None):
    extractor = RiverLFeatureExtractor()
    batch = collate_examples(examples or _examples(), extractor)
    model = DeepSetsRIVERL(RiverLModelConfig(feature_dim=extractor.feature_dim, hidden_dim=12))
    return model, batch, extractor


def test_forward_pass():
    model, batch, _extractor = _model_and_batch()
    outputs = model(batch.evidence_features, batch.evidence_mask, batch.hypothesis_mask)

    assert outputs["hypothesis_logits"].shape[:2] == batch.hypothesis_mask.shape
    assert outputs["state_logits"].shape == (len(batch.example_ids), 5)
    assert outputs["confidence"].shape == (len(batch.example_ids),)
    assert torch.all((outputs["confidence"] >= 0) & (outputs["confidence"] <= 1))


def test_batching_and_masks():
    examples = [item for item in load_jsonl(FIXTURE) if item["intervention"] == "evidence_removal"]
    _model, batch, _extractor = _model_and_batch(examples)

    assert batch.evidence_features.ndim == 4
    assert batch.hypothesis_mask.sum().item() == 2
    # The removal variant keeps a residual on-topic chunk and loses only the
    # decisive support, so it is shorter than the control rather than empty.
    control, removed = batch.evidence_mask[0].sum().item(), batch.evidence_mask[1].sum().item()
    assert 0 < removed < control
    assert batch.selected_targets.tolist() == [0, -100]


def test_permutation_equivalent_logits_within_tolerance():
    set_seed(7)
    model, batch, _extractor = _model_and_batch()
    model.eval()
    permuted = permute_evidence(batch, seed=99)

    with torch.no_grad():
        left = model(batch.evidence_features, batch.evidence_mask, batch.hypothesis_mask)
        right = model(permuted.evidence_features, permuted.evidence_mask, permuted.hypothesis_mask)

    torch.testing.assert_close(left["state_logits"], right["state_logits"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(left["hypothesis_logits"], right["hypothesis_logits"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(left["confidence"], right["confidence"], atol=1e-6, rtol=1e-6)


def test_prohibited_features_absent():
    standard = RiverLFeatureExtractor()
    negative = RiverLFeatureExtractor(RiverLFeatureConfig(include_arrival_feature=True))

    assert PROHIBITED_TRUTH_FEATURES.isdisjoint(standard.feature_names)
    assert "arrival_recency_negative_control" not in standard.feature_names
    assert "arrival_recency_negative_control" in negative.feature_names


def test_checkpoint_round_trip(tmp_path):
    set_seed(11)
    model, batch, extractor = _model_and_batch()
    path = tmp_path / "river_l.pt"
    save_checkpoint(path, model, extractor, manifest={"seed": 11})
    loaded, loaded_extractor, manifest = load_checkpoint(path)

    assert loaded_extractor.feature_names == extractor.feature_names
    assert manifest["seed"] == 11
    with torch.no_grad():
        original = model(batch.evidence_features, batch.evidence_mask, batch.hypothesis_mask)
        restored = loaded(batch.evidence_features, batch.evidence_mask, batch.hypothesis_mask)
    torch.testing.assert_close(original["state_logits"], restored["state_logits"])


def test_genuine_update_loss_direction():
    example = [
        item
        for item in load_jsonl(FIXTURE)
        if item["intervention"] == "supported_genuine_update"
        and item["gold_hypothesis_id"] == item["intervention_hypothesis_id"]
    ][:1]
    batch = collate_examples(example, RiverLFeatureExtractor())
    target = int(batch.selected_targets.item())
    wrong = 1 - target
    good_hyp = torch.full((1, 2), -3.0)
    bad_hyp = torch.full((1, 2), -3.0)
    good_hyp[0, target] = 4.0
    bad_hyp[0, wrong] = 4.0
    good = {"hypothesis_logits": good_hyp, "state_logits": torch.tensor([[4.0, -1.0, -1.0, -1.0, -1.0]]), "confidence": torch.ones(1)}
    bad = {"hypothesis_logits": bad_hyp, "state_logits": torch.tensor([[4.0, -1.0, -1.0, -1.0, -1.0]]), "confidence": torch.ones(1)}

    assert update_loss(good, batch) < update_loss(bad, batch)


def test_invariance_loss_direction():
    _model, batch, _extractor = _model_and_batch()
    base = {
        "hypothesis_logits": torch.tensor([[1.0, 0.0], [0.5, -0.5], [0.0, -1.0], [0.0, 0.0]]),
        "state_logits": torch.zeros(4, 5),
        "confidence": torch.ones(4) * 0.5,
    }
    same = {key: value.clone() for key, value in base.items()}
    different = {key: value.clone() for key, value in base.items()}
    different["state_logits"] = different["state_logits"] + 1.0
    different["hypothesis_logits"] = different["hypothesis_logits"] - 0.5

    assert invariance_loss(base, same, batch) < invariance_loss(base, different, batch)


def test_deterministic_seed_behavior():
    set_seed(13)
    first = DeepSetsRIVERL(RiverLModelConfig(feature_dim=4, hidden_dim=8))
    first_params = [item.detach().clone() for item in first.parameters()]
    set_seed(13)
    second = DeepSetsRIVERL(RiverLModelConfig(feature_dim=4, hidden_dim=8))

    for left, right in zip(first_params, second.parameters(), strict=True):
        torch.testing.assert_close(left, right)


def test_invalid_checkpoint_failure(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"not": "river-l"}, path)

    with pytest.raises(ValueError, match="not a valid RIVER-L checkpoint"):
        load_checkpoint(path)


def test_river_l_baseline_requires_checkpoint():
    with pytest.raises(BaselineUnavailable):
        build_baseline("river_l")


def test_tiny_synthetic_overfit_engineering_validation(tmp_path):
    rows = load_jsonl(FIXTURE)[:2]
    fixture = tmp_path / "tiny.jsonl"
    fixture.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "engineering_validation_only": True,
                "seed": 123,
                "features": {"use_temporal_features": True, "include_arrival_feature": False},
                "model": {"hidden_dim": 16, "dropout": 0.0},
                "loss": {"lambda_inv": 1.0, "lambda_upd": 1.0, "lambda_cal": 0.1},
                "learning_rate": 0.05,
                "batch_size": 2,
                "epochs": 15,
                "early_stopping_patience": 5,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "river_l.pt"
    manifest_path = tmp_path / "manifest.json"
    card_path = tmp_path / "model_card.md"

    manifest = train_from_config(
        SimpleNamespace(
            train=fixture,
            validation=fixture,
            config=config,
            output=output,
            manifest=manifest_path,
            model_card=card_path,
            seed=123,
            epochs=None,
        )
    )

    assert output.exists()
    assert manifest_path.exists()
    assert card_path.exists()
    assert manifest["engineering_validation_only"] is True
    assert manifest["epochs_ran"] >= 1
