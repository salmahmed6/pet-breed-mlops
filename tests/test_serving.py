import torch

from pet_breed_mlops.serving import predict_top_k, species_for_class


def test_species_mapping():
    assert species_for_class(0) == "cat"
    assert species_for_class(11) == "cat"
    assert species_for_class(12) == "dog"
    assert species_for_class(36) == "dog"


def test_top_three_contract_and_abstention():
    label_map = {"breed_0": 0, "breed_1": 1, "breed_2": 2, "breed_3": 3}
    logits = torch.tensor([[4.0, 3.0, 1.0, -1.0]])

    result = predict_top_k(
        logits,
        label_map,
        temperature=1.0,
        abstention_threshold=0.95,
        model_version="test-model",
    )

    assert len(result["predictions"]) == 3
    first = result["predictions"][0]
    assert first["breed"] == "breed_0"
    assert first["class_index"] == 0
    assert first["species"] == "cat"
    assert "calibrated_confidence" in first
    assert first["abstained"] is True
    assert first["model_version"] == "test-model"


def test_temperature_changes_calibrated_confidence():
    label_map = {"breed_0": 0, "breed_1": 1}
    logits = torch.tensor([[2.0, 1.0]])

    cold = predict_top_k(logits, label_map, temperature=0.5)
    warm = predict_top_k(logits, label_map, temperature=2.0)

    assert (
        cold["predictions"][0]["calibrated_confidence"]
        > warm["predictions"][0]["calibrated_confidence"]
    )
