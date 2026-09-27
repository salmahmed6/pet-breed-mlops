import torch

from pet_breed_mlops.serving import predict_top_k


def test_predict_top_k_contract():
    logits = torch.tensor([[1.0, 0.5, 0.1]])
    label_map = {"a": 0, "b": 1, "c": 2}
    result = predict_top_k(logits, label_map, top_k=3, model_version="v1")
    assert len(result["predictions"]) == 3
    assert result["predictions"][0]["breed"] == "a"
    assert result["predictions"][0]["model_version"] == "v1"
    assert result["abstained"] is False


def test_temperature_changes_calibrated_confidence():
    logits = torch.tensor([[4.0, 1.0, 0.0]])
    label_map = {"a": 0, "b": 1, "c": 2}
    cold = predict_top_k(logits, label_map, temperature=0.5)
    warm = predict_top_k(logits, label_map, temperature=2.0)
    assert cold["predictions"][0]["calibrated_confidence"] > warm["predictions"][0]["calibrated_confidence"]
