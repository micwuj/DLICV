from src.utils.metrics import compute_metrics


def test_perfect_predictions():
    y_true = [0, 1, 2, 3, 0, 1]
    y_pred = [0, 1, 2, 3, 0, 1]
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_all_wrong():
    y_true = [0, 0, 0, 0]
    y_pred = [1, 1, 1, 1]
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.0
    assert m["macro_f1"] == 0.0


def test_imbalanced_accuracy_vs_balanced():
    # 9 z klasy 0, 1 z klasy 1. Predykcja: wszystko 0.
    # Accuracy = 0.9, balanced = 0.5 (klasa 0 ma 100% recall, klasa 1 ma 0%).
    y_true = [0] * 9 + [1]
    y_pred = [0] * 10
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.9
    assert m["balanced_accuracy"] == 0.5


def test_confusion_matrix_is_serialisable():
    import json
    m = compute_metrics([0, 1], [0, 1])
    s = json.dumps(m)
    parsed = json.loads(s)
    assert parsed["confusion_matrix"] == [[1, 0], [0, 1]]
