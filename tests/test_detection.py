from itertools import permutations
from types import SimpleNamespace

import numpy as np
import pytest

from apexfocus.detection import Detector, detections, extract, heuristic_action, merge, regions, to_image
from apexfocus.evaluation import exact_oracle, match, operating, summarize
from apexfocus.feasibility import gate_decision, selected_run, validate_config

SETTINGS = {"score_floor": .001, "confidence": .25, "nms_iou": .5, "match_iou": .5,
            "alpha": 1., "max_det": 1000}


def truth(xyxy=(0, 0, 10, 10), class_id=0, area=.01):
    return {"xyxy": list(xyxy), "class_id": class_id, "area_fraction": area}


@pytest.mark.parametrize("width,height", [(11, 7), (3, 2), (640, 481), (6304, 4304)])
def test_grid_covers_every_original_pixel_once(width, height):
    grid = regions(width, height)
    coverage = np.zeros((height, width), dtype=np.uint8)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for x1, y1, x2, y2 in grid:
        coverage[y1:y2, x1:x2] += 1
        crop = extract(image, (x1, y1, x2, y2))
        assert crop.shape == (y2 - y1, x2 - x1, 3)
        assert crop.flags.c_contiguous
    assert np.all(coverage == 1)
    assert grid[-1][2:] == (width, height)


def test_crop_mapping_adds_only_offset_at_odd_border():
    region = regions(11, 7)[-1]
    assert region == (7, 3, 11, 7)
    rows = detections([[0., 0., 4., 4., .8, 2]])
    np.testing.assert_array_equal(to_image(rows, region), [[7., 3., 11., 7., .8, 2]])
    np.testing.assert_array_equal(rows[0, :4], [0, 0, 4, 4])
    assert to_image([], region).shape == (0, 6)
    with pytest.raises(ValueError, match="dimensions"):
        to_image([[0, 0, 5, 4, .8, 0]], region)
    with pytest.raises(ValueError):
        regions(2, 5)


def test_class_aware_nms_ties_and_duplicate_sources_are_order_independent():
    sources = [detections([[0, 0, 10, 10, .8, 0], [1, 0, 11, 10, .8, 0]]),
               detections([[0, 0, 10, 10, .8, 1]]), detections([[0, 0, 10, 10, .8, 0]])]
    expected = [[0, 0, 10, 10, .8, 0], [0, 0, 10, 10, .8, 1]]
    for order in permutations(sources):
        np.testing.assert_array_equal(merge(list(order), .5), expected)
        np.testing.assert_array_equal(merge([rows[::-1] for rows in order], .5), expected)
    assert merge([], .5).shape == (0, 6)


def test_union_restores_box_suppressed_in_an_intermediate_merge():
    # C suppresses A; A suppressed B, but C does not overlap B enough to suppress it.
    a = detections([[4, 0, 14, 10, .8, 0]])
    b = detections([[8, 0, 18, 10, .7, 0]])
    c = detections([[0, 0, 10, 10, .9, 0]])
    assert len(merge([a, b], .3)) == 1
    assert len(merge([merge([a, b], .3), c], .3)) == 1
    assert len(merge([a, b, c], .3)) == 2


@pytest.mark.parametrize("row", [[0, 0, 0, 1, .8, 0], [0, 0, 1, 1, float('nan'), 0],
                                 [0, 0, 1, 1, .8, 5], [0, 0, 1, 1, .8, .5]])
def test_invalid_detector_output_fails(row):
    with pytest.raises(ValueError):
        detections([row])


def test_matching_is_class_aware_one_to_one_and_confidence_ordered():
    boxes = [[0, 0, 10, 10, .6, 0], [0, 0, 10, 10, .9, 1], [0, 0, 10, 10, .8, 0]]
    rows, indices = match(boxes, [truth()], .5)
    assert rows[:, 4].tolist() == [.9, .8, .6]
    assert indices.tolist() == [-1, 0, -1]
    counts, _, _ = operating(boxes, [truth()], .7, .5, 2.)
    assert counts == {"tp": 1, "fp": 1, "utility": -1.}


def test_utility_difference_counts_correct_detections_lost_after_merge():
    coarse = [[0, 0, 10, 10, .8, 0]]
    candidate = [[4, 0, 14, 10, .9, 0]]
    before = operating(coarse, [truth()], .25, .5, 1.)[0]["utility"]
    after = operating(merge([coarse, candidate], .3), [truth()], .25, .5, 1.)[0]["utility"]
    assert after - before == -2.  # One lost TP plus one new FP.


def test_exact_oracle_searches_15_pairs_and_breaks_ties_lexicographically():
    candidates = [[], [[0, 0, 10, 10, .8, 0]], [], [], [[20, 0, 30, 10, .7, 1]], []]
    pair, final, scores = exact_oracle([], candidates, [truth(), truth((20, 0, 30, 10), 1)], SETTINGS)
    assert len(scores) == 15
    assert pair == (1, 4)
    assert scores[pair] == 2 and len(final) == 2
    pair, _, scores = exact_oracle([], [[] for _ in range(6)], [], SETTINGS)
    assert pair == (0, 1) and set(scores.values()) == {0.}


def test_ap_small_recall_and_absent_classes_have_declared_behavior():
    samples = [("a", [[0, 0, 10, 10, .8, 0]], [truth()])]
    result = summarize(samples, SETTINGS, .02)
    assert result["map50_95"] == 1.
    assert result["small_recall"] == 1.
    assert result["ap_classes"] == [0]
    # A higher-confidence FP before the TP gives precision-envelope AP = .5.
    samples[0][1].append([20, 0, 30, 10, .9, 0])
    result = summarize(samples, SETTINGS, .005)
    assert result["map50_95"] == .5
    assert result["small_recall"] is None
    assert result["precision"] == .5
    assert result["fp_per_image"] == 1.
    result = summarize([("a", [], [])], SETTINGS, .02)
    assert result["map50_95"] is None and result["recall"] is None


def test_adapter_uses_crop_pixels_and_explicit_prediction_settings():
    class Model:
        def predict(self, **kwargs):
            assert kwargs["rect"] is False and kwargs["half"] is False
            assert kwargs["imgsz"] == 640 and kwargs["source"].shape == (7, 11, 3)
            class Tensor:
                def detach(self): return self
                def cpu(self): return self
                def numpy(self): return np.array([[1, 2, 10, 7, .8, 0]])
            return [SimpleNamespace(boxes=SimpleNamespace(data=Tensor()))]
    detector = Detector.__new__(Detector)
    detector.model, detector.device, detector.settings = Model(), "cpu", SETTINGS
    rows = detector.predict(np.zeros((7, 11, 3), dtype=np.uint8), 640)
    np.testing.assert_array_equal(rows[0, :4], [1, 2, 10, 7])


def test_second_action_uses_first_crop_observations_and_runs_sequentially():
    calls = []
    class FakeDetector:
        def predict(self, image, size):
            calls.append(image.shape)
            return detections([[1, 1, 3, 3, .3, 0]]) if len(calls) == 2 else detections([])
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    observations = []
    def choose(current, grid, selected):
        observations.append((len(current), selected))
        return 0 if not selected else 1
    final, actions = selected_run(FakeDetector(), image, SETTINGS, choose)
    assert observations == [(0, ()), (1, (0,))]
    assert calls == [(8, 12, 3), (4, 4, 3), (4, 4, 3)]
    assert actions == [0, 1] and len(final) == 1
    assert heuristic_action([], regions(12, 8), (0,), .25) == 1
    assert heuristic_action([[5, 1, 7, 3, .3, 0]], regions(12, 8), (0,), .25) == 1
    with pytest.raises(ValueError, match="distinct"):
        selected_run(FakeDetector(), image, SETTINGS, lambda *_: 0)


def test_sequential_runner_retains_original_sources_for_final_union():
    outputs = iter([[[14, 0, 26, 10, .8, 0]], [[8, 0, 20, 10, .7, 0]], [[0, 0, 12, 10, .9, 0]]])
    class FakeDetector:
        def predict(self, image, size): return detections(next(outputs))
    final, actions = selected_run(FakeDetector(), np.zeros((20, 60, 3), dtype=np.uint8),
                                  dict(SETTINGS, nms_iou=.3), lambda current, grid, selected: len(selected))
    assert actions == [0, 1]
    np.testing.assert_array_equal(final, [[20, 0, 32, 10, .9, 0], [8, 0, 20, 10, .7, 0]])


def test_gate_cannot_pass_on_cpu_or_subset_and_enforces_compute():
    import yaml
    from pathlib import Path
    config = yaml.safe_load(Path("configs/experiment.yaml").read_text())
    validate_config(config)
    base = {"small_recall": .2, "fp_per_image": .1, "utility_per_image": 1., "p95_ms": 20.}
    table = {"full_640": base, "full_1280": dict(base, small_recall=.3, p95_ms=80.),
             "oracle_2": dict(base, small_recall=.5, utility_per_image=2., p95_ms=90.)}
    assert gate_decision(table, config)["status"] == "pass"
    assert gate_decision(table, dict(config, device="cpu"))["status"] == "pending"
    assert gate_decision(table, dict(config, limit=2))["status"] == "pending"
    table["oracle_2"]["p95_ms"] = 101.
    assert gate_decision(table, config)["status"] == "fail"
    with pytest.raises(ValueError):
        validate_config(dict(config, detector=dict(SETTINGS, score_floor=.3)))
