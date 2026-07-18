"""WS2: BADGE sampler - k-means++ picks, weighting, naive fallback."""
import numpy as np
import pytest

from app.badge import kmeanspp_pick, sample_crop_badge
from app.labels import ReviewStore


def test_kmeanspp_deterministic_under_seed():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(40, 8))
    w = rng.uniform(size=40)
    a = kmeanspp_pick(vecs, w, k=5, seed=123)
    b = kmeanspp_pick(vecs, w, k=5, seed=123)
    assert a == b and len(a) == 5 and len(set(a)) == 5


def test_kmeanspp_zero_weights_degenerates_to_spread():
    # Three tight clusters; zero weights must still yield a spread pick
    # (one per cluster), not an error or a collapsed pick.
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    vecs = np.concatenate([c + 0.01 * np.arange(6).reshape(3, 2)
                           for c in centers])
    picks = kmeanspp_pick(vecs, np.zeros(len(vecs)), k=3, seed=7)
    clusters = {int(np.argmin([np.linalg.norm(vecs[i] - c) for c in centers]))
                for i in picks}
    assert clusters == {0, 1, 2}


def test_kmeanspp_prefers_high_weight():
    # Two far points; one carries all the weight -> it is always first.
    vecs = np.array([[0.0, 0.0], [100.0, 100.0]])
    for seed in range(5):
        picks = kmeanspp_pick(vecs, np.array([0.0, 1.0]), k=1, seed=seed)
        assert picks == [1]


@pytest.fixture()
def crop_pool(tmp_path):
    """Six tiny crops under live_samples/, half with _uNN suffixes."""
    import cv2
    d = tmp_path / "live_samples" / "cam_a"
    d.mkdir(parents=True)
    names = ["1_person_40_u90.jpg", "2_person_41_u10.jpg", "3_car_55.jpg",
             "4_person_42_u70.jpg", "5_car_60.jpg", "6_person_39_u20.jpg"]
    rng = np.random.default_rng(1)
    for n in names:
        img = (rng.uniform(0, 255, size=(40, 24, 3))).astype(np.uint8)
        assert cv2.imwrite(str(d / n), img)
    return tmp_path


class FakeEmbedder:
    embedder_id = "fake-test"

    def embed(self, img, cls):
        return np.asarray([float(img[..., c].mean()) for c in range(3)],
                          dtype=np.float32)


def test_sample_crop_badge_end_to_end(crop_pool):
    from app.visual_search import SnapshotIndex

    store = ReviewStore(crop_pool / "reviews.json")
    idx = SnapshotIndex(crop_pool, embedder=FakeEmbedder())
    out = sample_crop_badge(store, crop_pool, batch=4, seed=42, index=idx)
    assert out and out["sampler"] == "badge"
    batch = out["batch"]
    assert len(batch) == 4
    # every entry has the naive-sampler contract + uncertainty
    for c in batch:
        assert c["path"].startswith("live_samples/")
        assert c["url"] == f"/snapshots/{c['path']}"
        assert 0.0 <= c["uncertainty"] <= 1.0
        assert c["remaining"] == 6
    # crops without _uNN fell back to the neutral weight, not excluded
    assert any(c["uncertainty"] == 0.5 for c in batch) or \
        all("_u" in c["path"] for c in batch)
    # reviewed crops leave the pool
    for c in batch:
        store.submit(c["path"], "correct", original_cls=c["cls"],
                     sampler=out["sampler"],
                     uncertainty_at_selection=c["uncertainty"])
    out2 = sample_crop_badge(store, crop_pool, batch=10, seed=42, index=idx)
    assert out2["batch"] and len(out2["batch"]) == 2
    assert not {c["path"] for c in out2["batch"]} & {c["path"] for c in batch}


def test_review_rows_carry_sampler_fields(tmp_path):
    p = tmp_path / "reviews.json"
    ReviewStore(p).submit("live_samples/cam/1_person_40_u90.jpg", "correct",
                          original_cls="person", sampler="badge",
                          uncertainty_at_selection=0.9)
    r = ReviewStore(p)._by_path["live_samples/cam/1_person_40_u90.jpg"]
    assert r.sampler == "badge" and r.uncertainty_at_selection == 0.9
