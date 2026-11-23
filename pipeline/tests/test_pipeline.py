import numpy as np
import pandas as pd
import pytest

from aslytics import landmarks as lm
from aslytics.asl_citizen import choose_exemplars, safe_name
from aslytics.captions import clean_text, expand_contractions, extract_video_id
from aslytics.manifest import COLUMNS, ManifestLog, load_manifest
from aslytics.normalize import fill_missing, normalize, passes_quality, quality
from aslytics.posefile import Clip, load_clip, save_clip
from aslytics.stitch import signing_bounds, stitch


def make_clip(frames=30, hands_up=(10, 20), offset=0.0, scale=1.0) -> Clip:
    """A synthetic signer: shoulders 0.2 apart, wrists raised during `hands_up`."""
    xyz = np.zeros((frames, lm.NUM_POINTS, 3), np.float32)
    xyz[:, lm.L_SHOULDER] = (0.6, 0.3, 0)
    xyz[:, lm.R_SHOULDER] = (0.4, 0.3, 0)
    xyz[:, lm.L_ELBOW] = (0.65, 0.5, 0)
    xyz[:, lm.R_ELBOW] = (0.35, 0.5, 0)
    xyz[:, lm.L_WRIST] = (0.65, 0.7, 0)
    xyz[:, lm.R_WRIST] = (0.35, 0.7, 0)
    xyz[:, lm.L_HIP] = (0.58, 0.8, 0)
    xyz[:, lm.R_HIP] = (0.42, 0.8, 0)
    xyz[hands_up[0]:hands_up[1], lm.R_WRIST] = (0.4, 0.35, 0)
    xyz[:, lm.LEFT_HAND] = xyz[:, lm.L_WRIST][:, None]
    xyz[:, lm.RIGHT_HAND] = xyz[:, lm.R_WRIST][:, None]
    xyz[:, lm.FACE] = (0.5, 0.15, 0)
    xyz = xyz * scale + offset
    return Clip(xyz, np.ones((frames, 4), bool))


def test_normalize_is_invariant_to_position_and_scale():
    a = normalize(make_clip())
    b = normalize(make_clip(offset=0.3, scale=2.5))
    np.testing.assert_allclose(a.xyz, b.xyz, atol=1e-4)
    width = np.linalg.norm(a.xyz[0, lm.L_SHOULDER] - a.xyz[0, lm.R_SHOULDER])
    assert width == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose((a.xyz[0, lm.L_SHOULDER] + a.xyz[0, lm.R_SHOULDER]) / 2, 0, atol=1e-5)


def test_fill_missing_interpolates_hand_gap():
    clip = make_clip()
    clip.xyz[5:8, lm.RIGHT_HAND] = np.nan
    clip.mask[5:8, 2] = False
    filled = fill_missing(clip)
    assert not np.isnan(filled.xyz).any()
    # The undetected hand sits on the body's wrist.
    np.testing.assert_allclose(filled.xyz[6, lm.RIGHT_HAND.start], filled.xyz[6, lm.R_WRIST])
    assert not filled.mask[6, 2]  # the mask still records the truth


def test_quality_gate_only_counts_frames_with_raised_hands():
    clip = make_clip(hands_up=(10, 20))
    clip.mask[:, 1:3] = False
    clip.mask[10:20, 2] = True  # hand tracked only while raised, as in real sign clips
    assert quality(clip)["hand_rate"] == 1.0
    assert passes_quality(quality(clip))
    clip.mask[10:18, 2] = False
    assert quality(clip)["hand_rate"] == pytest.approx(0.2)
    assert not passes_quality(quality(clip))
    assert quality(make_clip(hands_up=(0, 0)))["hand_rate"] == 0.0


def test_save_load_roundtrip(tmp_path):
    clip = normalize(make_clip())
    save_clip(clip, tmp_path / "c.npz")
    loaded = load_clip(tmp_path / "c.npz")
    np.testing.assert_allclose(loaded.xyz, clip.xyz, atol=2e-3)  # float16 on disk
    assert loaded.fps == clip.fps and loaded.mask.all()


def test_signing_bounds_finds_raised_hands():
    lo, hi = signing_bounds(normalize(make_clip(hands_up=(10, 20))))
    assert lo == 9 and hi == 21


def test_stitch_trims_and_bridges():
    a = normalize(make_clip(hands_up=(10, 20)))
    b = normalize(make_clip(hands_up=(8, 22)))
    out = stitch([a, b])
    assert out.xyz.shape[1:] == (lm.NUM_POINTS, 3)
    assert len(out) < len(a) + len(b)  # idle time between the signs was removed
    assert not np.isnan(out.xyz).any()
    # No jump between consecutive frames larger than the biggest one inside a source clip.
    step = np.abs(np.diff(out.xyz[:, lm.R_WRIST], axis=0)).max()
    source_step = np.abs(np.diff(a.xyz[:, lm.R_WRIST], axis=0)).max()
    assert step <= source_step + 1e-5


def test_stitch_single_and_empty():
    a = normalize(make_clip())
    assert len(stitch([a], trim=False)) == len(a)
    with pytest.raises(ValueError):
        stitch([])


def test_choose_exemplars_prefers_seed_signer():
    df = pd.DataFrame({
        "signer": ["P1", "P52", "P3", "P2"],
        "video_file": ["a.mp4", "b.mp4", "c.mp4", "d.mp4"],
        "gloss": ["APPLE", "APPLE", "APPLE", "HUG"],
        "asl_lex_code": ["x"] * 4,
        "split": ["train"] * 4,
    })
    picked = choose_exemplars(df, per_sign=2)
    apple = picked[picked["gloss"] == "APPLE"]
    assert list(apple["signer"]) == ["P52", "P1"]
    assert len(picked) == 3
    assert safe_name("SOCCER 2") == "SOCCER_2"


def test_manifest_resume_and_compact(tmp_path):
    row = dict.fromkeys(COLUMNS, "")
    row.update(clip_id="A__0", n_frames=10, body_rate=1.0, hand_rate=1.0, face_rate=1.0, ok=True)
    ManifestLog(tmp_path, "lexicon").append(row)
    log = ManifestLog(tmp_path, "lexicon")
    assert "A__0" in log.done
    log.append({**row, "clip_id": "B__0"})
    log.compact()
    assert list(load_manifest(tmp_path, "lexicon")["clip_id"]) == ["A__0", "B__0"]
    with pytest.raises(ValueError):
        log.append({"clip_id": "C"})


def test_caption_helpers():
    vid = "dQw4w9WgXcQ"
    assert extract_video_id(f"https://www.youtube.com/watch?v={vid}&t=3") == vid
    assert extract_video_id(f"https://youtu.be/{vid}") == vid
    assert extract_video_id(f"https://www.youtube.com/shorts/{vid}") == vid
    assert extract_video_id(vid) == vid
    assert extract_video_id("https://example.com/watch?v=abc") is None
    assert expand_contractions("I don't think it's here") == "I do not think it is here"
    assert clean_text("[Music] >> hello\nworld (applause)") == "hello world"
