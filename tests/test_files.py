import numpy as np
import pytest

from qdetect import files as F

PATHS = [
    "active_scan/malicious/singletone/neg40dbm/5805mhz_jamming_neg40dbm_singletone_39.csv",
    "active_scan/malicious/gaussian_noise/10dbm/5805mhz_jamming_10dbm_gaussiannoise_79.csv",
    "active_scan/benign/floor/5ghz_activescan_floor_9.csv",
    "active_scan/benign/background/location2/5ghz_activescan_background_loc2_766.csv",
    "passive_scan/malicious/gaussian_noise/3dbm/2457mhz_jamming_3dbm_gaussiannoise_992.csv",
    "passive_scan/benign/floor/2.4ghz_passivescan_floor_558.csv",
    "passive_scan/benign/background/location3/5ghz_passivescan_background_loc3_9392.csv",
]


def test_parse_path():
    r = [F.parse_path(p) for p in PATHS]
    assert (r[0]["scan_mode"], r[0]["waveform"], r[0]["power_dbm"], r[0]["channel_mhz"]) == ("active", "single_tone", -40, 5805)
    assert (r[1]["waveform"], r[1]["power_dbm"]) == ("gaussian_noise", 10)
    assert (r[2]["benign_subtype"], r[2]["location"], r[2]["band"]) == ("floor", "chamber", "5ghz")
    assert np.isnan(r[2]["power_dbm"]) and r[2]["waveform"] == ""
    assert (r[3]["benign_subtype"], r[3]["location"]) == ("background", "location2")
    assert (r[4]["scan_mode"], r[4]["channel_mhz"], r[4]["band"], r[4]["power_dbm"]) == ("passive", 2457, "2.4ghz", 3)
    assert r[5]["band"] == "2.4ghz" and r[6]["location"] == "location3"


def test_parse_path_rejects_unknown():
    with pytest.raises(ValueError):
        F.parse_path("active_scan/weird/x/y.csv")


def test_build_manifest_skips_checkpoints(tmp_path):
    for p in PATHS:
        f = tmp_path / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("freq1,noise\n5805,-100\n")
    ck = tmp_path / "active_scan/benign/floor/.ipynb_checkpoints/5ghz_activescan_floor_9-checkpoint.csv"
    ck.parent.mkdir(parents=True)
    ck.write_text("x")
    m = F.build_manifest(tmp_path)
    assert len(m) == len(PATHS)
    assert list(m.columns) == F.MANIFEST_COLUMNS
    assert set(m.label) == {"benign", "malicious"}
