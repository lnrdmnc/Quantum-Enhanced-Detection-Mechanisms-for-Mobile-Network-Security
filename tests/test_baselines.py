import numpy as np

from qdetect import baselines as B
from qdetect import data as D
from tests.test_data import fake_csv


def test_run_baselines_on_fake_split(tmp_path):
    p = tmp_path / "train_dataset.csv"
    fake_csv(p, n=6000, seed=0, bad_labels=0)
    df, _ = D.sample_per_class(p, 2500, seed=1)
    sp = D.make_split(df, n_train=300, n_train_large=800, n_val=200, n_test=400)
    P = D.prepare(sp)
    res = B.run_baselines(P, models=["Logistic Regression", "MLP small (10)", "SVM RBF"],
                          seeds=(1, 2), verbose=False)
    # LR e SVM deterministici: 1 seed; MLP: 2 seed; 2 insiemi di train
    assert len(res) == 2 * (1 + 2 + 1)
    assert (res["auroc"] > 0.5).all() and res["auroc"].max() > 0.85  # rssi e max_magnitude dipendono dall etichetta
    mlp = res[res["model"] == "MLP small (10)"]
    assert (mlp["n_params"] == 8 * 10 + 10 + 10 + 1).all()
    svm = res[res["model"] == "SVM RBF"]
    assert (svm["threshold"] == 0.0).all()
    s = B.summarize(res)
    assert set(s["n_seeds"]) == {1, 2}
    assert np.isfinite(s["auroc_mean"]).all()
