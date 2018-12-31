from vlgp.core import VLGP


def test_save_load(tmp_path):
    model1 = VLGP(n_factors=3)
    file = tmp_path / "model.pk"
    model1.save(file)

    model2 = VLGP.load(file)
    assert model1 == model2
