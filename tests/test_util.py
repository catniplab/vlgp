def test_save():
    import pathlib
    from vlgp.util import save
    fit = {'trials': {}, 'params': {}, 'config': {'path': "test_save.npy"}}
    save(fit, fit['config']['path'])
    path = pathlib.Path(fit['config']['path'])
    path.unlink()
