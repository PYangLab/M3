import m3


def test_version_exposed():
    assert isinstance(m3.__version__, str)
    assert m3.__version__.count(".") >= 2
