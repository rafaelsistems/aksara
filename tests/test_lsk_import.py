import importlib


def test_lsk_module_imports():
    module = importlib.import_module("aksara.linguistic.lsk")
    assert hasattr(module, "KBBIStore")
    assert hasattr(module, "LSKConfig")
    assert hasattr(module, "LapisanSemantikKBBI")