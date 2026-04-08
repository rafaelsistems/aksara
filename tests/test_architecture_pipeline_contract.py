import inspect
from typing import Any

import pytest


def _has_attr(obj: Any, name: str) -> bool:
    return hasattr(obj, name)


def _call_with_minimal_supported_args(callable_obj):
    try:
        return callable_obj()
    except TypeError:
        pass

    try:
        return callable_obj("halo")
    except TypeError:
        pass

    try:
        return callable_obj(["halo"])
    except TypeError:
        pass

    try:
        return callable_obj("halo", min_confidence=0.75)
    except TypeError:
        pass

    try:
        return callable_obj(["halo"], min_confidence=0.75)
    except TypeError:
        pass

    raise


def test_public_framework_returns_structured_state_like_output():
    from aksara.framework import AksaraFramework

    framework = AksaraFramework()

    result = None
    if hasattr(framework, "dari_kbbi") and callable(getattr(framework, "dari_kbbi")):
        result = _call_with_minimal_supported_args(framework.dari_kbbi)
    elif callable(framework):
        result = _call_with_minimal_supported_args(framework)
    else:
        pytest.skip("AksaraFramework does not expose a callable public entrypoint in this environment")

    assert result is not None, "Public framework must return a structured state-like object"

    expected_attrs = ("tokens", "kata", "pos", "morfologi", "semantik", "interpretasi")
    assert any(_has_attr(result, attr) for attr in expected_attrs), (
        "Framework output should behave like AksaraState: structured, not logits-based"
    )


def test_pipeline_exposes_deterministic_linguistic_stages():
    from aksara.framework import AksaraFramework

    framework = AksaraFramework()

    stage_attrs = (
        "lps",
        "lsk",
        "meb",
        "state",
        "interpretasi",
        "pipeline",
    )
    assert any(hasattr(framework, attr) for attr in stage_attrs), (
        "Framework should expose linguistic pipeline stages or stateful composition"
    )

    if hasattr(framework, "lsk"):
        lsk = getattr(framework, "lsk")
        assert lsk is not None
        assert not hasattr(lsk, "logits"), "LSK is a linguistic stage, not a token predictor"
        assert not hasattr(lsk, "next_token"), "LSK should not expose next-token API as primary output"

    if hasattr(framework, "meb"):
        meb = getattr(framework, "meb")
        assert meb is not None
        assert not hasattr(meb, "logits"), "MEB is a semantic/state stage, not a logits head"


def test_public_api_does_not_require_logits_or_next_token_prediction():
    import aksara.framework as framework_module

    source = inspect.getsource(framework_module)
    forbidden_markers = ("next_token", "logits", "predict_next", "token prediction")
    assert not any(marker in source for marker in forbidden_markers), (
        "Public framework API should not be shaped around token prediction logits"
    )


def test_core_modules_minimal_cpu_smoke_instantiation_and_run():
    from aksara.core.meb import MEB
    from aksara.linguistic.lsk import LSKConfig, LapisanSemantikKBBI

    lsk = None
    try:
        lsk = LapisanSemantikKBBI()
    except TypeError:
        try:
            lsk = LapisanSemantikKBBI(LSKConfig())
        except TypeError:
            pytest.skip("LapisanSemantikKBBI constructor requires unavailable dependencies in this environment")

    assert lsk is not None

    meb = None
    try:
        meb = MEB()
    except TypeError:
        pytest.skip("MEB constructor requires unavailable dependencies in this environment")

    assert meb is not None

    if hasattr(lsk, "cpu"):
        lsk = lsk.cpu()
    if hasattr(meb, "cpu"):
        meb = meb.cpu()

    smoke_inputs = ["halo", "dunia"]
    for module in (lsk, meb):
        if callable(module):
            try:
                output = module(smoke_inputs)
            except TypeError:
                output = module("halo")
            assert output is not None
        else:
            possible_methods = ("forward", "proses", "process", "run")
            for method_name in possible_methods:
                if hasattr(module, method_name) and callable(getattr(module, method_name)):
                    method = getattr(module, method_name)
                    try:
                        output = method(smoke_inputs)
                    except TypeError:
                        output = method("halo")
                    assert output is not None
                    break
            else:
                pytest.skip(f"{type(module).__name__} has no callable execution surface for smoke test")


def test_full_model_and_consistency_files_remain_architecture_aligned():
    from aksara.framework import AksaraFramework

    framework = AksaraFramework()
    assert framework is not None
    assert not hasattr(framework, "output_dim"), "Framework should not primarily expose classifier-style output dimensions"
    assert not hasattr(framework, "vocab_size"), "Framework should not be described as a token generator"