"""Locating a source-built libRoadRunner via ``$BENCH_ROADRUNNER_PATH``.

On linux-aarch64 there is no libRoadRunner wheel or sdist, so the suite is
pointed at a from-source build with this variable (docs/BUILD_NOTES.md). The
logic mirrors the ``BENCH_GLIBC_ROOT`` tree lookup in ``test_fetch.py``: an env
var names an external artifact, and an absent or malformed one reads as "use the
pinned wheel" without error.
"""

from __future__ import annotations

import sys

import pytest

from bench import env


def _mk_build(root, *, dist_info=True):
    """A minimal source-build layout: the package dir, optionally the dist-info."""
    (root / "roadrunner").mkdir(parents=True)
    (root / "roadrunner" / "__init__.py").write_text("", encoding="utf-8")
    if dist_info:
        d = root / "libroadrunner-2.11.0.dist-info"
        d.mkdir()
        (d / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: libroadrunner\nVersion: 2.11.0\n", encoding="utf-8"
        )
    return root


def test_source_path_is_none_without_the_var(tmp_path, monkeypatch):
    monkeypatch.delenv("BENCH_ROADRUNNER_PATH", raising=False)
    assert env.roadrunner_source_path() is None


def test_source_path_is_none_when_the_var_names_no_package(tmp_path, monkeypatch):
    # A directory that exists but holds no ``roadrunner`` package is not a build;
    # treated as unset so a typo falls back to the pinned wheel, not a hard error.
    monkeypatch.setenv("BENCH_ROADRUNNER_PATH", str(tmp_path))
    assert env.roadrunner_source_path() is None


def test_source_path_resolves_a_real_build(tmp_path, monkeypatch):
    root = _mk_build(tmp_path / "site-packages")
    monkeypatch.setenv("BENCH_ROADRUNNER_PATH", str(root))
    assert env.roadrunner_source_path() == root


def test_ensure_on_path_prepends_once(tmp_path, monkeypatch):
    root = _mk_build(tmp_path / "site-packages")
    monkeypatch.setenv("BENCH_ROADRUNNER_PATH", str(root))
    monkeypatch.setattr(sys, "path", list(sys.path))  # isolate the edit
    env._ensure_roadrunner_on_path()
    env._ensure_roadrunner_on_path()  # idempotent
    assert sys.path.count(str(root)) == 1
    assert sys.path[0] == str(root)  # prepended, so the build wins


def test_ensure_on_path_is_a_noop_without_the_var(monkeypatch):
    monkeypatch.delenv("BENCH_ROADRUNNER_PATH", raising=False)
    before = list(sys.path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    env._ensure_roadrunner_on_path()
    assert sys.path == before


def test_identity_records_the_source_when_set(tmp_path, monkeypatch):
    # The version still comes from importlib.metadata (the real libroadrunner the
    # test env has, or absent -> error); the point here is that ``source`` is
    # recorded so a source-built result is never mistaken for a wheel one.
    root = _mk_build(tmp_path / "site-packages")
    monkeypatch.setenv("BENCH_ROADRUNNER_PATH", str(root))
    monkeypatch.setattr(sys, "path", list(sys.path))
    env.roadrunner_identity.cache_clear()
    try:
        ident = env.roadrunner_identity()
    finally:
        env.roadrunner_identity.cache_clear()
    assert ident["source"] == str(root)


def test_identity_omits_source_when_unset(monkeypatch):
    monkeypatch.delenv("BENCH_ROADRUNNER_PATH", raising=False)
    env.roadrunner_identity.cache_clear()
    try:
        ident = env.roadrunner_identity()
    finally:
        env.roadrunner_identity.cache_clear()
    assert "source" not in ident
