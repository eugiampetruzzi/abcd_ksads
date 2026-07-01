"""Tests for the pydantic-based pipeline configuration."""

from pathlib import Path

import pytest

from abcd_ksads.config import Config

ENV_VARS = ["ABCD_70", "ABCD_51", "ABCD_RAW_PHENOTYPE", "ABCD_DEMOGRAPHICS"]


@pytest.fixture
def clean_env(monkeypatch):
    """Remove all ABCD env vars so defaults are exercised."""
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_default_abcd_70(clean_env):
    assert Config().ABCD_70 == Path("/path/to/abcd/release-7.0")


def test_default_abcd_51_core(clean_env):
    assert Config().ABCD_51_CORE == Path("/path/to/abcd/release-5.1/core")


def test_abcd_70_env_override(clean_env):
    clean_env.setenv("ABCD_70", "/data/abcd7")
    assert Config().ABCD_70 == Path("/data/abcd7")


def test_abcd_51_core_reads_abcd_51_alias(clean_env):
    clean_env.setenv("ABCD_51", "/data/abcd51/core")
    assert Config().ABCD_51_CORE == Path("/data/abcd51/core")


def test_raw_phenotype_derived_from_abcd_70(clean_env):
    clean_env.setenv("ABCD_70", "/data/abcd7")
    assert Config().RAW_PHENOTYPE == Path("/data/abcd7/rawdata/phenotype")


def test_demographics_derived_from_abcd_70(clean_env):
    clean_env.setenv("ABCD_70", "/data/abcd7")
    assert Config().DEMOGRAPHICS == Path("/data/abcd7/subject_demographics.tsv")


def test_raw_phenotype_explicit_env_wins(clean_env):
    clean_env.setenv("ABCD_70", "/data/abcd7")
    clean_env.setenv("ABCD_RAW_PHENOTYPE", "/elsewhere/pheno")
    assert Config().RAW_PHENOTYPE == Path("/elsewhere/pheno")


def test_demographics_explicit_env_wins(clean_env):
    clean_env.setenv("ABCD_70", "/data/abcd7")
    clean_env.setenv("ABCD_DEMOGRAPHICS", "/elsewhere/demo.tsv")
    assert Config().DEMOGRAPHICS == Path("/elsewhere/demo.tsv")


def test_path_fields_are_path_objects(clean_env):
    cfg = Config()
    for value in (cfg.ABCD_70, cfg.ABCD_51_CORE, cfg.RAW_PHENOTYPE, cfg.DEMOGRAPHICS):
        assert isinstance(value, Path)


def test_repo_points_to_repo_root(clean_env):
    repo = Config().REPO
    assert isinstance(repo, Path)
    assert (repo / "pyproject.toml").is_file()
    assert (repo / "src" / "abcd_ksads" / "config.py").is_file()


def test_repo_internal_paths_compose_from_repo(clean_env):
    cfg = Config()
    assert cfg.HARMONIZE == cfg.REPO / "harmonize"
    assert cfg.DERIV == cfg.ABCD_70 / "derivatives"
    assert cfg.FIGURES == cfg.REPO / "figures"
    assert cfg.TABLES == cfg.REPO / "tables"
    assert cfg.CODEBOOKS == cfg.REPO / "codebooks"
    assert cfg.DATASET == cfg.ABCD_70 / "abcd_ksads_harmonized"
    assert cfg.RAW_CACHE == cfg.DERIV / "raw_cache"


def test_repo_internal_paths_are_path_objects(clean_env):
    cfg = Config()
    for value in (cfg.REPO, cfg.HARMONIZE, cfg.DERIV, cfg.FIGURES, cfg.TABLES, cfg.CODEBOOKS, cfg.DATASET):
        assert isinstance(value, Path)


def test_module_singleton_and_reexports(clean_env):
    from abcd_ksads import config as config_module

    assert isinstance(config_module.config, Config)
    assert config_module.ABCD_70 == config_module.config.ABCD_70
    assert config_module.CODEBOOKS == config_module.config.CODEBOOKS
    assert config_module.DATASET == config_module.config.DATASET
