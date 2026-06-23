"""Tests for the artifact integrity primitives (⑦ 보안성 supply-chain)."""

import io
import os
import tarfile

import pytest

from core.security.artifact_integrity import (
    IntegrityError,
    load_verified_torch,
    read_sidecar,
    safe_extract_tar,
    sha256_file,
    sidecar_path,
    verify_artifact,
    write_sidecar,
)


def test_sha256_and_sidecar_roundtrip(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello-artifact")
    digest = write_sidecar(p)
    assert digest == sha256_file(p)
    assert read_sidecar(p) == digest
    assert os.path.exists(sidecar_path(p))


def test_verify_passes_with_correct_sidecar(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"data")
    write_sidecar(p)
    verify_artifact(p)  # no raise


def test_verify_fails_closed_on_tamper(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"data")
    write_sidecar(p)
    p.write_bytes(b"tampered")  # modified after sidecar written
    with pytest.raises(IntegrityError):
        verify_artifact(p)


def test_verify_allows_unverified_when_no_sidecar(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"data")
    verify_artifact(p)  # no sidecar, no expected → warn, no raise


def test_verify_explicit_expected_takes_precedence(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"data")
    with pytest.raises(IntegrityError):
        verify_artifact(p, expected_sha256="deadbeef")


def test_load_verified_torch_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    p = tmp_path / "m.pt"
    torch.save({"w": torch.tensor([1.0, 2.0])}, p)
    write_sidecar(p)
    loaded = load_verified_torch(p)
    assert loaded["w"].tolist() == [1.0, 2.0]


def test_load_verified_torch_rejects_tamper(tmp_path):
    torch = pytest.importorskip("torch")
    p = tmp_path / "m.pt"
    torch.save({"w": torch.tensor([1.0])}, p)
    write_sidecar(p)
    torch.save({"w": torch.tensor([9.0])}, p)  # re-save → digest changes
    with pytest.raises(IntegrityError):
        load_verified_torch(p)


def test_safe_extract_tar_rejects_traversal(tmp_path):
    tar_path = tmp_path / "evil.tar"
    with tarfile.open(tar_path, "w") as tf:
        data = b"x"
        info = tarfile.TarInfo(name="../escape.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    dest = tmp_path / "out"
    dest.mkdir()
    with tarfile.open(tar_path, "r") as tf:
        with pytest.raises(IntegrityError):
            safe_extract_tar(tf, dest)


def test_safe_extract_tar_allows_safe_members(tmp_path):
    tar_path = tmp_path / "good.tar"
    with tarfile.open(tar_path, "w") as tf:
        data = b"ok"
        info = tarfile.TarInfo(name="model/inner.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    dest = tmp_path / "out"
    dest.mkdir()
    with tarfile.open(tar_path, "r") as tf:
        safe_extract_tar(tf, dest)
    assert (dest / "model" / "inner.txt").read_bytes() == b"ok"
