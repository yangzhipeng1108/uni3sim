#!/usr/bin/env python3
"""Capture version-bound 910B FlashAttention launch facts without duration.

The capture reads the exact serialized payload from the ACLNN executor and
keeps the exact ACLNN workspace size.  If ``aclrtLaunchKernelWithHostArgs`` is
visible, its buffer and placeholder map are retained as an independent
byte-for-byte cross-check.  The payload is decoded only after reflecting the
exact CANN tiling-data class registered for the runtime variant.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tarfile


BASELINE_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
RUNTIME_SYMBOL_PREFIX = (
    "FusedInferAttentionScore_3b093497fc536d61a77a7a3293a524da_")
RUNTIME_SYMBOL_SUFFIX = "_mix_aic"
EXPECTED_LAUNCHES = 21
LIVE_CAPTURE_SCHEMA_VERSION = 2
CANDIDATE_TILING_ADDRESS_OFFSET = 280
CANDIDATE_TILING_DATA_OFFSET = 296
MAXIMUM_OP_PARAMETER_BYTES = 3008
OUTPUT_OWNER_PARTS = ("profiler_output", "microbench", "ascend_910b")
INTERPOSER_SOURCE = (
    "microbench/ascend_910b/operators/flash_attn/"
    "tiling_capture_interposer.cpp")
TILING_REFLECTION_SOURCE = (
    "microbench/ascend_910b/operators/flash_attn/"
    "tiling_reflection_runner.cpp")
APPLICATION_SOURCE = "scripts/kernel_dataset/ascend_910b/flash_attn_runner.py"
SOURCE_FILES = (
    INTERPOSER_SOURCE,
    TILING_REFLECTION_SOURCE,
    "microbench/ascend_910b/operators/flash_attn/tiling_capture_runner.py",
    APPLICATION_SOURCE,
)
TILING_REGISTRATION_PREFIX = "FusedInferAttentionScore_"
TILING_ROOT_CLASS = "FusedInferAttentionScoreTilingData"
SPLIT_FUSE_TILING_ROOT_CLASS = "FAInferTilingData"
CANN_KERNEL_METADATA_RELATIVE = (
    "opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_transformer/"
    "fused_infer_attention_score/"
    "FusedInferAttentionScore_3b093497fc536d61a77a7a3293a524da.json")
RETAINED_CAPTURE_SOURCE_FILES = (
    INTERPOSER_SOURCE,
    "microbench/ascend_910b/operators/flash_attn/tiling_capture_runner.py",
    APPLICATION_SOURCE,
)
TILING_DTYPE_FORMATS = {
    "bool": "?",
    "int8_t": "b",
    "uint8_t": "B",
    "int16_t": "h",
    "uint16_t": "H",
    "int32_t": "i",
    "uint32_t": "I",
    "int64_t": "q",
    "uint64_t": "Q",
    "float": "f",
    "double": "d",
}
VARIANT_TO_METADATA_TILING_KEY = {
    "103000000010022221": "103000000010022221",
    "103000000010222221": "103000000010222221",
    "103000000010422221": "103000000010422221",
    "103000000010522221": "103000000010522221",
    "103000000010622221": "103000000010622221",
    "105000000000222220": "105000000000222220",
    "105000000000322220": "105000000000322220",
    "5000000000000200203": "5000000000000200203",
    "5000000000010200203": "5000000000010200203",
}
VARIANT_TO_TILING_ROOT_CLASS = {
    "103000000010022221": TILING_ROOT_CLASS,
    "103000000010222221": TILING_ROOT_CLASS,
    "103000000010422221": TILING_ROOT_CLASS,
    "103000000010522221": TILING_ROOT_CLASS,
    "103000000010622221": TILING_ROOT_CLASS,
    "105000000000222220": TILING_ROOT_CLASS,
    "105000000000322220": TILING_ROOT_CLASS,
    "5000000000000200203": SPLIT_FUSE_TILING_ROOT_CLASS,
    "5000000000010200203": SPLIT_FUSE_TILING_ROOT_CLASS,
}
CANN_STATIC_EVIDENCE = {
    "aarch64-linux/include/register/tilingdata_base.h": (
        "b2f7c0b08c95e4559b899bababf67cd4d3740780762c889450b089bfe09478a8"),
    "aarch64-linux/lib64/libregister.so": (
        "1a4df97416ed6983f4a33be4d3cb10802ce6bd715d5bed229eb9066841369360"),
    "aarch64-linux/lib64/libopapi_transformer.so": (
        "f38529da0c43aa1eba26e36bacd7f5e1637ac97a835e26b90556fd6b550a41fa"),
    "opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/aarch64/"
    "libophost_transformer.so": (
        "83fd235aff5d9ea4a7ee972ddf9e1e80bacfc42214df0169c350f85bd671030e"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "fused_infer_attention_score/arch32/"
    "fused_infer_attention_score_tilingkey.h": (
        "5bbe5bab7caa58f65d43a261c83ee9e26f282634299397b44d2d56216e5d4e3e"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "fused_infer_attention_score/fused_infer_attention_score_v3.cpp": (
        "9315d547e7468e28c5e38656ec01b8c9ef876bf53715cd166a545c7232305961"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "fused_infer_attention_score/fused_infer_attention_score.cpp": (
        "6ae6d2f747fa2fc30ad322ca8701ad193ef0ab3f6d64a46c41fb8f05c0ea17a3"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "fused_infer_attention_score/flash_attention_regular.h": (
        "59267576d00a9c40cffb402b1953c3a82956b43eeb570b7464c4ddfc17318393"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "common/fia_public_define.h": (
        "1339e51f2bb1b2e6002e9a1b1b342523b4bb4007c86242d83b681d605ba52fb4"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "common/arch32/fia_kernel_nonquant.h": (
        "7cde05730d68bb02a2c45cd6b25605417ee61d42dfd87f21ccf0948c014b7e21"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "common/arch32/fia_kernel_nonquant_mla.h": (
        "18f001420f2bb7fe7a0109e3d0e39787cc56857172754df171a009d577942516"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "common/arch32/fia_block_cube_nonquant.h": (
        "8556be2b11dc361b8f8bf1ca9667bc0abb3f5f2749b4e8077fb76591c2169766"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "common/arch32/fia_block_cube_nonquant_gqa.h": (
        "acce194dcf01f370d864ba7987f0c5a9d5927f7a69fb47426afa7789a6f856ef"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "common/arch32/fia_block_cube_nonquant_mla.h": (
        "c2d054e58a94f72c1a01f6f70bd77c21eba12939f01fabfe263d062a2088212f"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "common/arch32/fia_block_vec_nonquant.h": (
        "259f5700812eafe32bcffd075f59d90c82cb71458f62bd47cb3d6f17f171e762"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "common/arch32/fia_block_vec_nonquant_mla.h": (
        "62a072eaa46f9ed2bf6b0da21ed7bd6ce5eaa288f784aa76307fa640cae3986e"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "common/arch32/fia_block_vec_flashdecode.h": (
        "04ebb6eb9dd85da8cf75d57d466be0c6ec8980b4cf366294dca5cb7495faef45"),
    CANN_KERNEL_METADATA_RELATIVE: (
        "e57163518dc1619f52737d44f35e03ac2c1023c133ea6af9ad819edf0ec8f64d"),
}
RETAINED_CAPTURE_CANN_FILES = {
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "fused_infer_attention_score/arch32/"
    "fused_infer_attention_score_tilingkey.h": (
        "5bbe5bab7caa58f65d43a261c83ee9e26f282634299397b44d2d56216e5d4e3e"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "fused_infer_attention_score/fused_infer_attention_score_v3.cpp": (
        "9315d547e7468e28c5e38656ec01b8c9ef876bf53715cd166a545c7232305961"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "fused_infer_attention_score/fused_infer_attention_score.cpp": (
        "6ae6d2f747fa2fc30ad322ca8701ad193ef0ab3f6d64a46c41fb8f05c0ea17a3"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "common/fia_public_define.h": (
        "1339e51f2bb1b2e6002e9a1b1b342523b4bb4007c86242d83b681d605ba52fb4"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "common/arch32/fia_kernel_nonquant.h": (
        "7cde05730d68bb02a2c45cd6b25605417ee61d42dfd87f21ccf0948c014b7e21"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "common/arch32/fia_kernel_nonquant_mla.h": (
        "18f001420f2bb7fe7a0109e3d0e39787cc56857172754df171a009d577942516"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "common/arch32/fia_block_cube_nonquant.h": (
        "8556be2b11dc361b8f8bf1ca9667bc0abb3f5f2749b4e8077fb76591c2169766"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "common/arch32/fia_block_cube_nonquant_gqa.h": (
        "acce194dcf01f370d864ba7987f0c5a9d5927f7a69fb47426afa7789a6f856ef"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "common/arch32/fia_block_cube_nonquant_mla.h": (
        "c2d054e58a94f72c1a01f6f70bd77c21eba12939f01fabfe263d062a2088212f"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "common/arch32/fia_block_vec_nonquant.h": (
        "259f5700812eafe32bcffd075f59d90c82cb71458f62bd47cb3d6f17f171e762"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "common/arch32/fia_block_vec_nonquant_mla.h": (
        "62a072eaa46f9ed2bf6b0da21ed7bd6ce5eaa288f784aa76307fa640cae3986e"),
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "common/arch32/fia_block_vec_flashdecode.h": (
        "04ebb6eb9dd85da8cf75d57d466be0c6ec8980b4cf366294dca5cb7495faef45"),
    "opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_transformer/"
    "fused_infer_attention_score/"
    "FusedInferAttentionScore_3b093497fc536d61a77a7a3293a524da.json": (
        "e57163518dc1619f52737d44f35e03ac2c1023c133ea6af9ad819edf0ec8f64d"),
    "aarch64-linux/include/aclnn/acl_meta.h": (
        "be93460abfc7aa14113d7b9a3536fe2dd45d5148a357f55e15847655c9d5ffed"),
    "aarch64-linux/include/acl/acl_rt.h": (
        "8cce153ef9eb5ed49b0f5bfb5608de706e8d466c836ad2eb50745ec2e8629bc9"),
    "aarch64-linux/lib64/libopapi.so": (
        "b3890625d3e0cc11ef32460083698ecaa8f702b5edd0169ea617e2e6bb76270c"),
    "aarch64-linux/lib64/libnnopbase.so": (
        "6dc3b4fcfcf57b10be0018e89f4edbc079f02b4bb0aaadb109df350939fd4c99"),
    "aarch64-linux/lib64/libacl_rt.so": (
        "b6d7e6a70c4677b385c45f47aad25006a7ecbaf041a41838f26962b708a557ec"),
    "aarch64-linux/lib64/libacl_rt_impl.so": (
        "e58189fd107b153a4c518a1f6f7393834e0cf1e21ceb6e9c422db258bd26f2b0"),
}
RETAINED_CAPTURE_CANN_VERSION_SHA256 = (
    "49437edb9a235b28a87d07d2a2ad189022b6d246dc7c4cbcd013d2b7ecaf838b")
RETAINED_CAPTURE_CANN_ROOT = Path("/usr/local/Ascend/cann-8.5.1")
RETAINED_CAPTURE_CANN_PROVENANCE_GAPS = (
    "aarch64-linux/include/register/tilingdata_base.h",
    "aarch64-linux/lib64/libregister.so",
    "aarch64-linux/lib64/libopapi_transformer.so",
    "opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/aarch64/"
    "libophost_transformer.so",
    "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/"
    "fused_infer_attention_score/flash_attention_regular.h",
)
RETAINED_CAPTURE_COMPILER = Path("/usr/bin/aarch64-linux-gnu-g++-11")
RETAINED_CAPTURE_PYTHON = Path(
    "/usr/local/python3.11.14/bin/python3.11")
RETAINED_CAPTURE_INTERPOSER = "build/libmidlayer_fa_tiling_capture.so"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def _require_integer(value, context, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{context} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{context} must be non-negative")
    return value


def _parse_decimal_integer(text, context, *, positive=False,
                           nonnegative=False):
    if not isinstance(text, str) or re.fullmatch(r"[0-9]+", text) is None:
        raise ValueError(f"{context} must be an unsigned decimal integer")
    return _require_integer(
        int(text), context, positive=positive, nonnegative=nonnegative)


def parse_runtime_variant(symbol):
    if not isinstance(symbol, str):
        raise TypeError("runtime kernel symbol must be a string")
    if not symbol.startswith(RUNTIME_SYMBOL_PREFIX):
        raise ValueError(f"unexpected FlashAttention symbol prefix: {symbol!r}")
    if not symbol.endswith(RUNTIME_SYMBOL_SUFFIX):
        raise ValueError(f"unexpected FlashAttention symbol suffix: {symbol!r}")
    variant = symbol[
        len(RUNTIME_SYMBOL_PREFIX):-len(RUNTIME_SYMBOL_SUFFIX)]
    if variant not in VARIANT_TO_METADATA_TILING_KEY:
        raise ValueError(f"unsupported runtime symbol variant: {variant!r}")
    return variant


def _validate_contract_request_lengths(contract):
    batch = _require_integer(contract["batch"], "batch", positive=True)
    q_limit = _require_integer(
        contract["seq_len_q"], "seq_len_q", positive=True)
    kv_limit = _require_integer(
        contract["seq_len_cache"], "seq_len_cache", positive=True)
    q_lens = contract["q_lens"]
    kv_lens = contract["kv_lens"]
    if (not isinstance(q_lens, list) or not isinstance(kv_lens, list)
            or len(q_lens) != batch or len(kv_lens) != batch):
        raise ValueError(
            "FLASH_ATTN q_lens and kv_lens must match the batch size")
    for name, values, limit in (
            ("q_lens", q_lens, q_limit),
            ("kv_lens", kv_lens, kv_limit)):
        if any(isinstance(value, bool) or not isinstance(value, int)
               or value <= 0 or value > limit for value in values):
            raise ValueError(
                f"FLASH_ATTN {name} must be positive integers <= {limit}")
    if any(query > key_value for query, key_value in zip(q_lens, kv_lens)):
        raise ValueError("FLASH_ATTN request q_len cannot exceed kv_len")
    kv_layout = contract["kv_layout"]
    allowed_layouts = {
        "contiguous", "paged", "paged_varlen", "chunked_prefill", "mla"}
    if kv_layout not in allowed_layouts:
        raise ValueError(f"unsupported FLASH_ATTN kv_layout: {kv_layout!r}")
    if (kv_layout == "contiguous" and contract["causal"] is False
            and (any(value != q_limit for value in q_lens)
                 or any(value != kv_limit for value in kv_lens))):
        raise ValueError(
            "contiguous decode requires uniform Q and KV request lengths")
    if (kv_layout == "mla"
            and any(value != kv_limit for value in kv_lens)):
        raise ValueError("MLA requires uniform KV request lengths")
    return max(kv_lens)


def load_case_contract(dataset_path, expected_sha256, case_key):
    if SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError("canonical dataset SHA256 must be 64 lowercase hex")
    actual_sha256 = _sha256(dataset_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"canonical dataset SHA256 mismatch: expected={expected_sha256}, "
            f"actual={actual_sha256}")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict):
        raise TypeError("canonical FlashAttention dataset must be a mapping")
    entry = dataset[case_key]
    if not isinstance(entry, dict):
        raise TypeError(f"canonical case {case_key!r} must be a mapping")
    kernels = entry["kernels"]
    if not isinstance(kernels, list) or len(kernels) != 1:
        raise ValueError(
            f"canonical case {case_key!r} must contain one fused kernel")
    kernel = kernels[0]
    symbol = kernel["Metadata"]["Kernel Name"]
    variant = parse_runtime_variant(symbol)
    ascend = kernel["Ascend"]
    block_dim = _require_integer(
        ascend["block_dim"], "canonical block_dim", positive=True)
    mix_block_dim = _require_integer(
        ascend["mix_block_dim"], "canonical mix_block_dim", positive=True)
    if mix_block_dim != 2 * block_dim:
        raise ValueError(
            "canonical mixed FlashAttention launch must use AIC:AIV=1:2")
    attributes = entry["op_info"]["Attr"]
    expected_attribute_keys = {
        "B", "S", "Q_LEN", "Q_REQUESTED", "KV_LEN", "KV_REQUESTED",
        "HQ", "HKV", "D", "SP", "SP_REQUESTED", "Q_LENS", "KV_LENS",
        "split_policy", "kv_layout", "kv_dist", "causal", "layer",
    }
    if set(attributes) != expected_attribute_keys:
        raise ValueError(
            f"canonical case attributes changed: {sorted(attributes)}")
    q_lens = attributes["Q_LENS"]
    kv_lens = attributes["KV_LENS"]
    if not isinstance(q_lens, list) or not isinstance(kv_lens, list):
        raise TypeError("canonical request lengths must be lists")
    contract = {
        "case_key": case_key,
        "canonical_dataset_path": str(dataset_path.resolve()),
        "canonical_dataset_sha256": actual_sha256,
        "runtime_symbol": symbol,
        "runtime_variant": variant,
        "metadata_tiling_key": VARIANT_TO_METADATA_TILING_KEY[variant],
        "block_dim": block_dim,
        "mix_block_dim": mix_block_dim,
        "batch": _require_integer(attributes["B"], "B", positive=True),
        "seq_len_cache": _require_integer(
            attributes["KV_LEN"], "KV_LEN", positive=True),
        "num_heads_q": _require_integer(
            attributes["HQ"], "HQ", positive=True),
        "num_heads_kv": _require_integer(
            attributes["HKV"], "HKV", positive=True),
        "head_dim": _require_integer(attributes["D"], "D", positive=True),
        "seq_len_q": _require_integer(
            attributes["Q_LEN"], "Q_LEN", positive=True),
        "kv_layout": attributes["kv_layout"],
        "kv_dist": attributes["kv_dist"],
        "causal": attributes["causal"],
        "q_lens": q_lens,
        "kv_lens": kv_lens,
    }
    if not isinstance(contract["causal"], bool):
        raise TypeError("canonical causal attribute must be boolean")
    for field in ("kv_layout", "kv_dist"):
        if not isinstance(contract[field], str) or not contract[field]:
            raise TypeError(f"canonical {field} must be a non-empty string")
    _validate_contract_request_lengths(contract)
    return contract


def verify_deployed_source_archive(
        archive_path, repository_root, baseline_commit):
    if BASELINE_PATTERN.fullmatch(baseline_commit) is None:
        raise ValueError("baseline commit must be 40 lowercase hex")
    try:
        archive = tarfile.open(archive_path, "r:*")
    except tarfile.TarError as error:
        raise ValueError(
            f"deployed source archive is unreadable: {archive_path}") from error
    verified = []
    with archive:
        if "comment" not in archive.pax_headers:
            raise ValueError("deployed source is not a Git archive with commit PAX")
        archive_commit = archive.pax_headers["comment"]
        if archive_commit != baseline_commit:
            raise ValueError(
                f"archive commit {archive_commit!r} does not match "
                f"baseline {baseline_commit}")
        members = [member for member in archive.getmembers()
                   if member.isfile()]
        for relative in SOURCE_FILES:
            matches = [
                member for member in members
                if member.name == relative
                or member.name.endswith(f"/{relative}")
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"archive must contain exactly one {relative}, "
                    f"found {len(matches)}")
            stream = archive.extractfile(matches[0])
            if stream is None:
                raise ValueError(f"cannot read archived source {relative}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            archived_sha256 = digest.hexdigest()
            current_path = repository_root / relative
            if not current_path.is_file():
                raise FileNotFoundError(
                    f"deployed source file is missing: {current_path}")
            current_sha256 = _sha256(current_path)
            if current_sha256 != archived_sha256:
                raise ValueError(
                    f"deployed source differs from archive: {relative}")
            verified.append({
                "archive_member": matches[0].name,
                "path": str(current_path.resolve()),
                "sha256": current_sha256,
            })
    return {
        "archive_path": str(archive_path.resolve()),
        "archive_sha256": _sha256(archive_path),
        "git_archive_commit": archive_commit,
        "verified_sources": verified,
    }


def verify_cann_static_evidence(cann_root):
    identities = []
    for relative, expected_sha256 in CANN_STATIC_EVIDENCE.items():
        path = cann_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required CANN evidence is missing: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"CANN evidence SHA256 mismatch for {relative}: "
                f"expected={expected_sha256}, actual={actual_sha256}")
        identities.append({
            "path": str(path.resolve()),
            "sha256": actual_sha256,
        })
    _verify_cann_metadata_tiling_keys(
        cann_root / CANN_KERNEL_METADATA_RELATIVE)
    runtime_files = (
        "aarch64-linux/include/aclnn/acl_meta.h",
        "aarch64-linux/include/acl/acl_rt.h",
        "aarch64-linux/lib64/libopapi.so",
        "aarch64-linux/lib64/libnnopbase.so",
        "aarch64-linux/lib64/libacl_rt.so",
        "aarch64-linux/lib64/libacl_rt_impl.so",
    )
    for relative in runtime_files:
        path = cann_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required CANN runtime file is missing: {path}")
        identities.append({
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        })
    version_path = cann_root / "opp" / "version.info"
    version_text = version_path.read_text(encoding="utf-8")
    if "Version=8.5.1" not in version_text.splitlines():
        raise ValueError("tiling capture requires exact CANN Version=8.5.1")
    return {
        "cann_root": str(cann_root.resolve()),
        "version_path": str(version_path.resolve()),
        "version_sha256": _sha256(version_path),
        "files": identities,
    }


def _verify_cann_metadata_tiling_keys(metadata_path):
    """Bind exact, arbitrary-precision metadata keys to full kernel names."""

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("CANN kernel metadata root must be a mapping")
    kernel_list = payload["kernelList"]
    if not isinstance(kernel_list, list) or not kernel_list:
        raise ValueError("CANN kernel metadata requires a non-empty kernelList")
    for variant, expected_key in VARIANT_TO_METADATA_TILING_KEY.items():
        expected_name = RUNTIME_SYMBOL_PREFIX + variant
        matches = [
            entry for entry in kernel_list
            if isinstance(entry, dict)
            and entry["kernelName"] == expected_name
        ]
        if len(matches) != 1:
            raise ValueError(
                "CANN kernel metadata must contain one exact runtime entry: "
                f"name={expected_name!r}, count={len(matches)}")
        tiling_key = matches[0]["tilingKey"]
        if isinstance(tiling_key, bool) or not isinstance(tiling_key, int):
            raise TypeError(
                f"CANN metadata tilingKey must be an integer: {tiling_key!r}")
        if str(tiling_key) != expected_key:
            raise ValueError(
                "CANN metadata tilingKey differs from its exact decimal "
                f"identity: variant={variant}, expected={expected_key}, "
                f"actual={tiling_key}")


def parse_tiling_reflection(path, expected_registration_key):
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError("tiling reflection output is incomplete")
    if lines[0].split("\t") != ["SCHEMA", "1"]:
        raise ValueError("unsupported tiling reflection schema")
    root = lines[1].split("\t")
    if len(root) != 6 or root[0] != "ROOT":
        raise ValueError("tiling reflection root record is invalid")
    registration_key = root[1]
    if registration_key != expected_registration_key:
        raise ValueError(
            "tiling reflection registration key mismatch: "
            f"expected={expected_registration_key!r}, actual={registration_key!r}")
    class_name = root[2]
    data_size = _parse_decimal_integer(
        root[3], "reflected tiling data size", positive=True)
    struct_count = _parse_decimal_integer(
        root[4], "reflected struct count", positive=True)
    field_count = _parse_decimal_integer(
        root[5], "reflected field count", positive=True)
    if len(lines) != 2 + struct_count + field_count:
        raise ValueError("tiling reflection record count is inconsistent")

    structs = []
    for line in lines[2:2 + struct_count]:
        columns = line.split("\t")
        if len(columns) != 6 or columns[0] != "STRUCT":
            raise ValueError("tiling reflection struct record is invalid")
        structs.append({
            "path": columns[1],
            "class_name": columns[2],
            "offset_bytes": _parse_decimal_integer(
                columns[3], "reflected struct offset", nonnegative=True),
            "size_bytes": _parse_decimal_integer(
                columns[4], "reflected struct size", positive=True),
            "field_count": _parse_decimal_integer(
                columns[5], "reflected struct field count", positive=True),
        })
    fields = []
    for line in lines[2 + struct_count:]:
        columns = line.split("\t")
        if len(columns) != 6 or columns[0] != "FIELD":
            raise ValueError("tiling reflection field record is invalid")
        dtype = columns[2]
        if dtype not in TILING_DTYPE_FORMATS:
            raise ValueError(f"unsupported reflected tiling dtype: {dtype!r}")
        count = _parse_decimal_integer(
            columns[4], "reflected field count", positive=True)
        size = _parse_decimal_integer(
            columns[5], "reflected field size", positive=True)
        expected_size = struct.calcsize(
            f"<{count}{TILING_DTYPE_FORMATS[dtype]}")
        if size != expected_size:
            raise ValueError(
                f"reflected field size mismatch for {columns[1]!r}: "
                f"expected={expected_size}, actual={size}")
        fields.append({
            "path": columns[1],
            "dtype": dtype,
            "offset_bytes": _parse_decimal_integer(
                columns[3], "reflected field offset", nonnegative=True),
            "count": count,
            "size_bytes": size,
        })
    struct_paths = [entry["path"] for entry in structs]
    field_paths = [entry["path"] for entry in fields]
    if len(struct_paths) != len(set(struct_paths)):
        raise ValueError("reflected tiling struct paths are not unique")
    if len(field_paths) != len(set(field_paths)):
        raise ValueError("reflected tiling field paths are not unique")
    root_structs = [entry for entry in structs if entry["path"] == "$"]
    if len(root_structs) != 1:
        raise ValueError("tiling reflection must contain exactly one root struct")
    if (root_structs[0]["class_name"] != class_name
            or root_structs[0]["offset_bytes"] != 0
            or root_structs[0]["size_bytes"] != data_size):
        raise ValueError("tiling reflection root metadata is inconsistent")
    cursor = 0
    for field in sorted(fields, key=lambda entry: entry["offset_bytes"]):
        if field["offset_bytes"] != cursor:
            raise ValueError(
                "reflected leaf fields do not exactly cover the tiling struct")
        cursor += field["size_bytes"]
    if cursor != data_size:
        raise ValueError(
            "reflected leaf fields do not end at the tiling struct boundary")
    for entry in structs:
        if entry["offset_bytes"] + entry["size_bytes"] > data_size:
            raise ValueError("reflected nested struct exceeds the root struct")
    return {
        "schema_version": 1,
        "registration_key": registration_key,
        "class_name": class_name,
        "data_size_bytes": data_size,
        "structs": structs,
        "fields": fields,
    }


def decode_reflected_tiling(records, capture_summary, layout, contract):
    expected_class = VARIANT_TO_TILING_ROOT_CLASS[
        contract["runtime_variant"]]
    if layout["class_name"] != expected_class:
        raise ValueError(
            f"unexpected reflected tiling class: {layout['class_name']!r}")
    expected_registration_key = (
        TILING_REGISTRATION_PREFIX + contract["runtime_variant"])
    if layout["registration_key"] != expected_registration_key:
        raise ValueError("reflected tiling registration key changed")
    size = layout["data_size_bytes"]
    payload_source = capture_summary["tiling_payload_source"]
    serialized = []
    trailers = []
    if payload_source == "executor_getter":
        if capture_summary["executor_tiling_size_bytes"] != size:
            raise ValueError(
                "reflected tiling struct differs from executor payload size")
        executor_records = [
            record for record in records
            if record["record_type"] == "executor_tiling"]
        if len(executor_records) != EXPECTED_LAUNCHES:
            raise ValueError("executor tiling record set is incomplete")
        start = 0
        end = size
        for record in executor_records:
            payload = Path(record["tiling_path"]).read_bytes()
            if len(payload) != record["tiling_size"]:
                raise ValueError(
                    "captured executor tiling changed during decode")
            serialized.append(payload)
        host_args_visibility = capture_summary["host_args_visibility"]
        if host_args_visibility == "complete":
            host_start = capture_summary["candidate_tiling_data_offset_bytes"]
            host_end = capture_summary[
                "candidate_tiling_region_end_offset_bytes"]
            candidate_end = capture_summary[
                "candidate_tiling_region_size_bytes"]
            launch_records = [
                record for record in records
                if record["record_type"] == "launch_args"]
            if len(launch_records) != EXPECTED_LAUNCHES:
                raise ValueError("visible host-args record set is incomplete")
            for record in launch_records:
                args_bytes = Path(record["args_path"]).read_bytes()
                if len(args_bytes) != record["args_size"]:
                    raise ValueError(
                        "captured host arguments changed during tiling decode")
                trailers.append(args_bytes[host_start + size:host_end])
        elif host_args_visibility == "absent":
            candidate_end = size
            trailers = [b""] * len(serialized)
        else:
            raise ValueError(
                f"unknown host-args visibility: {host_args_visibility!r}")
    elif payload_source == "host_args_legacy":
        if not capture_summary["candidate_tiling_boundary_status"].startswith(
                "start_verified_"):
            raise ValueError(
                "cannot decode legacy tiling data without its live placeholder")
        start = capture_summary["candidate_tiling_data_offset_bytes"]
        end = start + size
        candidate_end = capture_summary[
            "candidate_tiling_region_end_offset_bytes"]
        if end > candidate_end:
            raise ValueError(
                "reflected tiling struct exceeds the live candidate region")
        launch_records = [
            record for record in records
            if record["record_type"] == "launch_args"]
        for record in launch_records:
            args_bytes = Path(record["args_path"]).read_bytes()
            if len(args_bytes) != record["args_size"]:
                raise ValueError(
                    "captured host arguments changed during tiling decode")
            serialized.append(args_bytes[start:end])
            trailers.append(args_bytes[end:candidate_end])
    else:
        raise ValueError(f"unknown tiling payload source: {payload_source!r}")
    serialized_hashes = sorted({
        hashlib.sha256(payload).hexdigest() for payload in serialized})
    if len(serialized_hashes) != 1:
        raise ValueError("serialized tiling data changed across identical calls")
    payload = serialized[0]
    decoded_fields = {}
    for field in layout["fields"]:
        offset = field["offset_bytes"]
        values = struct.unpack_from(
            f"<{field['count']}{TILING_DTYPE_FORMATS[field['dtype']]}",
            payload, offset)
        if field["dtype"] in {"float", "double"} and any(
                not math.isfinite(value) for value in values):
            raise ValueError(
                f"non-finite reflected tiling value in {field['path']!r}")
        decoded_fields[field["path"]] = (
            values[0] if field["count"] == 1 else list(values))

    approximate = {}
    if expected_class == TILING_ROOT_CLASS:
        required = {
            "baseParams.bSize": contract["batch"],
            "baseParams.n2Size": contract["num_heads_kv"],
            "baseParams.s1Size": contract["seq_len_q"],
            "baseParams.headDim": contract["head_dim"],
            "baseParams.usedCoreNum": contract["block_dim"],
        }
        if contract["num_heads_q"] % contract["num_heads_kv"] != 0:
            raise ValueError(
                "canonical attention heads must form an integer GQA group")
        required["baseParams.gSize"] = (
            contract["num_heads_q"] // contract["num_heads_kv"])
        maximum_kv_length = _validate_contract_request_lengths(contract)
        kv_layout = contract["kv_layout"]
        if kv_layout == "contiguous":
            required.update({
                "baseParams.s2Size": maximum_kv_length,
                "pageAttenParams.blockSize": 0,
                "pageAttenParams.maxBlockNumPerBatch": 0,
            })
        elif kv_layout in {
                "paged", "paged_varlen", "chunked_prefill", "mla"}:
            block_size_path = "pageAttenParams.blockSize"
            if block_size_path not in decoded_fields:
                raise ValueError(
                    "reflected paged tiling lacks pageAttenParams.blockSize")
            block_size = decoded_fields[block_size_path]
            if (isinstance(block_size, bool) or not isinstance(block_size, int)
                    or block_size <= 0):
                raise TypeError(
                    "reflected pageAttenParams.blockSize must be positive integer")
            # The executable runner fixes paged cache blocks at 128 tokens.
            # CANN records the widest physical block-table row in s2Size while
            # actualSeqLengths retains the real per-request lengths.
            if block_size != 128:
                raise ValueError(
                    "reflected paged block size contradicts the runner contract")
            maximum_blocks = (
                maximum_kv_length + block_size - 1) // block_size
            required.update({
                "baseParams.s2Size": maximum_blocks * block_size,
                block_size_path: block_size,
                "pageAttenParams.maxBlockNumPerBatch": maximum_blocks,
            })
        else:
            raise ValueError(
                f"unsupported FLASH_ATTN kv_layout: {kv_layout!r}")
    elif expected_class == SPLIT_FUSE_TILING_ROOT_CLASS:
        if not contract["causal"]:
            raise ValueError(
                "captured SplitFuse variants require a causal case contract")
        required = {
            "numHeads": contract["num_heads_q"],
            "embeddingSize": contract["head_dim"],
            "embeddingSizeV": contract["head_dim"],
            "kvHeads": contract["num_heads_kv"],
            "batch": contract["batch"],
            "maskType": 1,
        }
        approximate = {
            "scaleValue": 1.0 / math.sqrt(contract["head_dim"]),
        }
    else:
        raise AssertionError(f"unhandled tiling root class: {expected_class!r}")
    missing = sorted(set(required) - set(decoded_fields))
    missing += sorted(set(approximate) - set(decoded_fields))
    if missing:
        raise ValueError(f"reflected tiling fields are missing: {missing}")
    mismatches = {
        path: {"expected": expected, "actual": decoded_fields[path]}
        for path, expected in required.items()
        if decoded_fields[path] != expected
    }
    mismatches.update({
        path: {"expected": expected, "actual": decoded_fields[path]}
        for path, expected in approximate.items()
        if not math.isclose(
            decoded_fields[path], expected, rel_tol=1e-6, abs_tol=0.0)
    })
    if mismatches:
        raise ValueError(
            f"decoded tiling fields contradict the case contract: {mismatches}")

    trailer_hashes = sorted({
        hashlib.sha256(payload).hexdigest() for payload in trailers})
    trailer_variable_offsets = [
        offset for offset in range(candidate_end - end)
        if len({payload[offset] for payload in trailers}) > 1
    ]
    if payload_source == "executor_getter":
        decode_limitations = [
            "the exact metadata tilingKey is static evidence, not a live field",
            "decoded fields are version-bound discrete launch facts, not timing parameters",
        ]
        if capture_summary["host_args_visibility"] == "complete":
            decode_limitations.extend([
                "executor payload is cross-checked byte-for-byte against the host-args prefix",
                "the bytes after the executor payload have unresolved runtime semantics",
            ])
        elif capture_summary["host_args_visibility"] == "absent":
            decode_limitations.append(
                "the selected executor exposes no synchronous host-args launch")
        else:
            raise AssertionError(
                "executor payload has unknown host-args visibility")
    elif payload_source == "host_args_legacy":
        decode_limitations = [
            "the exact metadata tilingKey is static evidence, not a live host-args field",
            "the bytes after the reflected struct have unresolved runtime semantics",
            "decoded fields are version-bound discrete launch facts, not timing parameters",
        ]
    else:
        raise AssertionError(
            f"unhandled tiling payload source: {payload_source!r}")
    return {
        "schema_version": 1,
        "reflection": layout,
        "serialized_tiling": {
            "start_offset_bytes": start,
            "end_offset_bytes": end,
            "size_bytes": size,
            "sha256": serialized_hashes[0],
            "unique_sha256_count": len(serialized_hashes),
            "matches_observed_stable_prefix": (
                size == capture_summary[
                    "candidate_tiling_stable_prefix_size_bytes"]),
        },
        "pre_next_placeholder_trailer": {
            "start_offset_bytes": end,
            "end_offset_bytes": candidate_end,
            "size_bytes": candidate_end - end,
            "unique_sha256_count": len(trailer_hashes),
            "sha256s": trailer_hashes,
            "variable_byte_offsets": trailer_variable_offsets,
        },
        "case_contract_cross_checks": {**required, **approximate},
        "fields": decoded_fields,
        "limitations": decode_limitations,
    }


def _validate_retained_source_provenance(provenance, baseline_commit):
    required = {
        "archive_path", "archive_sha256", "git_archive_commit",
        "verified_sources",
    }
    if not isinstance(provenance, dict) or set(provenance) != required:
        raise ValueError("retained source provenance schema changed")
    if provenance["git_archive_commit"] != baseline_commit:
        raise ValueError("retained source archive commit mismatch")
    archive_path = Path(provenance["archive_path"])
    if not archive_path.is_absolute():
        raise ValueError("retained source archive path must be absolute")
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"retained source archive is missing: {archive_path}")
    if (_sha256(archive_path) != provenance["archive_sha256"]
            or SHA256_PATTERN.fullmatch(provenance["archive_sha256"]) is None):
        raise ValueError("retained source archive SHA256 mismatch")
    entries = provenance["verified_sources"]
    if not isinstance(entries, list) or len(entries) != len(
            RETAINED_CAPTURE_SOURCE_FILES):
        raise ValueError("retained source file list is incomplete")
    by_member = {}
    for entry in entries:
        if (not isinstance(entry, dict)
                or set(entry) != {"archive_member", "path", "sha256"}):
            raise ValueError("retained source file identity schema changed")
        member = entry["archive_member"]
        if member in by_member:
            raise ValueError("retained source archive member is duplicated")
        by_member[member] = entry
    if set(by_member) != set(RETAINED_CAPTURE_SOURCE_FILES):
        raise ValueError("retained source archive members changed")
    with tarfile.open(archive_path, "r:*") as archive:
        if ("comment" not in archive.pax_headers
                or archive.pax_headers["comment"] != baseline_commit):
            raise ValueError("retained Git archive PAX commit mismatch")
        members = {
            member.name: member for member in archive.getmembers()
            if member.isfile()
        }
        for relative in RETAINED_CAPTURE_SOURCE_FILES:
            if relative not in members:
                raise ValueError(
                    f"retained Git archive lacks source {relative}")
            entry = by_member[relative]
            deployed_path = Path(entry["path"])
            expected_deployed_path = archive_path.parent / "tree" / relative
            if (deployed_path != expected_deployed_path
                    or SHA256_PATTERN.fullmatch(entry["sha256"]) is None):
                raise ValueError(
                    f"retained source identity is invalid for {relative}")
            stream = archive.extractfile(members[relative])
            if stream is None:
                raise ValueError(
                    f"cannot read retained Git archive source {relative}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(
                    f"retained source hash mismatch for {relative}")
    return by_member


def _validate_retained_cann_provenance(provenance):
    required = {"cann_root", "version_path", "version_sha256", "files"}
    if not isinstance(provenance, dict) or set(provenance) != required:
        raise ValueError("retained CANN provenance schema changed")
    cann_root = Path(provenance["cann_root"])
    if cann_root != RETAINED_CAPTURE_CANN_ROOT:
        raise ValueError("retained CANN root changed")
    cann_root = cann_root.resolve()
    if not cann_root.is_dir():
        raise FileNotFoundError(f"retained CANN root is missing: {cann_root}")
    version_path = Path(provenance["version_path"]).resolve()
    if version_path != cann_root / "opp" / "version.info":
        raise ValueError("retained CANN version path is invalid")
    if (provenance["version_sha256"]
            != RETAINED_CAPTURE_CANN_VERSION_SHA256
            or not version_path.is_file()
            or _sha256(version_path) != RETAINED_CAPTURE_CANN_VERSION_SHA256):
        raise ValueError("retained CANN version identity mismatch")
    if "Version=8.5.1" not in version_path.read_text(
            encoding="utf-8").splitlines():
        raise ValueError("retained capture did not use CANN 8.5.1")
    files = provenance["files"]
    if not isinstance(files, list) or len(files) != len(
            RETAINED_CAPTURE_CANN_FILES):
        raise ValueError("retained CANN file list is incomplete")
    identities = {}
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError("retained CANN file identity schema changed")
        path = Path(entry["path"]).resolve()
        if cann_root not in path.parents:
            raise ValueError("retained CANN file is outside its root")
        relative = str(path.relative_to(cann_root))
        if relative in identities:
            raise ValueError("retained CANN file identity is duplicated")
        identities[relative] = entry
    if set(identities) != set(RETAINED_CAPTURE_CANN_FILES):
        raise ValueError("retained CANN file set changed")
    for relative, expected_sha256 in RETAINED_CAPTURE_CANN_FILES.items():
        entry = identities[relative]
        path = cann_root / relative
        if (entry["sha256"] != expected_sha256 or not path.is_file()
                or _sha256(path) != expected_sha256):
            raise ValueError(
                f"retained CANN identity mismatch for {relative}")
    return cann_root


def _validate_relocated_artifact(identity, capture_directory, relative, label,
                                 historical_capture_root=None):
    if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
        raise ValueError(f"retained {label} identity schema changed")
    historical_path = Path(identity["path"])
    if (not historical_path.is_absolute()
            or not historical_path.as_posix().endswith(f"/{relative}")):
        raise ValueError(f"retained {label} path is invalid")
    if (historical_capture_root is not None
            and historical_path != historical_capture_root / relative):
        raise ValueError(
            f"retained {label} path changed its historical capture root")
    path = capture_directory / relative
    if (not path.is_file()
            or SHA256_PATTERN.fullmatch(identity["sha256"]) is None
            or _sha256(path) != identity["sha256"]):
        raise ValueError(f"retained {label} SHA256 mismatch")
    return path


def _validate_npu_smi_idle(path, device_id, label):
    text = path.read_text(encoding="utf-8")
    if "npu-smi" not in text or "AICore(%)" not in text:
        raise ValueError(f"retained {label} lacks npu-smi AICore evidence")
    current_device = None
    target_rows = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        columns = [column.strip() for column in line.strip("|").split("|")]
        if len(columns) < 3:
            continue
        identity = columns[0].split()
        if (len(identity) >= 2 and identity[0].isdigit()
                and identity[1].startswith("910")):
            current_device = int(identity[0])
            if current_device == device_id and columns[1] != "OK":
                raise ValueError(f"retained {label} reports unhealthy device")
            continue
        if (current_device == device_id
                and re.fullmatch(r"[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:"
                                 r"[0-9A-Fa-f]{2}\.[0-9A-Fa-f]",
                                 columns[1]) is not None):
            values = columns[2].split()
            if not values or re.fullmatch(r"[0-9]+", values[0]) is None:
                raise ValueError(f"retained {label} has invalid AICore value")
            target_rows.append(int(values[0]))
    if target_rows != [0]:
        raise ValueError(
            f"retained {label} does not prove device {device_id} AICore idle")


def _semantic_capture_records(records, historical_capture_root):
    if not isinstance(records, list):
        raise TypeError("retained records.json must be a list")
    normalized = []
    path_fields = {
        "args_path", "placeholder_path", "tiling_path", "metadata_path"}
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("retained capture record must be a mapping")
        normalized_record = {}
        for key, value in record.items():
            if key in path_fields:
                path = Path(value)
                if (not path.is_absolute() or path.parent.name != "raw"
                        or path != historical_capture_root / "raw" / path.name):
                    raise ValueError(
                        f"retained record {key} changed its capture root")
                normalized_record[key] = path.name
            else:
                normalized_record[key] = value
        normalized.append(normalized_record)
    return normalized


def _validate_retained_compile_command(command, cann_root, source_identities,
                                        interposer_identity):
    if (not isinstance(command, list)
            or not all(isinstance(item, str) for item in command)):
        raise TypeError("retained compile command must be a string list")
    if len(command) != 21:
        raise ValueError("retained compile command length changed")
    compiler = Path(command[0])
    if compiler != RETAINED_CAPTURE_COMPILER:
        raise ValueError("retained compiler identity changed")
    output = Path(command[8])
    source = Path(command[9])
    if output != Path(interposer_identity["path"]):
        raise ValueError("retained compile output differs from interposer identity")
    if source != Path(source_identities[INTERPOSER_SOURCE]["path"]):
        raise ValueError("retained compile source differs from source provenance")
    expected = [
        command[0], "-std=c++17", "-O2", "-fPIC", "-shared",
        "-Wall", "-Wextra", "-o", command[8], command[9],
        "-I", str(cann_root / "aarch64-linux" / "include"),
        "-I", str(cann_root / "aarch64-linux" / "pkg_inc"),
        "-I", str(cann_root / "aarch64-linux" / "pkg_inc" / "profiling"),
        "-I", str(cann_root / "aarch64-linux" / "pkg_inc" / "toolchain"),
        "-I", str(cann_root / "aarch64-linux" / "pkg_inc" / "runtime"),
        "-ldl",
    ]
    if command != expected:
        raise ValueError("retained interposer compile command changed")


def _validate_retained_environment(environment, contract):
    required = {
        "ASCEND_DEVICE_ID", "ASCEND_RT_VISIBLE_DEVICES",
        "MIDLAYER_ASCEND_RUNTIME_VARIANT",
    }
    if not isinstance(environment, dict) or set(environment) != required:
        raise ValueError("retained capture environment schema changed")
    device_id = _parse_decimal_integer(
        environment["ASCEND_DEVICE_ID"], "retained device id",
        nonnegative=True)
    if environment["ASCEND_RT_VISIBLE_DEVICES"] != str(device_id):
        raise ValueError("retained visible-device binding changed")
    if (environment["MIDLAYER_ASCEND_RUNTIME_VARIANT"]
            != contract["runtime_variant"]):
        raise ValueError("retained runtime-variant binding changed")
    return device_id


def _validate_retained_application_command(
        command, capture_directory, historical_capture_root, contract,
        source_identities):
    if (not isinstance(command, list)
            or not all(isinstance(item, str) for item in command)):
        raise TypeError("retained application command must be a string list")
    if len(command) != 24:
        raise ValueError("retained application command length changed")
    python = Path(command[0])
    application = Path(command[1])
    historical_workload = Path(command[17])
    if python != RETAINED_CAPTURE_PYTHON:
        raise ValueError("retained Python identity changed")
    if application != Path(source_identities[APPLICATION_SOURCE]["path"]):
        raise ValueError(
            "retained application source differs from source provenance")
    if historical_workload != historical_capture_root / "workload.json":
        raise ValueError("retained workload changed its historical capture root")
    expected = [
        command[0], command[1],
        "--batch", str(contract["batch"]),
        "--seq_len_cache", str(contract["seq_len_cache"]),
        "--num_heads_q", str(contract["num_heads_q"]),
        "--num_heads_kv", str(contract["num_heads_kv"]),
        "--head_dim", str(contract["head_dim"]),
        "--dtype", "bf16",
        "--seq_len_q", str(contract["seq_len_q"]),
        "--workload-json", command[17],
        "--kv_layout", contract["kv_layout"],
        "--kv_dist", contract["kv_dist"],
        "--causal", "1" if contract["causal"] else "0",
    ]
    if command != expected:
        raise ValueError("retained application command changed")
    workload_path = capture_directory / "workload.json"
    if not workload_path.is_file():
        raise FileNotFoundError("retained capture lacks workload.json")
    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    if (not isinstance(workload, dict)
            or set(workload) != {"q_lens", "kv_lens"}
            or workload["q_lens"] != contract["q_lens"]
            or workload["kv_lens"] != contract["kv_lens"]):
        raise ValueError("retained workload differs from the case contract")


def _load_existing_schema_v1_capture(
        capture_directory, expected_manifest_sha256,
        expected_records_sha256, expected_baseline_commit, contract):
    if SHA256_PATTERN.fullmatch(expected_manifest_sha256) is None:
        raise ValueError("existing manifest SHA256 must be 64 lowercase hex")
    if SHA256_PATTERN.fullmatch(expected_records_sha256) is None:
        raise ValueError("existing records SHA256 must be 64 lowercase hex")
    if BASELINE_PATTERN.fullmatch(expected_baseline_commit) is None:
        raise ValueError("existing baseline commit must be 40 lowercase hex")
    capture_directory = capture_directory.resolve()
    if not capture_directory.is_dir():
        raise FileNotFoundError(
            f"existing capture directory is missing: {capture_directory}")
    manifest_path = capture_directory / "manifest.json"
    records_path = capture_directory / "records.json"
    raw_directory = capture_directory / "raw"
    if not manifest_path.is_file() or not records_path.is_file():
        raise FileNotFoundError("existing capture lacks manifest.json or records.json")
    if not raw_directory.is_dir():
        raise FileNotFoundError("existing capture lacks its raw directory")
    actual_manifest_sha256 = _sha256(manifest_path)
    actual_records_sha256 = _sha256(records_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "existing capture manifest SHA256 mismatch: "
            f"expected={expected_manifest_sha256}, "
            f"actual={actual_manifest_sha256}")
    if actual_records_sha256 != expected_records_sha256:
        raise ValueError(
            "existing capture records SHA256 mismatch: "
            f"expected={expected_records_sha256}, actual={actual_records_sha256}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_keys = {
        "application_command", "baseline_commit", "cann_provenance",
        "capture_state", "case_contract", "compile_command",
        "duration_consumed", "environment", "interposer",
        "post_telemetry", "pre_telemetry", "schema_version",
        "source_provenance", "summary",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_keys:
        raise ValueError("existing capture manifest schema changed")
    if manifest["schema_version"] != 1:
        raise ValueError("existing capture must use manifest schema 1")
    if manifest["capture_state"] != "complete_candidate":
        raise ValueError("existing capture is not a complete candidate")
    if manifest["duration_consumed"] is not False:
        raise ValueError("existing capture must not consume duration")
    if manifest["baseline_commit"] != expected_baseline_commit:
        raise ValueError("existing capture baseline commit mismatch")
    existing_contract = manifest["case_contract"]
    if not isinstance(existing_contract, dict) or set(existing_contract) != set(contract):
        raise ValueError("existing capture case-contract schema changed")
    for key, value in contract.items():
        if key == "canonical_dataset_path":
            continue
        if existing_contract[key] != value:
            raise ValueError(
                f"existing capture case contract differs at {key!r}")
    source_identities = _validate_retained_source_provenance(
        manifest["source_provenance"], expected_baseline_commit)
    cann_root = _validate_retained_cann_provenance(
        manifest["cann_provenance"])
    interposer_identity = manifest["interposer"]
    if (not isinstance(interposer_identity, dict)
            or set(interposer_identity) != {"path", "sha256"}):
        raise ValueError("retained interposer identity schema changed")
    historical_capture_root = Path(interposer_identity["path"]).parent.parent
    _validate_retained_compile_command(
        manifest["compile_command"], cann_root, source_identities,
        interposer_identity)
    device_id = _validate_retained_environment(
        manifest["environment"], contract)
    _validate_retained_application_command(
        manifest["application_command"], capture_directory,
        historical_capture_root, contract, source_identities)
    _validate_relocated_artifact(
        interposer_identity, capture_directory,
        RETAINED_CAPTURE_INTERPOSER, "interposer", historical_capture_root)
    pre_telemetry_path = _validate_relocated_artifact(
        manifest["pre_telemetry"], capture_directory,
        "logs/npu_smi_before.txt", "pre-capture telemetry",
        historical_capture_root)
    post_telemetry_path = _validate_relocated_artifact(
        manifest["post_telemetry"], capture_directory,
        "logs/npu_smi_after.txt", "post-capture telemetry",
        historical_capture_root)
    _validate_npu_smi_idle(pre_telemetry_path, device_id, "pre-capture telemetry")
    _validate_npu_smi_idle(
        post_telemetry_path, device_id, "post-capture telemetry")
    records = parse_capture_records(raw_directory)
    persisted_records = json.loads(records_path.read_text(encoding="utf-8"))
    if (_semantic_capture_records(
            persisted_records, historical_capture_root)
            != _semantic_capture_records(records, capture_directory)):
        raise ValueError("existing records.json differs from raw capture records")
    summary = summarize_capture_records(records, contract)
    if manifest["summary"] != summary:
        raise ValueError("existing manifest summary differs from raw capture")
    return {
        "capture_directory": str(capture_directory),
        "baseline_commit": expected_baseline_commit,
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_manifest_sha256,
        "records_path": str(records_path),
        "records_sha256": actual_records_sha256,
        "records": records,
        "summary": summary,
        "provenance_state": "capture_time_cann_incomplete",
        "provenance_gaps": list(RETAINED_CAPTURE_CANN_PROVENANCE_GAPS),
    }


def _validate_complete_source_provenance(provenance, baseline_commit):
    required = {
        "archive_path", "archive_sha256", "git_archive_commit",
        "verified_sources",
    }
    if not isinstance(provenance, dict) or set(provenance) != required:
        raise ValueError("complete source provenance schema changed")
    if provenance["git_archive_commit"] != baseline_commit:
        raise ValueError("complete source archive commit mismatch")
    archive_path = Path(provenance["archive_path"])
    if not archive_path.is_absolute() or not archive_path.is_file():
        raise FileNotFoundError("complete source archive path is invalid")
    if (SHA256_PATTERN.fullmatch(provenance["archive_sha256"]) is None
            or _sha256(archive_path) != provenance["archive_sha256"]):
        raise ValueError("complete source archive SHA256 mismatch")
    entries = provenance["verified_sources"]
    if not isinstance(entries, list) or len(entries) != len(SOURCE_FILES):
        raise ValueError("complete source file list is incomplete")
    by_member = {}
    deployed_root = None
    for entry in entries:
        if (not isinstance(entry, dict)
                or set(entry) != {"archive_member", "path", "sha256"}):
            raise ValueError("complete source identity schema changed")
        member = entry["archive_member"]
        if member in by_member:
            raise ValueError("complete source archive member is duplicated")
        if member not in SOURCE_FILES:
            raise ValueError(f"unexpected complete source member: {member!r}")
        deployed_path = Path(entry["path"])
        relative = Path(member)
        if (not deployed_path.is_absolute() or not deployed_path.is_file()
                or SHA256_PATTERN.fullmatch(entry["sha256"]) is None
                or _sha256(deployed_path) != entry["sha256"]):
            raise ValueError(
                f"complete deployed source identity is invalid for {member}")
        candidate_root = deployed_path.parents[len(relative.parts) - 1]
        if deployed_path != candidate_root / relative:
            raise ValueError(
                f"complete deployed source path is invalid for {member}")
        if deployed_root is None:
            deployed_root = candidate_root
        elif deployed_root != candidate_root:
            raise ValueError("complete deployed sources do not share one root")
        by_member[member] = entry
    if set(by_member) != set(SOURCE_FILES):
        raise ValueError("complete source archive members changed")
    with tarfile.open(archive_path, "r:*") as archive:
        if ("comment" not in archive.pax_headers
                or archive.pax_headers["comment"] != baseline_commit):
            raise ValueError("complete Git archive PAX commit mismatch")
        members = {
            member.name: member for member in archive.getmembers()
            if member.isfile()
        }
        for relative in SOURCE_FILES:
            if relative not in members:
                raise ValueError(
                    f"complete Git archive lacks source {relative}")
            stream = archive.extractfile(members[relative])
            if stream is None:
                raise ValueError(
                    f"cannot read complete Git archive source {relative}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != by_member[relative]["sha256"]:
                raise ValueError(
                    f"complete source archive hash mismatch for {relative}")
    return by_member


def _validate_capture_inventory(
        capture_directory, expected_inventory_sha256):
    if (not isinstance(expected_inventory_sha256, str)
            or SHA256_PATTERN.fullmatch(expected_inventory_sha256) is None):
        raise ValueError("existing artifact inventory SHA256 is invalid")
    inventory_path = capture_directory / "artifact.sha256"
    if not inventory_path.is_file():
        raise FileNotFoundError("complete capture lacks artifact.sha256")
    if _sha256(inventory_path) != expected_inventory_sha256:
        raise ValueError("complete capture artifact inventory SHA256 mismatch")
    entries = []
    for line_number, line in enumerate(
            inventory_path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\x00]+)", line)
        if match is None:
            raise ValueError(
                f"invalid artifact inventory line {line_number}")
        relative = Path(match.group(2))
        if (relative.is_absolute() or ".." in relative.parts
                or relative.as_posix() != match.group(2)
                or relative.name == "artifact.sha256"):
            raise ValueError("artifact inventory contains an unsafe path")
        entries.append((match.group(2), match.group(1)))
    paths = [relative for relative, _ in entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("artifact inventory paths must be sorted and unique")
    actual_paths = sorted(
        str(path.relative_to(capture_directory))
        for path in capture_directory.rglob("*")
        if path.is_file() and path != inventory_path)
    if paths != actual_paths:
        raise ValueError("artifact inventory does not exactly cover the capture")
    for relative, expected_sha256 in entries:
        if _sha256(capture_directory / relative) != expected_sha256:
            raise ValueError(
                f"artifact inventory SHA256 mismatch for {relative}")
    return {
        "path": str(inventory_path),
        "sha256": expected_inventory_sha256,
        "entry_count": len(entries),
    }


def _freeze_artifact_inventory(output_directory):
    inventory_path = output_directory / "artifact.sha256"
    if inventory_path.exists():
        raise FileExistsError(
            f"artifact inventory already exists: {inventory_path}")
    artifact_paths = sorted(
        path for path in output_directory.rglob("*") if path.is_file())
    if not artifact_paths:
        raise ValueError("cannot freeze an empty tiling artifact")
    lines = [
        f"{_sha256(path)}  {path.relative_to(output_directory).as_posix()}\n"
        for path in artifact_paths
    ]
    with inventory_path.open("x", encoding="utf-8") as stream:
        stream.writelines(lines)
    return {
        "path": str(inventory_path.resolve()),
        "sha256": _sha256(inventory_path),
        "entry_count": len(artifact_paths),
    }


def _index_cann_provenance(provenance, label):
    required = {"cann_root", "version_path", "version_sha256", "files"}
    if not isinstance(provenance, dict) or set(provenance) != required:
        raise ValueError(f"{label} CANN provenance schema changed")
    if not isinstance(provenance["cann_root"], str):
        raise TypeError(f"{label} CANN root must be a string")
    cann_root = Path(provenance["cann_root"])
    if not cann_root.is_absolute():
        raise ValueError(f"{label} CANN root must be absolute")
    cann_root = cann_root.resolve()
    if not isinstance(provenance["version_path"], str):
        raise TypeError(f"{label} CANN version path must be a string")
    version_path = Path(provenance["version_path"])
    if not version_path.is_absolute():
        raise ValueError(f"{label} CANN version path must be absolute")
    version_path = version_path.resolve()
    if version_path != cann_root / "opp" / "version.info":
        raise ValueError(f"{label} CANN version path is invalid")
    version_sha256 = provenance["version_sha256"]
    if (not isinstance(version_sha256, str)
            or SHA256_PATTERN.fullmatch(version_sha256) is None):
        raise ValueError(f"{label} CANN version SHA256 is invalid")
    files = provenance["files"]
    if not isinstance(files, list):
        raise TypeError(f"{label} CANN file identities must be a list")
    identities = {}
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError(f"{label} CANN file identity schema changed")
        if not isinstance(entry["path"], str):
            raise TypeError(f"{label} CANN file path must be a string")
        path = Path(entry["path"])
        if not path.is_absolute():
            raise ValueError(f"{label} CANN file path must be absolute")
        path = path.resolve()
        if cann_root not in path.parents:
            raise ValueError(f"{label} CANN file is outside its root")
        relative = path.relative_to(cann_root).as_posix()
        if relative in identities:
            raise ValueError(f"{label} CANN file identity is duplicated")
        sha256 = entry["sha256"]
        if (not isinstance(sha256, str)
                or SHA256_PATTERN.fullmatch(sha256) is None):
            raise ValueError(
                f"{label} CANN file SHA256 is invalid for {relative}")
        identities[relative] = sha256
    return cann_root, version_path, version_sha256, identities


def _validate_cann_provenance_superset(existing, current):
    (existing_root, existing_version_path, existing_version_sha256,
     existing_files) = _index_cann_provenance(
         existing, "complete capture")
    (current_root, current_version_path, current_version_sha256,
     current_files) = _index_cann_provenance(current, "current")
    if existing_root != current_root:
        raise ValueError("complete capture CANN root changed")
    if existing_version_path != current_version_path:
        raise ValueError("complete capture CANN version path changed")
    if existing_version_sha256 != current_version_sha256:
        raise ValueError("complete capture CANN version SHA256 changed")
    missing = sorted(set(existing_files) - set(current_files))
    if missing:
        raise ValueError(
            f"current CANN provenance lacks captured file {missing[0]}")
    for relative, sha256 in existing_files.items():
        if current_files[relative] != sha256:
            raise ValueError(
                f"complete capture CANN file SHA256 changed for {relative}")
    return current_root


def _load_existing_schema_v2_capture(
        capture_directory, expected_manifest_sha256,
        expected_records_sha256, expected_baseline_commit, contract,
        expected_inventory_sha256, current_cann_provenance):
    if current_cann_provenance is None:
        raise ValueError(
            "complete capture re-decode requires current CANN provenance")
    capture_directory = capture_directory.resolve()
    manifest_path = capture_directory / "manifest.json"
    records_path = capture_directory / "records.json"
    raw_directory = capture_directory / "raw"
    if not capture_directory.is_dir() or not raw_directory.is_dir():
        raise FileNotFoundError("complete capture directory is incomplete")
    if not manifest_path.is_file() or not records_path.is_file():
        raise FileNotFoundError(
            "complete capture lacks manifest.json or records.json")
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("complete capture manifest SHA256 mismatch")
    if _sha256(records_path) != expected_records_sha256:
        raise ValueError("complete capture records SHA256 mismatch")
    inventory = _validate_capture_inventory(
        capture_directory, expected_inventory_sha256)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_keys = {
        "application_command", "baseline_commit", "cann_provenance",
        "capture_state", "case_contract", "duration_consumed", "environment",
        "interposer", "interposer_compile_command", "post_telemetry",
        "pre_telemetry", "reflection_command", "reflection_compile_command",
        "reflection_environment", "schema_version", "source_provenance",
        "summary", "tiling_data", "tiling_reflection_runner",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_keys:
        raise ValueError("complete capture manifest schema changed")
    if manifest["schema_version"] != LIVE_CAPTURE_SCHEMA_VERSION:
        raise ValueError("complete capture must use manifest schema 2")
    if manifest["capture_state"] != "complete_candidate":
        raise ValueError("complete capture is not a complete candidate")
    if manifest["duration_consumed"] is not False:
        raise ValueError("complete capture consumed duration")
    if manifest["baseline_commit"] != expected_baseline_commit:
        raise ValueError("complete capture baseline commit mismatch")
    existing_contract = manifest["case_contract"]
    if (not isinstance(existing_contract, dict)
            or set(existing_contract) != set(contract)):
        raise ValueError("complete capture case-contract schema changed")
    for key, value in contract.items():
        if key != "canonical_dataset_path" and existing_contract[key] != value:
            raise ValueError(
                f"complete capture case contract differs at {key!r}")
    source_identities = _validate_complete_source_provenance(
        manifest["source_provenance"], expected_baseline_commit)
    cann_root = _validate_cann_provenance_superset(
        manifest["cann_provenance"], current_cann_provenance)
    interposer_identity = manifest["interposer"]
    historical_capture_root = Path(interposer_identity["path"]).parent.parent
    _validate_retained_compile_command(
        manifest["interposer_compile_command"],
        cann_root, source_identities,
        interposer_identity)
    environment = manifest["environment"]
    required_environment = {
        "ASCEND_DEVICE_ID", "ASCEND_RT_VISIBLE_DEVICES", "LD_LIBRARY_PATH",
        "MIDLAYER_ASCEND_RUNTIME_VARIANT",
    }
    if (not isinstance(environment, dict)
            or set(environment) != required_environment):
        raise ValueError("complete capture environment schema changed")
    device_id = _parse_decimal_integer(
        environment["ASCEND_DEVICE_ID"], "complete capture device id",
        nonnegative=True)
    if environment["ASCEND_RT_VISIBLE_DEVICES"] != str(device_id):
        raise ValueError("complete visible-device binding changed")
    if (environment["MIDLAYER_ASCEND_RUNTIME_VARIANT"]
            != contract["runtime_variant"]):
        raise ValueError("complete runtime-variant binding changed")
    expected_runtime_environment = _cann_live_environment(
        cann_root)
    if environment["LD_LIBRARY_PATH"] != expected_runtime_environment[
            "LD_LIBRARY_PATH"]:
        raise ValueError("complete runtime library path changed")
    _validate_retained_application_command(
        manifest["application_command"], capture_directory,
        historical_capture_root, contract, source_identities)
    _validate_relocated_artifact(
        interposer_identity, capture_directory,
        RETAINED_CAPTURE_INTERPOSER, "interposer", historical_capture_root)
    _validate_relocated_artifact(
        manifest["tiling_reflection_runner"], capture_directory,
        "build/tiling_reflection_runner", "tiling reflection runner",
        historical_capture_root)
    tiling_data_path = _validate_relocated_artifact(
        manifest["tiling_data"], capture_directory,
        "tiling_data.json", "tiling data", historical_capture_root)
    pre_telemetry_path = _validate_relocated_artifact(
        manifest["pre_telemetry"], capture_directory,
        "logs/npu_smi_before.txt", "pre-capture telemetry",
        historical_capture_root)
    post_telemetry_path = _validate_relocated_artifact(
        manifest["post_telemetry"], capture_directory,
        "logs/npu_smi_after.txt", "post-capture telemetry",
        historical_capture_root)
    _validate_npu_smi_idle(pre_telemetry_path, device_id, "pre-capture telemetry")
    _validate_npu_smi_idle(
        post_telemetry_path, device_id, "post-capture telemetry")
    records = parse_capture_records(raw_directory)
    persisted_records = json.loads(records_path.read_text(encoding="utf-8"))
    if (_semantic_capture_records(
            persisted_records, historical_capture_root)
            != _semantic_capture_records(records, capture_directory)):
        raise ValueError("complete records.json differs from raw capture records")
    input_tiling_data = json.loads(
        tiling_data_path.read_text(encoding="utf-8"))
    reflection = input_tiling_data["reflection"]
    summary = summarize_capture_records(
        records, contract, require_executor_tiling=True,
        expected_tiling_size=reflection["data_size_bytes"])
    input_summary = manifest["summary"]
    derived_summary_keys = {
        "tiling_reflection", "serialized_tiling_end_offset_bytes",
        "pre_next_placeholder_trailer_size_bytes",
    }
    if (not isinstance(input_summary, dict)
            or set(input_summary) != set(summary) | derived_summary_keys):
        raise ValueError("complete capture summary schema changed")
    for key, value in summary.items():
        if key != "limitations" and input_summary[key] != value:
            raise ValueError(
                f"complete capture raw summary differs at {key!r}")
    expected_reflection_summary = {
        "registration_key": reflection["registration_key"],
        "class_name": reflection["class_name"],
        "serialized_size_bytes": reflection["data_size_bytes"],
    }
    if input_summary["tiling_reflection"] != expected_reflection_summary:
        raise ValueError("complete capture reflection summary changed")
    if (input_summary["serialized_tiling_end_offset_bytes"]
            != input_tiling_data["serialized_tiling"]["end_offset_bytes"]
            or input_summary["pre_next_placeholder_trailer_size_bytes"]
            != input_tiling_data["pre_next_placeholder_trailer"]["size_bytes"]
            or input_summary["limitations"]
            != input_tiling_data["limitations"]):
        raise ValueError("complete capture derived summary changed")
    return {
        "capture_directory": str(capture_directory),
        "baseline_commit": expected_baseline_commit,
        "manifest_path": str(manifest_path),
        "manifest_sha256": expected_manifest_sha256,
        "records_path": str(records_path),
        "records_sha256": expected_records_sha256,
        "artifact_inventory": inventory,
        "records": records,
        "summary": summary,
        "provenance_state": "complete",
        "provenance_gaps": [],
    }


def load_existing_capture(
        capture_directory, expected_manifest_sha256,
        expected_records_sha256, expected_baseline_commit, contract,
        *, expected_inventory_sha256=None, current_cann_provenance=None):
    if SHA256_PATTERN.fullmatch(expected_manifest_sha256) is None:
        raise ValueError("existing manifest SHA256 must be 64 lowercase hex")
    if SHA256_PATTERN.fullmatch(expected_records_sha256) is None:
        raise ValueError("existing records SHA256 must be 64 lowercase hex")
    if BASELINE_PATTERN.fullmatch(expected_baseline_commit) is None:
        raise ValueError("existing baseline commit must be 40 lowercase hex")
    manifest_path = capture_directory.resolve() / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("existing capture lacks manifest.json")
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("existing capture manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = manifest["schema_version"]
    if schema_version == 1:
        if expected_inventory_sha256 is not None:
            raise ValueError(
                "retained schema-v1 capture must not claim a complete inventory")
        return _load_existing_schema_v1_capture(
            capture_directory, expected_manifest_sha256,
            expected_records_sha256, expected_baseline_commit, contract)
    if schema_version == LIVE_CAPTURE_SCHEMA_VERSION:
        if expected_inventory_sha256 is None:
            raise ValueError(
                "complete schema-v2 capture requires artifact inventory identity")
        return _load_existing_schema_v2_capture(
            capture_directory, expected_manifest_sha256,
            expected_records_sha256, expected_baseline_commit, contract,
            expected_inventory_sha256, current_cann_provenance)
    raise ValueError(
        f"unsupported existing capture manifest schema: {schema_version!r}")


def _parse_metadata(path):
    values = {}
    for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or "=" not in line:
            raise ValueError(f"invalid capture metadata at {path}:{line_number}")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise ValueError(
                f"duplicate or empty capture key at {path}:{line_number}")
        values[key] = value
    return values


def parse_capture_records(raw_directory):
    metadata_paths = sorted(raw_directory.glob("capture_*.meta"))
    if not metadata_paths:
        raise ValueError("capture interposer emitted no metadata records")
    records = []
    schema_versions = []
    for path in metadata_paths:
        values = _parse_metadata(path)
        common = {
            "schema_version", "record_type", "sequence", "api", "pid",
            "tid", "runtime_variant",
        }
        schema_version = _parse_decimal_integer(
            values["schema_version"], "capture schema version", positive=True)
        if schema_version not in {1, LIVE_CAPTURE_SCHEMA_VERSION}:
            raise ValueError(f"unsupported capture schema in {path}")
        schema_versions.append(schema_version)
        record_type = values["record_type"]
        if record_type == "workspace":
            expected = common | {"workspace_size"}
            if set(values) != expected:
                raise ValueError(
                    f"workspace capture fields changed in {path}: "
                    f"{sorted(values)}")
            record = {
                "record_type": record_type,
                "sequence": _parse_decimal_integer(
                    values["sequence"], "capture sequence", positive=True),
                "api": values["api"],
                "pid": _parse_decimal_integer(
                    values["pid"], "capture pid", positive=True),
                "tid": _parse_decimal_integer(
                    values["tid"], "capture tid", positive=True),
                "runtime_variant": values["runtime_variant"],
                "workspace_size": _parse_decimal_integer(
                    values["workspace_size"], "workspace size",
                    nonnegative=True),
                "metadata_path": str(path.resolve()),
            }
        elif record_type == "executor_tiling":
            if schema_version != LIVE_CAPTURE_SCHEMA_VERSION:
                raise ValueError(
                    "executor tiling records require capture schema version 2")
            expected = common | {"execute_api", "tiling_size", "tiling_file"}
            if set(values) != expected:
                raise ValueError(
                    f"executor tiling capture fields changed in {path}: "
                    f"{sorted(values)}")
            tiling_path = raw_directory / values["tiling_file"]
            if tiling_path.parent != raw_directory or not tiling_path.is_file():
                raise ValueError(
                    f"executor tiling file is invalid: {tiling_path}")
            tiling_bytes = tiling_path.read_bytes()
            tiling_size = _parse_decimal_integer(
                values["tiling_size"], "executor tiling size", positive=True)
            if tiling_size > MAXIMUM_OP_PARAMETER_BYTES:
                raise ValueError("executor tiling exceeds metadata opParaSize")
            if len(tiling_bytes) != tiling_size:
                raise ValueError(
                    f"executor tiling size mismatch for {tiling_path}: "
                    f"metadata={tiling_size}, file={len(tiling_bytes)}")
            record = {
                "record_type": record_type,
                "sequence": _parse_decimal_integer(
                    values["sequence"], "capture sequence", positive=True),
                "api": values["api"],
                "execute_api": values["execute_api"],
                "pid": _parse_decimal_integer(
                    values["pid"], "capture pid", positive=True),
                "tid": _parse_decimal_integer(
                    values["tid"], "capture tid", positive=True),
                "runtime_variant": values["runtime_variant"],
                "tiling_size": tiling_size,
                "tiling_sha256": hashlib.sha256(tiling_bytes).hexdigest(),
                "tiling_path": str(tiling_path.resolve()),
                "metadata_path": str(path.resolve()),
            }
        elif record_type == "launch_args":
            expected = common | {
                "execute_api", "metadata_tiling_key", "stub_symbol",
                "block_dim", "args_size", "candidate_tiling_address_offset",
                "candidate_tiling_data_offset",
                "candidate_tiling_pointer_slot_value", "args_file",
                "placeholder_count", "placeholder_entry_size",
                "placeholder_file",
            }
            if set(values) != expected:
                raise ValueError(
                    f"launch-argument capture fields changed in {path}: "
                    f"{sorted(values)}")
            args_path = raw_directory / values["args_file"]
            if args_path.parent != raw_directory or not args_path.is_file():
                raise ValueError(f"capture args file is invalid: {args_path}")
            args_bytes = args_path.read_bytes()
            args_size = _parse_decimal_integer(
                values["args_size"], "launch args size", positive=True)
            if len(args_bytes) != args_size:
                raise ValueError(
                    f"launch args size mismatch for {args_path}: "
                    f"metadata={args_size}, file={len(args_bytes)}")
            candidate_offset = _parse_decimal_integer(
                values["candidate_tiling_data_offset"],
                "candidate tiling data offset", positive=True)
            if candidate_offset >= args_size:
                raise ValueError("candidate tiling data offset exceeds args")
            candidate_address_offset = _parse_decimal_integer(
                values["candidate_tiling_address_offset"],
                "candidate tiling address offset", nonnegative=True)
            if candidate_address_offset + 8 > args_size:
                raise ValueError("candidate tiling pointer slot exceeds args")
            candidate_pointer_value = _parse_decimal_integer(
                values["candidate_tiling_pointer_slot_value"],
                "candidate tiling pointer slot value", nonnegative=True)
            if struct.unpack_from(
                    "<Q", args_bytes, candidate_address_offset)[0] \
                    != candidate_pointer_value:
                raise ValueError(
                    "candidate tiling pointer slot differs from raw args")
            placeholder_count = _parse_decimal_integer(
                values["placeholder_count"], "placeholder count",
                nonnegative=True)
            placeholder_entry_size = _parse_decimal_integer(
                values["placeholder_entry_size"], "placeholder entry size",
                positive=True)
            if placeholder_entry_size != struct.calcsize("<II"):
                raise ValueError("unexpected host-args placeholder entry size")
            placeholder_path = raw_directory / values["placeholder_file"]
            if (placeholder_path.parent != raw_directory
                    or not placeholder_path.is_file()):
                raise ValueError(
                    f"capture placeholder file is invalid: {placeholder_path}")
            placeholder_bytes = placeholder_path.read_bytes()
            if len(placeholder_bytes) != (
                    placeholder_count * placeholder_entry_size):
                raise ValueError(
                    f"placeholder size mismatch for {placeholder_path}")
            placeholders = [
                {"address_offset": address_offset,
                 "data_offset": data_offset}
                for address_offset, data_offset in struct.iter_unpack(
                    "<II", placeholder_bytes)
            ]
            if len({entry["address_offset"] for entry in placeholders}) \
                    != len(placeholders):
                raise ValueError("host-args placeholder offsets are not unique")
            if any(entry["address_offset"] + 8 > args_size
                   or entry["data_offset"] >= args_size
                   for entry in placeholders):
                raise ValueError("host-args placeholder exceeds raw args")
            record = {
                "record_type": record_type,
                "sequence": _parse_decimal_integer(
                    values["sequence"], "capture sequence", positive=True),
                "api": values["api"],
                "execute_api": values["execute_api"],
                "pid": _parse_decimal_integer(
                    values["pid"], "capture pid", positive=True),
                "tid": _parse_decimal_integer(
                    values["tid"], "capture tid", positive=True),
                "runtime_variant": values["runtime_variant"],
                "metadata_tiling_key": values["metadata_tiling_key"],
                "stub_symbol": values["stub_symbol"],
                "block_dim": _parse_decimal_integer(
                    values["block_dim"], "capture block_dim", positive=True),
                "args_size": args_size,
                "args_sha256": hashlib.sha256(args_bytes).hexdigest(),
                "candidate_tiling_address_offset": candidate_address_offset,
                "candidate_tiling_data_offset": candidate_offset,
                "candidate_tiling_pointer_slot_value": (
                    candidate_pointer_value),
                "placeholder_count": placeholder_count,
                "placeholder_entry_size": placeholder_entry_size,
                "placeholders": placeholders,
                "placeholder_sha256": hashlib.sha256(
                    placeholder_bytes).hexdigest(),
                "args_path": str(args_path.resolve()),
                "placeholder_path": str(placeholder_path.resolve()),
                "metadata_path": str(path.resolve()),
            }
        else:
            raise ValueError(f"unknown capture record type {record_type!r}")
        records.append(record)
    sequences = [record["sequence"] for record in records]
    if len(sequences) != len(set(sequences)):
        raise ValueError("capture record sequences are not unique")
    if len(set(schema_versions)) != 1:
        raise ValueError("capture directory mixes raw schema versions")
    return sorted(records, key=lambda record: record["sequence"])


def summarize_capture_records(
        records, contract, *, require_executor_tiling=False,
        expected_tiling_size=None):
    workspace_records = [
        record for record in records if record["record_type"] == "workspace"]
    executor_records = [
        record for record in records
        if record["record_type"] == "executor_tiling"]
    launch_records = [
        record for record in records if record["record_type"] == "launch_args"]
    if len(workspace_records) != EXPECTED_LAUNCHES:
        raise ValueError(
            f"expected {EXPECTED_LAUNCHES} workspace records, "
            f"found {len(workspace_records)}")
    if len(executor_records) not in {0, EXPECTED_LAUNCHES}:
        raise ValueError(
            "executor tiling records must be either absent for a retained "
            f"schema-v1 capture or complete: found {len(executor_records)}")
    if require_executor_tiling and len(executor_records) != EXPECTED_LAUNCHES:
        raise ValueError(
            f"live capture requires {EXPECTED_LAUNCHES} executor tiling "
            f"records, found {len(executor_records)}")
    if executor_records:
        if len(launch_records) not in {0, EXPECTED_LAUNCHES}:
            raise ValueError(
                "host-args observations must be completely absent or complete: "
                f"found {len(launch_records)}")
    elif len(launch_records) != EXPECTED_LAUNCHES:
        raise ValueError(
            f"retained schema-v1 capture requires {EXPECTED_LAUNCHES} target "
            f"launch records, found {len(launch_records)}")
    if expected_tiling_size is not None:
        if (isinstance(expected_tiling_size, bool)
                or not isinstance(expected_tiling_size, int)
                or not 0 < expected_tiling_size <= MAXIMUM_OP_PARAMETER_BYTES):
            raise ValueError("expected executor tiling size is invalid")
        if not executor_records:
            raise ValueError(
                "an expected executor tiling size requires executor records")

    expected_variant = contract["runtime_variant"]
    expected_block_dim = contract["block_dim"]
    if any(record["runtime_variant"] != expected_variant
           for record in records):
        raise ValueError("captured runtime-variant label changed within one run")
    execute_apis = {record["api"] for record in workspace_records}
    if len(execute_apis) != 1:
        raise ValueError("workspace execute API changed across identical calls")
    execute_api = next(iter(execute_apis))
    if any(record["block_dim"] != expected_block_dim
           for record in launch_records):
        raise ValueError("captured block_dim differs from canonical observation")
    if any(record["api"] != "aclrtLaunchKernelWithHostArgs"
           for record in launch_records):
        raise ValueError(
            "target executor used an unaudited host-args launch API")
    if any(record["metadata_tiling_key"] != "unavailable"
           for record in launch_records):
        raise ValueError("raw metadata tiling key must remain explicitly unavailable")
    if any(record["candidate_tiling_address_offset"]
           != CANDIDATE_TILING_ADDRESS_OFFSET for record in launch_records):
        raise ValueError("candidate tiling address offset changed")
    if any(record["candidate_tiling_data_offset"]
           != CANDIDATE_TILING_DATA_OFFSET for record in launch_records):
        raise ValueError("candidate tiling data offset changed")
    workspace_sizes = {
        record["workspace_size"] for record in workspace_records}
    if len(workspace_sizes) != 1:
        raise ValueError("workspace size changed across identical calls")

    if executor_records:
        has_host_args = bool(launch_records)
        group_size = 3 if has_host_args else 2
        for index in range(EXPECTED_LAUNCHES):
            group = records[group_size * index:group_size * (index + 1)]
            expected_types = (
                ["workspace", "executor_tiling", "launch_args"]
                if has_host_args else ["workspace", "executor_tiling"])
            if [record["record_type"] for record in group] != expected_types:
                raise ValueError(
                    "each execute must emit workspace, executor tiling and, "
                    "when visible, one immediately associated host-args record")
            workspace, executor = group[:2]
            if (workspace["pid"], workspace["tid"]) != (
                    executor["pid"], executor["tid"]):
                raise ValueError(
                    "workspace and executor tiling must share one process/thread")
            if (workspace["api"] != executor["execute_api"]
                    or executor["api"] != "NnopbaseGetTilingData"):
                raise ValueError(
                    "executor tiling getter and execute API identities do not match")
            if has_host_args:
                launch = group[2]
                if (workspace["pid"], workspace["tid"]) != (
                        launch["pid"], launch["tid"]):
                    raise ValueError(
                        "visible host-args launch must share the execute thread")
                if workspace["api"] != launch["execute_api"]:
                    raise ValueError(
                        "workspace and host-args execute API identities do not match")
    else:
        has_host_args = True
        for index in range(EXPECTED_LAUNCHES):
            workspace = records[2 * index]
            launch = records[2 * index + 1]
            if workspace["record_type"] != "workspace":
                raise ValueError("workspace record must precede each target launch")
            if launch["record_type"] != "launch_args":
                raise ValueError(
                    "each workspace record must pair with one target launch")
            if workspace["tid"] != launch["tid"]:
                raise ValueError("workspace and launch records must share one thread")
            if workspace["api"] != launch["execute_api"]:
                raise ValueError("workspace and launch API identities do not match")

    if executor_records:
        executor_sizes = {record["tiling_size"] for record in executor_records}
        if len(executor_sizes) != 1:
            raise ValueError("executor tiling size changed across identical calls")
        executor_size = next(iter(executor_sizes))
        if (expected_tiling_size is not None
                and executor_size != expected_tiling_size):
            raise ValueError(
                "executor tiling size differs from the reflected CANN struct: "
                f"executor={executor_size}, reflected={expected_tiling_size}")
        executor_hashes = sorted({
            record["tiling_sha256"] for record in executor_records})
        if len(executor_hashes) != 1:
            raise ValueError(
                "executor tiling payload changed across identical calls")
        tiling_payload_source = "executor_getter"
    else:
        executor_size = None
        executor_hashes = []
        tiling_payload_source = "host_args_legacy"

    if launch_records:
        placeholder_layouts = {
            tuple((entry["address_offset"], entry["data_offset"])
                  for entry in record["placeholders"])
            for record in launch_records
        }
        if len(placeholder_layouts) != 1:
            raise ValueError(
                "host-args placeholder layout changed across identical calls")
        placeholder_layout = next(iter(placeholder_layouts))
        expected_tiling_placeholder = (
            CANDIDATE_TILING_ADDRESS_OFFSET, CANDIDATE_TILING_DATA_OFFSET)
        has_expected_tiling_placeholder = (
            expected_tiling_placeholder in placeholder_layout)
        later_data_offsets = sorted({
            data_offset for _, data_offset in placeholder_layout
            if data_offset > CANDIDATE_TILING_DATA_OFFSET
        })
        args_sizes = {record["args_size"] for record in launch_records}
        if len(args_sizes) != 1:
            raise ValueError("host-args size changed across identical calls")
        args_size = next(iter(args_sizes))
        if later_data_offsets:
            candidate_region_end = later_data_offsets[0]
            candidate_region_end_basis = "next_host_input_placeholder"
        else:
            candidate_region_end = args_size
            candidate_region_end_basis = "args_size"
        if not (CANDIDATE_TILING_DATA_OFFSET
                < candidate_region_end <= args_size):
            raise ValueError("candidate tiling region boundary is invalid")
        candidate_region_size = (
            candidate_region_end - CANDIDATE_TILING_DATA_OFFSET)
        if candidate_region_size > MAXIMUM_OP_PARAMETER_BYTES:
            raise ValueError("candidate tiling region exceeds metadata opParaSize")
        candidate_regions = []
        for record in launch_records:
            args_bytes = Path(record["args_path"]).read_bytes()
            candidate_regions.append(args_bytes[
                CANDIDATE_TILING_DATA_OFFSET:candidate_region_end])
        candidate_region_hashes = sorted({
            hashlib.sha256(region).hexdigest()
            for region in candidate_regions
        })
        variable_offsets = [
            offset for offset in range(candidate_region_size)
            if len({region[offset] for region in candidate_regions}) > 1
        ]
        stable_prefix_size = (
            variable_offsets[0] if variable_offsets else candidate_region_size)
        if not has_expected_tiling_placeholder:
            boundary_status = "unverified"
        elif candidate_region_end_basis == "next_host_input_placeholder":
            boundary_status = "start_verified_end_bounded_by_next_placeholder"
        elif candidate_region_end_basis == "args_size":
            boundary_status = "start_verified_end_at_args_size_candidate"
        else:
            raise ValueError(
                f"unknown candidate-region end basis: {candidate_region_end_basis}")
        if executor_records:
            executor_payloads = [
                Path(record["tiling_path"]).read_bytes()
                for record in executor_records]
            if candidate_region_size < executor_size:
                raise ValueError(
                    "visible host-args tiling region is smaller than the "
                    "executor payload")
            if any(candidate[:executor_size] != executor
                   for candidate, executor in zip(
                    candidate_regions, executor_payloads)):
                raise ValueError(
                    "visible host-args tiling bytes differ from executor payload")
    else:
        placeholder_layout = ()
        candidate_region_end = None
        candidate_region_end_basis = None
        candidate_region_size = None
        candidate_region_hashes = []
        variable_offsets = []
        stable_prefix_size = executor_size
        boundary_status = "not_exposed_executor_payload_authoritative"

    if executor_records and launch_records:
        limitations = [
            "exact metadata tilingKey is static evidence, not a live field",
            "executor payload is cross-checked byte-for-byte against the host-args prefix",
            "tiling fields still require exact version-bound CANN struct decode",
        ]
    elif executor_records:
        limitations = [
            "exact metadata tilingKey is static evidence, not a live field",
            "the selected executor does not expose a synchronous host-args launch",
            "tiling fields still require exact version-bound CANN struct decode",
        ]
    else:
        limitations = [
            "retained schema-v1 capture predates NnopbaseGetTilingData capture",
            "exact metadata tilingKey is absent from the live host-args launch",
            "candidate region end is not a decoded tiling struct size",
            "stable prefix is evidence, not permission to mask its trailer",
            "tiling fields still require exact CANN struct decode",
            "opParaSize=3008 is a capacity, not an observed payload length",
        ]
    return {
        "record_count": len(records),
        "workspace_record_count": len(workspace_records),
        "executor_tiling_record_count": len(executor_records),
        "launch_record_count": len(launch_records),
        "execute_api": execute_api,
        "workspace_size_bytes": next(iter(workspace_sizes)),
        "runtime_symbol_variant": expected_variant,
        "metadata_tiling_key_from_static_metadata": (
            contract["metadata_tiling_key"]),
        "metadata_tiling_key_from_live_launch": None,
        "tiling_payload_source": tiling_payload_source,
        "executor_tiling_api": (
            "NnopbaseGetTilingData" if executor_records else None),
        "executor_tiling_size_bytes": executor_size,
        "executor_tiling_unique_sha256_count": len(executor_hashes),
        "executor_tiling_sha256s": executor_hashes,
        "host_args_visibility": "complete" if launch_records else "absent",
        "launch_api": launch_records[0]["api"] if launch_records else None,
        "candidate_tiling_boundary_status": boundary_status,
        "candidate_tiling_data_offset_bytes": (
            CANDIDATE_TILING_DATA_OFFSET if launch_records else None),
        "candidate_tiling_region_end_offset_bytes": candidate_region_end,
        "candidate_tiling_region_end_basis": candidate_region_end_basis,
        "candidate_tiling_region_size_bytes": candidate_region_size,
        "candidate_tiling_region_unique_sha256_count": len(
            candidate_region_hashes),
        "candidate_tiling_region_sha256s": candidate_region_hashes,
        "candidate_tiling_region_variable_byte_offsets": variable_offsets,
        "candidate_tiling_stable_prefix_size_bytes": stable_prefix_size,
        "host_args_placeholders": [
            {"address_offset": address_offset, "data_offset": data_offset}
            for address_offset, data_offset in placeholder_layout
        ],
        "stub_symbols": sorted({
            record["stub_symbol"] for record in launch_records}),
        "limitations": limitations,
    }


def _run_checked(command, log_path, *, environment=None, cwd=None,
                 timeout=None):
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT,
            env=environment, cwd=cwd, timeout=timeout, check=True)


def _library_environment(library_directories, context):
    missing = [str(path) for path in library_directories if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"required {context} library directories are missing: {missing}")
    environment = os.environ.copy()
    injected = sorted(
        key for key, value in environment.items()
        if key.startswith("LD_") and key != "LD_LIBRARY_PATH" and value)
    if injected:
        raise ValueError(
            f"{context} rejects inherited dynamic-loader controls: {injected}")
    environment["LD_LIBRARY_PATH"] = ":".join(
        str(path) for path in library_directories)
    return environment


def _validate_cann_library_alias(cann_root):
    alias = cann_root / "lib64"
    canonical = cann_root / "aarch64-linux" / "lib64"
    if not canonical.is_dir():
        raise FileNotFoundError(
            f"canonical CANN library directory is missing: {canonical}")
    if not alias.is_dir() or alias.resolve() != canonical.resolve():
        raise ValueError(
            "CANN lib64 search alias does not resolve to the hashed "
            "aarch64-linux/lib64 directory")
    return alias


def _cann_reflection_environment(cann_root):
    return _library_environment(
        (_validate_cann_library_alias(cann_root),), "CANN reflection")


def _cann_live_environment(cann_root):
    library_alias = _validate_cann_library_alias(cann_root)
    return _library_environment((
        library_alias,
        library_alias / "plugin" / "opskernel",
        library_alias / "plugin" / "nnengine",
        cann_root / "opp" / "built-in" / "op_impl" / "ai_core" / "tbe"
        / "op_tiling" / "lib" / "linux" / "aarch64",
        cann_root / "tools" / "aml" / "lib64",
        cann_root / "tools" / "aml" / "lib64" / "plugin",
        Path("/usr/local/Ascend/driver/lib64"),
        Path("/usr/local/Ascend/driver/lib64/common"),
        Path("/usr/local/Ascend/driver/lib64/driver"),
    ), "CANN live-runtime")


def _capture_npu_smi(executable, output_path, environment, device_id, label):
    _run_checked(
        [str(executable), "info"], output_path, environment=environment,
        timeout=60)
    text = output_path.read_text(encoding="utf-8")
    if "NPU" not in text or "AICore" not in text:
        raise ValueError("npu-smi output lacks NPU/AICore status tables")
    _validate_npu_smi_idle(output_path, device_id, label)
    return {
        "path": str(output_path.resolve()),
        "sha256": _sha256(output_path),
    }


def _resolve_executable(command, context):
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(
            f"required {context} executable is unavailable: {command!r}")
    path = Path(resolved).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"resolved {context} executable is not a file: {path}")
    return path


def _require_output_scope(output_dir, repository_root):
    owner_root = repository_root.joinpath(*OUTPUT_OWNER_PARTS).resolve()
    resolved = output_dir.resolve()
    if resolved.parent != owner_root:
        raise ValueError(
            f"output directory must be a direct child of {owner_root}")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    owner_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--deployed-source-archive", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--canonical-dataset", type=Path, required=True)
    parser.add_argument("--canonical-dataset-sha256", required=True)
    parser.add_argument("--case-key", required=True)
    parser.add_argument("--device-id", type=int)
    parser.add_argument("--existing-capture-dir", type=Path)
    parser.add_argument("--existing-capture-manifest-sha256")
    parser.add_argument("--existing-capture-records-sha256")
    parser.add_argument("--existing-capture-baseline-commit")
    parser.add_argument("--existing-capture-artifact-inventory-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cann-root", type=Path,
        default=Path("/usr/local/Ascend/cann-8.5.1"))
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--npu-smi", default="npu-smi")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args(argv)

    existing_arguments = (
        args.existing_capture_manifest_sha256,
        args.existing_capture_records_sha256,
        args.existing_capture_baseline_commit,
    )
    if args.existing_capture_dir is None:
        if args.device_id is None:
            raise ValueError("live capture requires --device-id")
        if any(value is not None for value in existing_arguments):
            raise ValueError(
                "existing-capture identity requires --existing-capture-dir")
        if args.existing_capture_artifact_inventory_sha256 is not None:
            raise ValueError(
                "artifact inventory identity requires --existing-capture-dir")
    else:
        if args.device_id is not None:
            raise ValueError("offline decode must not select an NPU device")
        if any(value is None for value in existing_arguments):
            raise ValueError(
                "offline decode requires manifest, records and baseline identity")
    if args.device_id is not None and args.device_id < 0:
        raise ValueError("device id must be non-negative")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    repository_root = args.repository_root.resolve()
    if not repository_root.is_dir():
        raise FileNotFoundError(f"repository root is missing: {repository_root}")
    _require_output_scope(args.output_dir, repository_root)
    output_dir = args.output_dir.resolve()
    logs_dir = output_dir / "logs"
    raw_dir = output_dir / "raw"
    build_dir = output_dir / "build"
    logs_dir.mkdir()
    raw_dir.mkdir()
    build_dir.mkdir()

    source_provenance = verify_deployed_source_archive(
        args.deployed_source_archive.resolve(), repository_root,
        args.baseline_commit)
    contract = load_case_contract(
        args.canonical_dataset.resolve(), args.canonical_dataset_sha256,
        args.case_key)
    cann_provenance = verify_cann_static_evidence(args.cann_root.resolve())
    _write_json(output_dir / "case_contract.json", contract)
    reflection_environment = _cann_reflection_environment(
        args.cann_root.resolve())

    compiler = _resolve_executable(args.compiler, "compiler")
    reflection_source = repository_root / TILING_REFLECTION_SOURCE
    reflection_executable = build_dir / "tiling_reflection_runner"
    register_library_directory = args.cann_root / "aarch64-linux" / "lib64"
    reflection_compile_command = [
        str(compiler), "-std=c++17", "-O2", "-Wall", "-Wextra",
        "-o", str(reflection_executable), str(reflection_source),
        "-I", str(args.cann_root / "aarch64-linux" / "include"),
        "-L", str(register_library_directory),
        f"-Wl,-rpath,{register_library_directory}",
        "-lregister", "-ldl",
    ]
    _run_checked(
        reflection_compile_command, logs_dir / "reflection_compile.log",
        timeout=300)
    if not reflection_executable.is_file():
        raise RuntimeError("compiler did not produce the tiling reflection runner")
    registration_key = (
        TILING_REGISTRATION_PREFIX + contract["runtime_variant"])
    op_host_library = (
        args.cann_root / "opp" / "built-in" / "op_impl" / "ai_core"
        / "tbe" / "op_host" / "lib" / "linux" / "aarch64"
        / "libophost_transformer.so")
    reflection_command = [
        str(reflection_executable), str(op_host_library), registration_key]
    reflection_output_path = build_dir / "tiling_reflection.tsv"
    _run_checked(
        reflection_command, reflection_output_path,
        environment=reflection_environment, timeout=300)
    layout = parse_tiling_reflection(
        reflection_output_path, registration_key)
    _write_json(output_dir / "tiling_layout.json", layout)

    if args.existing_capture_dir is not None:
        existing_capture = load_existing_capture(
            args.existing_capture_dir,
            args.existing_capture_manifest_sha256,
            args.existing_capture_records_sha256,
            args.existing_capture_baseline_commit,
            contract,
            expected_inventory_sha256=(
                args.existing_capture_artifact_inventory_sha256),
            current_cann_provenance=cann_provenance)
        summary = existing_capture["summary"]
        tiling_data = decode_reflected_tiling(
            existing_capture["records"], summary, layout, contract)
        provenance_state = existing_capture["provenance_state"]
        if provenance_state == "capture_time_cann_incomplete":
            provenance_limitation = (
                "retained capture-time CANN provenance lacks the host-tiling and "
                "registration identities listed in input_capture.provenance_gaps; "
                "decoded fields are diagnostic and cannot enter production launch "
                "selection")
            tiling_data["limitations"].append(provenance_limitation)
            artifact_type = "existing_capture_tiling_decode"
            decode_state = "provenance_incomplete_candidate"
            input_artifact_inventory = None
        elif provenance_state == "complete":
            artifact_type = "existing_complete_capture_tiling_redecode"
            decode_state = "complete_candidate"
            input_artifact_inventory = existing_capture["artifact_inventory"]
        else:
            raise ValueError(
                f"unknown existing capture provenance state: {provenance_state!r}")
        summary["tiling_reflection"] = {
            "registration_key": layout["registration_key"],
            "class_name": layout["class_name"],
            "serialized_size_bytes": layout["data_size_bytes"],
        }
        summary["serialized_tiling_end_offset_bytes"] = (
            tiling_data["serialized_tiling"]["end_offset_bytes"])
        summary["pre_next_placeholder_trailer_size_bytes"] = (
            tiling_data["pre_next_placeholder_trailer"]["size_bytes"])
        summary["limitations"] = tiling_data["limitations"]
        tiling_data_path = output_dir / "tiling_data.json"
        _write_json(tiling_data_path, tiling_data)
        offline_manifest = {
            "schema_version": 2,
            "artifact_type": artifact_type,
            "decode_state": decode_state,
            "duration_consumed": False,
            "baseline_commit": args.baseline_commit,
            "case_contract": contract,
            "source_provenance": source_provenance,
            "cann_provenance": cann_provenance,
            "input_capture": {
                "capture_directory": existing_capture["capture_directory"],
                "baseline_commit": existing_capture["baseline_commit"],
                "manifest_path": existing_capture["manifest_path"],
                "manifest_sha256": existing_capture["manifest_sha256"],
                "records_path": existing_capture["records_path"],
                "records_sha256": existing_capture["records_sha256"],
                "artifact_inventory": input_artifact_inventory,
                "provenance_state": existing_capture["provenance_state"],
                "provenance_gaps": existing_capture["provenance_gaps"],
            },
            "reflection_compile_command": reflection_compile_command,
            "reflection_command": reflection_command,
            "reflection_environment": {
                "LD_LIBRARY_PATH": reflection_environment["LD_LIBRARY_PATH"],
            },
            "tiling_reflection_runner": {
                "path": str(reflection_executable),
                "sha256": _sha256(reflection_executable),
            },
            "tiling_reflection_output": {
                "path": str(reflection_output_path),
                "sha256": _sha256(reflection_output_path),
            },
            "tiling_data": {
                "path": str(tiling_data_path),
                "sha256": _sha256(tiling_data_path),
            },
            "summary": summary,
        }
        _write_json(output_dir / "manifest.json", offline_manifest)
        _freeze_artifact_inventory(output_dir)
        print(json.dumps(summary, sort_keys=True))
        return

    python = _resolve_executable(args.python, "python")
    npu_smi = _resolve_executable(args.npu_smi, "npu-smi")
    runtime_environment = _cann_live_environment(args.cann_root.resolve())
    interposer_source = repository_root / INTERPOSER_SOURCE
    interposer_library = build_dir / "libmidlayer_fa_tiling_capture.so"
    interposer_compile_command = [
        str(compiler), "-std=c++17", "-O2", "-fPIC", "-shared",
        "-Wall", "-Wextra", "-o", str(interposer_library),
        str(interposer_source),
        "-I", str(args.cann_root / "aarch64-linux" / "include"),
        "-I", str(args.cann_root / "aarch64-linux" / "pkg_inc"),
        "-I", str(args.cann_root / "aarch64-linux" / "pkg_inc" / "profiling"),
        "-I", str(args.cann_root / "aarch64-linux" / "pkg_inc" / "toolchain"),
        "-I", str(args.cann_root / "aarch64-linux" / "pkg_inc" / "runtime"),
        "-ldl",
    ]
    _run_checked(
        interposer_compile_command, logs_dir / "interposer_compile.log",
        timeout=300)
    if not interposer_library.is_file():
        raise RuntimeError("compiler did not produce the capture interposer")
    pre_telemetry = _capture_npu_smi(
        npu_smi, logs_dir / "npu_smi_before.txt", runtime_environment,
        args.device_id, "pre-capture telemetry")
    workload_path = output_dir / "workload.json"
    _write_json(workload_path, {
        "q_lens": contract["q_lens"],
        "kv_lens": contract["kv_lens"],
    })
    application = repository_root / APPLICATION_SOURCE
    application_command = [
        str(python), str(application),
        "--batch", str(contract["batch"]),
        "--seq_len_cache", str(contract["seq_len_cache"]),
        "--num_heads_q", str(contract["num_heads_q"]),
        "--num_heads_kv", str(contract["num_heads_kv"]),
        "--head_dim", str(contract["head_dim"]),
        "--dtype", "bf16",
        "--seq_len_q", str(contract["seq_len_q"]),
        "--workload-json", str(workload_path),
        "--kv_layout", contract["kv_layout"],
        "--kv_dist", contract["kv_dist"],
        "--causal", "1" if contract["causal"] else "0",
    ]
    environment = runtime_environment.copy()
    if "LD_PRELOAD" in environment and environment["LD_PRELOAD"]:
        raise ValueError("tiling capture requires an initially empty LD_PRELOAD")
    environment["LD_PRELOAD"] = str(interposer_library)
    environment["MIDLAYER_ASCEND_TILING_CAPTURE_DIR"] = str(raw_dir)
    environment["MIDLAYER_ASCEND_RUNTIME_VARIANT"] = contract["runtime_variant"]
    environment["ASCEND_DEVICE_ID"] = str(args.device_id)
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    _run_checked(
        application_command, logs_dir / "application.log",
        environment=environment, cwd=repository_root,
        timeout=args.timeout_seconds)
    post_telemetry = _capture_npu_smi(
        npu_smi, logs_dir / "npu_smi_after.txt", runtime_environment,
        args.device_id, "post-capture telemetry")

    records = parse_capture_records(raw_dir)
    summary = summarize_capture_records(
        records, contract, require_executor_tiling=True,
        expected_tiling_size=layout["data_size_bytes"])
    tiling_data = decode_reflected_tiling(
        records, summary, layout, contract)
    summary["tiling_reflection"] = {
        "registration_key": layout["registration_key"],
        "class_name": layout["class_name"],
        "serialized_size_bytes": layout["data_size_bytes"],
    }
    summary["serialized_tiling_end_offset_bytes"] = (
        tiling_data["serialized_tiling"]["end_offset_bytes"])
    summary["pre_next_placeholder_trailer_size_bytes"] = (
        tiling_data["pre_next_placeholder_trailer"]["size_bytes"])
    summary["limitations"] = tiling_data["limitations"]
    _write_json(output_dir / "records.json", records)
    tiling_data_path = output_dir / "tiling_data.json"
    _write_json(tiling_data_path, tiling_data)
    manifest = {
        "schema_version": 2,
        "capture_state": "complete_candidate",
        "duration_consumed": False,
        "baseline_commit": args.baseline_commit,
        "case_contract": contract,
        "source_provenance": source_provenance,
        "cann_provenance": cann_provenance,
        "interposer_compile_command": interposer_compile_command,
        "reflection_compile_command": reflection_compile_command,
        "reflection_command": reflection_command,
        "reflection_environment": {
            "LD_LIBRARY_PATH": reflection_environment["LD_LIBRARY_PATH"],
        },
        "application_command": application_command,
        "environment": {
            "ASCEND_DEVICE_ID": environment["ASCEND_DEVICE_ID"],
            "ASCEND_RT_VISIBLE_DEVICES": environment[
                "ASCEND_RT_VISIBLE_DEVICES"],
            "MIDLAYER_ASCEND_RUNTIME_VARIANT": environment[
                "MIDLAYER_ASCEND_RUNTIME_VARIANT"],
            "LD_LIBRARY_PATH": environment["LD_LIBRARY_PATH"],
        },
        "interposer": {
            "path": str(interposer_library),
            "sha256": _sha256(interposer_library),
        },
        "tiling_reflection_runner": {
            "path": str(reflection_executable),
            "sha256": _sha256(reflection_executable),
        },
        "tiling_data": {
            "path": str(tiling_data_path),
            "sha256": _sha256(tiling_data_path),
        },
        "pre_telemetry": pre_telemetry,
        "post_telemetry": post_telemetry,
        "summary": summary,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _freeze_artifact_inventory(output_dir)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
