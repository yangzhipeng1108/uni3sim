#!/usr/bin/env python3
"""Capture the CANN FIA P-copy strided ND2NZ mechanism."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


CANONICAL_GEOMETRIES = (
    # Compact controls plus the source-exact continuous-BSH K/V tiles used
    # by the CANN FIA standard-GQA D64/D128 paths.
    (4, 64, 64),
    (256, 64, 64),
    (256, 64, 512),
    (256, 128, 128),
    (256, 128, 1024),
    (256, 128, 4096),
    # The source P copy is short-row and can carry the score-row stride.
    (4, 256, 256),
    (4, 256, 512),
    (4, 256, 1024),
)
DETERMINISTIC_SPLIT_K_GEOMETRIES = (
    # Source-exact CopyIn geometry for the CANN small-M deterministic split-K
    # program, plus contiguous and intermediate-stride mechanism controls.
    (128, 384, 384),
    (128, 384, 4096),
    (128, 384, 28672),
    (128, 256, 256),
    (128, 256, 4096),
    (128, 256, 28672),
)
PAGED_GQA_128_GEOMETRIES = (
    # Independent controls around the source-exact paged-GQA K/V descriptor.
    # The contiguous and wider-stride cells are frozen validation peers; only
    # the 1024-element row stride maps to the D128/HKV8 source program.
    (128, 128, 128),
    (128, 128, 1024),
    (128, 128, 4096),
)
PAGED_GQA_TAIL_RESOLUTION_GEOMETRIES = (
    # v2 Arm 1: locate the instrument's minimum resolvable row count at the
    # D128/HKV8 paged K/V geometry.  A refusal here is a valid result -- it
    # bounds the domain -- because a decode tail is (context + 1) mod 128 and
    # is therefore approximately uniform over 1..127 across a workload, so the
    # measurable fraction has to be known before the tail widths below can be
    # claimed to cover production.  Geometric ladder spanning the refused
    # 1-row point and the measured tails; fixed before any data is opened.
    (4, 128, 1024),
    (8, 128, 1024),
    (16, 128, 1024),
    (32, 128, 1024),
)
# The binary rejects a batch of more than 128 cells, and 124 rows x 2 issue
# modes is 248, so the exhaustive domain is issued as two runs whose union is
# exactly rows 4..127.  Splitting the execution does not change the cell set,
# the criteria or the tolerances.
PAGED_GQA_TAIL_EXHAUSTIVE_LO_GEOMETRIES = tuple(
    (rows, 128, 1024) for rows in range(4, 66)
)
PAGED_GQA_TAIL_EXHAUSTIVE_HI_GEOMETRIES = tuple(
    (rows, 128, 1024) for rows in range(66, 128)
)
PAGED_GQA_TAIL_EXHAUSTIVE_GEOMETRIES = tuple(
    # v3 main arm: every row count the production tail domain contains.  v1
    # established that 1 row is refused by the instrument and v2's Arm 1 that
    # the resolvable bound lies in (1, 4], so the reachable domain is exactly
    # rows 4..127 -- 124 values.  Measuring all of them removes interpolation
    # and any fitted functional form: a non-smooth service is unproblematic
    # when every point in the domain is measured.  Exhaustive mechanism
    # measurement, not a fit; no parameter is estimated from the data.
    (rows, 128, 1024)
    for rows in range(4, 128)
)
PAGED_GQA_TAIL_GROUP_TRANSFER_GEOMETRIES = (
    # v3 group-independence arm: production decode grids land on 22 and 23 as
    # well as 24, so the main arm's group-24 service must be shown to transfer.
    # Where it does not, those rows publish for group 24 only and the other
    # groups are reported unsupported rather than scaled.
    (4, 128, 1024),
    (61, 128, 1024),
    (127, 128, 1024),
)
PAGED_GQA_TAIL_GEOMETRIES = (
    # Same D128/HKV8 page geometry as PAGED_GQA_128, for the short final page
    # of a paged K/V panel.  A decode step appends one token, so a request's
    # KV extent is (context + 1) and its last block-table page carries
    # (context + 1) mod 128 rows; the full-page cells above never describe it.
    # The row set is defined by that arithmetic -- the widest tail plus the
    # interior widths production shows -- not by any observed duration.
    # A 1-row tail (256 source bytes) is rejected by this instrument's own
    # validation ("differential interval did not increase") before any
    # duration is produced: the short/long differential cannot resolve a
    # descriptor that small.  That is an instrument domain limit, not a
    # result, so the 1-row cell is not carried here; tails below the
    # resolvable bound stay unsupported and must be reported as such.
    (61, 128, 1024),
    (81, 128, 1024),
    (127, 128, 1024),
    # Frozen validation peers at the same tail width, mirroring the
    # contiguous and wider-stride controls used for the full page.
    (61, 128, 128),
    (61, 128, 4096),
)
FIA_SCORE_P_SHORT_STRIDE_GEOMETRIES = (
    # CANN 8.5.1 regular-GQA CopyPToL1 for a four-row MM2 probability
    # panel.  ``srcDValue`` is the source score-row width: 512 and 1024
    # occur in varlen source tasks, while a full 2048-row task retains 2048.
    # The 22-AIC topology is the exact matched-control launch domain.
    (4, 256, 512),
    (4, 256, 1024),
    (4, 256, 2048),
)
FIA_SCORE_P_REGULAR_GQA_GEOMETRIES = tuple(
    # Complete rectangular mechanism sweep for the standard regular-GQA
    # CopyPToL1 source rows and score widths.  It is deliberately defined by
    # public descriptor geometry, not by an operator name or latency result.
    (rows, 256, source_row_stride)
    for rows in (4, 5, 8)
    for source_row_stride in (512, 1024, 2048, 4096, 8192)
)
GEMM_N_MAJOR_B_COLD_GEOMETRIES = (
    # Source-exact MatMulV3 B CopyIn descriptor for the M16 small-M family.
    # Each worker walks distinct N-major 64x256 tiles with the full N stride.
    (64, 256, 28672),
)
FUSED_MOE_K2880_B_COLD_CELLS = (
    # Exact BL1 descriptor emitted by the GroupedMatmul M128/N256/K64
    # program when the transposed BF16 weight matrix has physical K stride.
    (64, 256, 2880, 24),
)
FUSED_MOE_K6144_B_COLD_CELLS = (
    # M=1 GroupedMatmul disables B L2 residency.  Measure only its 64x256
    # BF16 BL1 descriptor with physical K-major stride and 24 AICs; the
    # repeated stream is non-production and includes no A, Cube, routing,
    # output, or complete MoE work.
    (64, 256, 6144, 24),
)
GEMM_LEGACY_K14336_COLD_GEOMETRIES = (
    # Exact A1/B1 descriptor shapes emitted by the 22-AIC MatMulV2 K14336,
    # N4096 family. Both operands retain the physical K-major source stride.
    (16, 4096, 14336),
    (96, 640, 14336),
)
GEMM_V2_K14336_EXACT_COLD_CELLS = (
    # Exact AL1/BL1 descriptors decoded from the CANN 8.5.1 MatMulV2
    # K14336/N4096 small-M launch programs.  The cells retain the physical
    # K-major source stride and isolate descriptor service from Cube/output.
    (16, 512, 14336, 22),
    (16, 256, 14336, 22),
    (32, 256, 14336, 22),
    (48, 256, 14336, 22),
    (64, 256, 14336, 22),
    (96, 256, 14336, 22),
)
GEMM_V3_K1536_N5120_COLD_CELLS = (
    # Exact BL1 descriptors emitted by the MatMulV3 K1536/N5120 family.
    # M16 uses N224 on 23 AICs; wider M uses N256 on 20 or 24 AICs.
    (224, 256, 5120, 23),
    (256, 256, 5120, 20),
    (256, 256, 5120, 24),
)
GEMM_SMALL_M_REMAINING_COLD_CELLS = (
    # Exact A1/B1 descriptors for the remaining K8192/N8192 V3,
    # K5120/N5120 V2, K5120/N1536 V3, and K4096/N4096 V2 families.
    (16, 4096, 8192, 16),
    (256, 256, 8192, 16),
    (128, 512, 5120, 23),
    (112, 512, 5120, 23),
    (16, 4096, 5120, 24),
    (64, 1024, 1536, 24),
    (128, 512, 4096, 22),
    (96, 640, 4096, 22),
)
GEMM_V2_K5120_N5120_COLD_CELLS = (
    # CANN 8.5.1 MatMulV2 dry-launch payloads encode a full-load A1 and a
    # 256-element B1 K slab for M16, while M32/M48/M64 encode 256-element
    # K slabs for both ping-pong operands.  Measure the individual source
    # descriptors rather than combining two slots into a synthetic copy.
    (16, 5120, 5120, 23),
    (224, 256, 5120, 23),
    (32, 256, 5120, 23),
    (48, 256, 5120, 23),
    (64, 256, 5120, 23),
    (112, 256, 5120, 23),
)
GEMM_V2_K5120_N1536_COLD_CELLS = (
    # Exact AL1/BL1 descriptors decoded from the CANN 8.5.1 key-98513
    # launch payload for M1/K5120/N1536.  Both operands retain their source
    # row stride while each of the 24 AICs owns one N64 output tile.
    (16, 512, 5120, 24),
    (64, 512, 1536, 24),
)
GEMM_V2_KEY98499_COLD_CELLS = (
    # Exact AL1/BL1 descriptors decoded from CANN 8.5.1 key-98499 dry
    # launches.  B4/M1/K4096/N4096 uses a full-K A1 plus one N176/K256 B1
    # slab on 24 AICs.  B8/M1/K1536/N5120 uses a full-K A1 on 23 AICs; its
    # N224/K256 B1 descriptor is retained here as the paired validation cell.
    (16, 4096, 4096, 24),
    (176, 256, 4096, 24),
    (16, 1536, 1536, 23),
    (224, 256, 5120, 23),
)
GEMM_K5120_N13824_SOURCE_ORDER_COLD_CELLS = (
    # Exact AL1/BL1 descriptors emitted by the normalized CANN 8.5.1 source
    # programs for the M1/K5120/N13824 family.  A keeps its physical K stride;
    # transposed B is N-major and therefore keeps the physical N stride.  The
    # worker-wide N extent is issued as physical N256 B1 descriptors rather
    # than one synthetic large-N descriptor.
    (16, 512, 5120, 18),
    (64, 512, 5120, 18),
    (256, 256, 13824, 18),
    (128, 256, 5120, 22),
    (256, 256, 13824, 22),
    (128, 256, 5120, 24),
    (128, 512, 5120, 24),
    (256, 256, 13824, 24),
)
GEMM_P90_BOUNDARY_COLD_CELLS = (
    # Exact AL1/BL1 descriptors from the unresolved MatMulV2 small-M
    # programs at the 910B GEMM P90 boundary and the K8192/N8192 tail.
    # The cells retain the source row stride and non-empty AIC count while
    # measuring descriptor service independently from the production op.
    (16, 4096, 4096, 23),
    (256, 256, 14336, 23),
    (32, 2048, 8192, 22),
    (64, 1024, 8192, 22),
    (256, 256, 8192, 22),
    (32, 2048, 11008, 16),
    (256, 256, 4096, 16),
    (48, 1344, 8192, 22),
    (16, 4096, 11008, 22),
    (176, 320, 4096, 22),
    (64, 1024, 1536, 20),
    (256, 256, 5120, 20),
)
GEMM_MAX_APE_K4096_N11008_COLD_CELLS = (
    # Exact AL1/BL1 descriptors decoded from the CANN 8.5.1 MatMulV3
    # B64/M1/K4096/N11008 source program.  All 22 AICs issue one A1 K1024
    # slab and one B1 N256/K256 slab with their physical source strides.
    (64, 1024, 4096, 22),
    (256, 256, 11008, 22),
)
GEMM_MAX_APE_K13824_N5120_COLD_CELLS = (
    # Exact AL1/BL1 descriptors decoded from the CANN 8.5.1 MatMulV2
    # M512/K13824/N5120 source program.  The 4x6 worker grid issues four
    # output tiles per AIC with an eight-base-K A1 slab and a four-base-K B1
    # slab.  Measure the two source descriptors independently from GEMM.
    (128, 512, 13824, 24),
    (224, 256, 13824, 24),
)
GEMM_DECODE_V2_COLD_CELLS = (
    # Exact AL1/BL1 descriptors from the CANN 8.5.1 MatMulV2 decode launches
    # of the qwen3-4b linear ops (M16 persistent full-A, split_k=1), as
    # selected by the production analytical launch selector:
    #
    # * qkv/gate_up/lm_head: base tile 16x256x64, depth_b1=2, b_l1_k_tiles=4
    #   -> B1 slab = K256 x N256; one base-K MTE2 descriptor = K64 x N256.
    # * o/down: base tile 16x112x128, depth_b1=2, b_l1_k_tiles=2
    #   -> B1 slab = K256 x N112; one base-K MTE2 descriptor = K128 x N112.
    # * A1 is the full-K M16 contiguous slab (stride == own row length).
    #
    # Every B descriptor is N-major: rows are the K extent, columns the N
    # extent, and source_row_stride is the physical K of the weight matrix.
    # Active AIC counts are the launch blockDim (qkv/lm_head 24, gate_up 19,
    # o/down 23).  The decode serving regime streams every weight byte from
    # DRAM each step, so only the cold stream behavior of these exact
    # descriptors prices the serving path.  Both the B1-slab granularity and
    # the base-K descriptor granularity are measured: the Forward service
    # consumes whichever matches its accounting unit.
    #
    # The down-proj K9728 A1 stream cannot be issued as one MTE2 descriptor:
    # a 16x9728 fp16 NZ tile is 304 KiB, past the 256-KiB per-descriptor L1
    # budget enforced by the benchmark kernel, so the hardware-visible A1
    # unit is a contiguous half-K chunk (16x4864, 152 KiB).  The measured
    # per-chunk service composes to the full-K A1 stream.
    (16, 2560, 2560, 24),
    (256, 256, 2560, 24),
    (64, 256, 2560, 24),
    (16, 2560, 2560, 19),
    (256, 256, 2560, 19),
    (64, 256, 2560, 19),
    (16, 4096, 4096, 23),
    (256, 112, 4096, 23),
    (128, 112, 4096, 23),
    (16, 4864, 4864, 23),
    (256, 112, 9728, 23),
    (128, 112, 9728, 23),
)
GEMM_DECODE_B1_SLAB_NATURAL_ORIENTATION_CELLS = (
    # The qwen3-4b decode o/down B1 slab expressed in the orientation the
    # canonical GEMM launch actually stores and reads it.
    #
    # launch_data/ascend_910b/gemm_launch_selection.csv.zst for (K,N,M) =
    # (4096,2560,40) and (9728,2560,40) gives b_layout=row_major,
    # trans_b=true, single_core_n=112, b_l1_k_tiles=8, tile_k=64.  B is
    # therefore stored [N, K] row-major and one B1 slab is N112 x K512:
    # 112 rows, 512 contiguous elements each, row stride = physical K.
    #
    # The already-measured cold_exact cells for the same slab are written
    # (512, 112, K) -- rows and columns transposed -- which reads 112
    # contiguous elements from each of 512 rows instead.  Both spell the
    # same byte count; only the contiguity differs.  These two cells close
    # the natural orientation at the launch blockDim (23) that the existing
    # (112, 512, 4096, 22/24) resident cells do not cover.
    #
    # Geometry is taken from the launch LUT only.  No duration, error,
    # MAPE or makespan selects any cell.
    (112, 512, 4096, 23),
    (112, 512, 9728, 23),
)
GEMM_V2_K13824_N5120_RESIDENT_CELLS = (
    # Exact A1/B1 descriptors decoded from the narrow-M key-99025 source
    # program.  A 128-MiB cyclic source keeps the descriptor lines inside
    # the published 192-MiB L2 while preserving their physical K stride.
    (16, 512, 13824, 23),
    (112, 512, 13824, 23),
)
GEMM_SMALL_M_DESCRIPTOR_RESIDENT_LOW_STRIDE_CELLS = (
    # Stride-1536..5120 half of the complete set of A1/B1 descriptor geometries issued
    # by the 910B production GEMM launches whose output has at most 64 rows,
    # excluding the four geometries already published in
    # aic_mte2_gemm_large_resident_descriptor_cann_8_5_1.json.  The domain is
    # defined purely by (rows, columns, source row stride, active AIC groups)
    # decoded from the versioned launch selection; no duration, error metric
    # or operator name selects a cell.  The two halves split on the physical
    # source row stride only, because one batch may carry at most 128 cells.
    # A 128-MiB cyclic source keeps every descriptor line inside the published
    # 192-MiB L2 while preserving its physical K stride.
    (16, 512, 4096, 23),
    (16, 512, 5120, 24),
    (16, 1536, 1536, 23),
    (16, 4096, 4096, 19),
    (16, 4096, 4096, 22),
    (16, 4096, 4096, 23),
    (16, 5120, 5120, 18),
    (16, 5120, 5120, 24),
    (32, 256, 5120, 22),
    (32, 1536, 1536, 23),
    (32, 4096, 4096, 19),
    (32, 4096, 4096, 22),
    (32, 4096, 4096, 24),
    (48, 256, 4096, 22),
    (48, 256, 4096, 23),
    (48, 256, 5120, 22),
    (48, 256, 5120, 24),
    (48, 1536, 1536, 23),
    (64, 256, 4096, 22),
    (64, 256, 4096, 23),
    (64, 256, 5120, 22),
    (64, 256, 5120, 24),
    (64, 512, 5120, 24),
    (64, 1536, 1536, 23),
    (96, 256, 4096, 22),
    (96, 512, 4096, 23),
    (128, 256, 4096, 22),
    (128, 256, 4096, 23),
    (128, 256, 5120, 22),
    (128, 512, 4096, 23),
    (192, 256, 5120, 24),
    (224, 256, 1536, 23),
    (256, 256, 4096, 19),
    (256, 256, 4096, 22),
    (256, 256, 5120, 18),
)
GEMM_SMALL_M_DESCRIPTOR_RESIDENT_HIGH_STRIDE_CELLS = (
    # Stride-8192..14336 half of the complete set of A1/B1 descriptor geometries issued
    # by the 910B production GEMM launches whose output has at most 64 rows,
    # excluding the four geometries already published in
    # aic_mte2_gemm_large_resident_descriptor_cann_8_5_1.json.  The domain is
    # defined purely by (rows, columns, source row stride, active AIC groups)
    # decoded from the versioned launch selection; no duration, error metric
    # or operator name selects a cell.  The two halves split on the physical
    # source row stride only, because one batch may carry at most 128 cells.
    # A 128-MiB cyclic source keeps every descriptor line inside the published
    # 192-MiB L2 while preserving its physical K stride.
    (16, 256, 11008, 22),
    (16, 256, 13824, 23),
    (16, 256, 14336, 22),
    (16, 512, 8192, 22),
    (16, 512, 11008, 22),
    (16, 512, 13824, 23),
    (16, 512, 14336, 22),
    (16, 8192, 8192, 16),
    (16, 8192, 8192, 23),
    (32, 256, 8192, 22),
    (32, 256, 8192, 23),
    (32, 256, 11008, 22),
    (32, 256, 13824, 23),
    (32, 256, 14336, 22),
    (48, 256, 8192, 22),
    (48, 256, 8192, 23),
    (48, 256, 11008, 22),
    (48, 256, 13824, 23),
    (48, 256, 14336, 22),
    (64, 256, 8192, 22),
    (64, 256, 8192, 23),
    (64, 256, 11008, 22),
    (64, 256, 13824, 23),
    (64, 256, 14336, 22),
    (96, 256, 11008, 22),
    (96, 256, 14336, 22),
    (96, 512, 11008, 22),
    (96, 512, 14336, 22),
    (112, 256, 13824, 23),
    (112, 512, 13824, 23),
    (128, 256, 8192, 22),
    (128, 256, 8192, 23),
    (128, 512, 8192, 22),
    (256, 256, 8192, 16),
    (256, 256, 8192, 23),
)
GEMM_SMALL_M_DESCRIPTOR_PENDING_RECHECK_CELLS = (
    # The eleven small-M descriptor geometries whose cross-round median
    # spread exceeded ten percent on a shared card and were therefore
    # withheld from the model.  Re-measured with more independent runs to
    # establish whether the spread is contention or is intrinsic to the
    # geometry.  Membership is inherited from that earlier campaign; no
    # duration or error metric selects a cell.
    (16, 4096, 4096, 19),
    (16, 5120, 5120, 18),
    (16, 8192, 8192, 16),
    (32, 4096, 4096, 19),
    (48, 256, 14336, 22),
    (96, 256, 14336, 22),
    (128, 256, 8192, 22),
    (128, 256, 8192, 23),
    (128, 512, 8192, 22),
    (256, 256, 4096, 19),
    (256, 256, 8192, 16),
)
GEMM_SERVING_DECODE_DESCRIPTOR_CELLS = (
    # Every A1/B1 descriptor geometry the ShareGPT qwen3-8b tp1_qps16
    # Serving projection actually requests and that is not already
    # measured.  The set was collected by logging the operand-group
    # geometry (tile extent, group K, physical K stride, active AIC
    # groups) of that run; no duration or error metric selects a cell.
    (16, 256, 12288, 22),
    (32, 256, 12288, 22),
    (48, 256, 4096, 24),
    (48, 256, 12288, 22),
    (64, 256, 4096, 24),
    (64, 256, 12288, 22),
    (80, 256, 4096, 22),
    (80, 256, 4096, 24),
    (80, 256, 12288, 22),
    (80, 512, 4096, 22),
    (80, 512, 4096, 24),
    (80, 512, 12288, 22),
    (96, 256, 4096, 24),
    (96, 256, 12288, 22),
    (96, 512, 4096, 22),
    (96, 512, 4096, 24),
    (96, 512, 12288, 22),
    (96, 512, 12288, 24),
    (112, 256, 4096, 22),
    (112, 256, 4096, 24),
    (112, 256, 12288, 22),
    (112, 512, 4096, 22),
    (112, 512, 4096, 24),
    (112, 512, 12288, 22),
    (112, 512, 12288, 24),
    (128, 256, 4096, 24),
    (128, 256, 12288, 22),
    (128, 256, 12288, 24),
    (128, 512, 4096, 22),
    (128, 512, 4096, 24),
    (128, 512, 12288, 22),
    (128, 512, 12288, 24),
    (192, 256, 12288, 22),
    (192, 512, 4096, 24),
    (192, 512, 12288, 24),
    (208, 256, 4096, 22),
    (208, 256, 4096, 24),
    (208, 256, 12288, 24),
    (224, 256, 4096, 24),
    (224, 256, 12288, 24),
    (240, 256, 4096, 24),
    (240, 256, 12288, 24),
    (256, 256, 4096, 24),
    (256, 256, 12288, 24),
    (320, 256, 4096, 24),
)
GEMM_LARGE_CONTIGUOUS_DESCRIPTOR_CELLS = tuple(
    # Extend the generic contiguous ND2NZ service beyond its existing 64-KiB
    # endpoint.  The two 128-KiB geometries are the exact B1 descriptors from
    # the small-M V2 and A4/B4 V3 programs; paired 256-KiB shapes test whether
    # byte count alone remains a valid service coordinate.  Every cell keeps
    # disjoint per-AIC source partitions and measures only descriptor
    # completion/stream interval, never a production GEMM.
    (rows, columns, columns, active_aic_groups)
    for rows, columns in (
        (128, 512), (256, 256),
        (128, 1024), (256, 512),
    )
    for active_aic_groups in (1, 8, 22, 24)
)
PROFILE_NAMES = (
    "canonical", "deterministic_split_k", "paged_gqa_128",
    "paged_gqa_tail", "paged_gqa_tail_resolution",
    "paged_gqa_tail_exhaustive_lo", "paged_gqa_tail_exhaustive_hi",
    "paged_gqa_tail_group_transfer",
    "fia_score_p_short_stride",
    "fia_score_p_regular_gqa",
    "gemm_n_major_b_cold", "gemm_legacy_k14336_cold",
    "gemm_v2_k14336_exact_cold",
    "gemm_v3_k1536_n5120_cold", "gemm_small_m_remaining_cold",
    "gemm_v2_k5120_n5120_cold", "gemm_v2_k5120_n1536_cold",
    "gemm_v2_key98499_cold", "gemm_k5120_n13824_source_order_cold",
    "gemm_p90_boundary_cold", "gemm_max_ape_k4096_n11008_cold",
    "gemm_max_ape_k13824_n5120_cold",
    "gemm_v2_k13824_n5120_resident",
    "gemm_serving_decode_descriptor",
    "gemm_small_m_descriptor_pending_recheck",
    "gemm_small_m_descriptor_resident_low_stride",
    "gemm_small_m_descriptor_resident_high_stride",
    "gemm_large_contiguous_descriptor",
    "gemm_decode_v2_cold",
    "gemm_decode_b1_slab_natural_orientation",
    "fused_moe_k2880_b_cold", "fused_moe_k6144_b_cold")
DEFAULT_FOOTPRINT_MB = 512
GEMM_N_MAJOR_B_COLD_FOOTPRINT_MB = 51200
GEMM_N_MAJOR_B_COLD_ITERATIONS = 640
GEMM_LEGACY_K14336_COLD_FOOTPRINT_MB = 8192
GEMM_LEGACY_K14336_COLD_ITERATIONS = 224
GEMM_V3_K1536_N5120_COLD_FOOTPRINT_MB = 8192
GEMM_V3_K1536_N5120_COLD_ITERATIONS = 224
GEMM_SMALL_M_REMAINING_COLD_FOOTPRINT_MB = 8192
GEMM_SMALL_M_REMAINING_COLD_ITERATIONS = 224
GEMM_V2_K5120_N5120_COLD_FOOTPRINT_MB = 8192
GEMM_V2_K5120_N5120_COLD_ITERATIONS = 224
GEMM_V2_K5120_N1536_COLD_FOOTPRINT_MB = 8192
GEMM_V2_K5120_N1536_COLD_ITERATIONS = 224
GEMM_V2_KEY98499_COLD_FOOTPRINT_MB = 8192
GEMM_V2_KEY98499_COLD_ITERATIONS = 224
GEMM_K5120_N13824_SOURCE_ORDER_COLD_FOOTPRINT_MB = 8192
GEMM_K5120_N13824_SOURCE_ORDER_COLD_ITERATIONS = 224
GEMM_P90_BOUNDARY_COLD_FOOTPRINT_MB = 8192
GEMM_P90_BOUNDARY_COLD_ITERATIONS = 224
GEMM_MAX_APE_K4096_N11008_COLD_FOOTPRINT_MB = 8192
GEMM_MAX_APE_K4096_N11008_COLD_ITERATIONS = 224
TRIALS = 4
CHILD_FIELDS = 17
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "aic_mte2_nd2nz_stride_kernel.cpp",
    "aic_mte2_nd2nz_stride_main.cpp",
    "aic_mte2_nd2nz_stride_runner.py",
)
_ACTIVE_PROCESS = None


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_snapshot_sha256(root):
    digest = hashlib.sha256()
    for filename in SOURCE_FILENAMES:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"benchmark source is missing: {path}")
        payload = path.read_bytes()
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _positive_integer(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _cells(profile):
    if profile in (
            "gemm_v3_k1536_n5120_cold", "gemm_small_m_remaining_cold",
            "gemm_v2_k5120_n5120_cold",
            "gemm_v2_k14336_exact_cold",
            "gemm_v2_k5120_n1536_cold", "gemm_v2_key98499_cold",
            "gemm_k5120_n13824_source_order_cold",
            "gemm_p90_boundary_cold", "gemm_max_ape_k4096_n11008_cold",
            "gemm_max_ape_k13824_n5120_cold",
            "gemm_v2_k13824_n5120_resident",
            "gemm_serving_decode_descriptor",
            "gemm_small_m_descriptor_pending_recheck",
            "gemm_small_m_descriptor_resident_low_stride",
            "gemm_small_m_descriptor_resident_high_stride",
            "gemm_large_contiguous_descriptor",
            "gemm_decode_v2_cold",
            "gemm_decode_b1_slab_natural_orientation",
            "fused_moe_k2880_b_cold", "fused_moe_k6144_b_cold"):
        profile_cells = {
            "gemm_v3_k1536_n5120_cold":
                GEMM_V3_K1536_N5120_COLD_CELLS,
            "gemm_small_m_remaining_cold":
                GEMM_SMALL_M_REMAINING_COLD_CELLS,
            "gemm_v2_k5120_n5120_cold":
                GEMM_V2_K5120_N5120_COLD_CELLS,
            "gemm_v2_k14336_exact_cold":
                GEMM_V2_K14336_EXACT_COLD_CELLS,
            "gemm_v2_k5120_n1536_cold":
                GEMM_V2_K5120_N1536_COLD_CELLS,
            "gemm_v2_key98499_cold": GEMM_V2_KEY98499_COLD_CELLS,
            "gemm_k5120_n13824_source_order_cold":
                GEMM_K5120_N13824_SOURCE_ORDER_COLD_CELLS,
            "gemm_p90_boundary_cold": GEMM_P90_BOUNDARY_COLD_CELLS,
            "gemm_max_ape_k4096_n11008_cold":
                GEMM_MAX_APE_K4096_N11008_COLD_CELLS,
            "gemm_max_ape_k13824_n5120_cold":
                GEMM_MAX_APE_K13824_N5120_COLD_CELLS,
            "gemm_v2_k13824_n5120_resident":
                GEMM_V2_K13824_N5120_RESIDENT_CELLS,
            "gemm_serving_decode_descriptor":
                GEMM_SERVING_DECODE_DESCRIPTOR_CELLS,
            "gemm_small_m_descriptor_pending_recheck":
                GEMM_SMALL_M_DESCRIPTOR_PENDING_RECHECK_CELLS,
            "gemm_small_m_descriptor_resident_low_stride":
                GEMM_SMALL_M_DESCRIPTOR_RESIDENT_LOW_STRIDE_CELLS,
            "gemm_small_m_descriptor_resident_high_stride":
                GEMM_SMALL_M_DESCRIPTOR_RESIDENT_HIGH_STRIDE_CELLS,
            "gemm_large_contiguous_descriptor":
                GEMM_LARGE_CONTIGUOUS_DESCRIPTOR_CELLS,
            "gemm_decode_v2_cold": GEMM_DECODE_V2_COLD_CELLS,
            "gemm_decode_b1_slab_natural_orientation":
                GEMM_DECODE_B1_SLAB_NATURAL_ORIENTATION_CELLS,
            "fused_moe_k2880_b_cold": FUSED_MOE_K2880_B_COLD_CELLS,
            "fused_moe_k6144_b_cold": FUSED_MOE_K6144_B_COLD_CELLS,
        }[profile]
        return tuple(
            (rows, columns, source_row_stride, mode, active_groups)
            for rows, columns, source_row_stride, active_groups in (
                profile_cells)
            for mode in (0, 1)
        )
    if profile == "canonical":
        geometries = CANONICAL_GEOMETRIES
        active_aic_groups = (8, 16, 24)
    elif profile == "deterministic_split_k":
        geometries = DETERMINISTIC_SPLIT_K_GEOMETRIES
        active_aic_groups = (24,)
    elif profile == "paged_gqa_128":
        geometries = PAGED_GQA_128_GEOMETRIES
        active_aic_groups = (8, 16, 24)
    elif profile == "paged_gqa_tail":
        geometries = PAGED_GQA_TAIL_GEOMETRIES
        active_aic_groups = (22, 23, 24)
    elif profile == "paged_gqa_tail_resolution":
        geometries = PAGED_GQA_TAIL_RESOLUTION_GEOMETRIES
        active_aic_groups = (22, 23, 24)
    elif profile == "paged_gqa_tail_exhaustive_lo":
        geometries = PAGED_GQA_TAIL_EXHAUSTIVE_LO_GEOMETRIES
        active_aic_groups = (24,)
    elif profile == "paged_gqa_tail_exhaustive_hi":
        geometries = PAGED_GQA_TAIL_EXHAUSTIVE_HI_GEOMETRIES
        active_aic_groups = (24,)
    elif profile == "paged_gqa_tail_group_transfer":
        geometries = PAGED_GQA_TAIL_GROUP_TRANSFER_GEOMETRIES
        active_aic_groups = (22, 23)
    elif profile == "fia_score_p_short_stride":
        geometries = FIA_SCORE_P_SHORT_STRIDE_GEOMETRIES
        active_aic_groups = (22,)
    elif profile == "fia_score_p_regular_gqa":
        geometries = FIA_SCORE_P_REGULAR_GQA_GEOMETRIES
        active_aic_groups = (8, 16, 22, 24)
    elif profile == "gemm_n_major_b_cold":
        geometries = GEMM_N_MAJOR_B_COLD_GEOMETRIES
        active_aic_groups = (23,)
    elif profile == "gemm_legacy_k14336_cold":
        geometries = GEMM_LEGACY_K14336_COLD_GEOMETRIES
        active_aic_groups = (22,)
    else:
        raise ValueError(f"unknown benchmark profile: {profile!r}")
    return tuple(
        (rows, columns, source_row_stride, mode, active_groups)
        for rows, columns, source_row_stride in geometries
        for mode in (0, 1)
        for active_groups in active_aic_groups
    )


def _execution_parameters(profile, footprint_mb, iterations):
    if profile == "gemm_n_major_b_cold":
        resolved_footprint_mb = (
            GEMM_N_MAJOR_B_COLD_FOOTPRINT_MB
            if footprint_mb is None else footprint_mb)
        resolved_iterations = (
            GEMM_N_MAJOR_B_COLD_ITERATIONS
            if iterations is None else iterations)
    elif profile == "gemm_legacy_k14336_cold":
        resolved_footprint_mb = (
            GEMM_LEGACY_K14336_COLD_FOOTPRINT_MB
            if footprint_mb is None else footprint_mb)
        resolved_iterations = (
            GEMM_LEGACY_K14336_COLD_ITERATIONS
            if iterations is None else iterations)
    elif profile == "gemm_v2_k14336_exact_cold":
        resolved_footprint_mb = (
            GEMM_LEGACY_K14336_COLD_FOOTPRINT_MB
            if footprint_mb is None else footprint_mb)
        resolved_iterations = (
            GEMM_LEGACY_K14336_COLD_ITERATIONS
            if iterations is None else iterations)
    elif profile in ("fused_moe_k2880_b_cold", "fused_moe_k6144_b_cold"):
        resolved_footprint_mb = (
            GEMM_SMALL_M_REMAINING_COLD_FOOTPRINT_MB
            if footprint_mb is None else footprint_mb)
        resolved_iterations = (
            GEMM_SMALL_M_REMAINING_COLD_ITERATIONS
            if iterations is None else iterations)
    elif profile == "gemm_decode_v2_cold":
        resolved_footprint_mb = (
            GEMM_SMALL_M_REMAINING_COLD_FOOTPRINT_MB
            if footprint_mb is None else footprint_mb)
        resolved_iterations = (
            GEMM_SMALL_M_REMAINING_COLD_ITERATIONS
            if iterations is None else iterations)
    elif profile == "gemm_v3_k1536_n5120_cold":
        resolved_footprint_mb = (
            GEMM_V3_K1536_N5120_COLD_FOOTPRINT_MB
            if footprint_mb is None else footprint_mb)
        resolved_iterations = (
            GEMM_V3_K1536_N5120_COLD_ITERATIONS
            if iterations is None else iterations)
    elif profile == "gemm_small_m_remaining_cold":
        resolved_footprint_mb = (
            GEMM_SMALL_M_REMAINING_COLD_FOOTPRINT_MB
            if footprint_mb is None else footprint_mb)
        resolved_iterations = (
            GEMM_SMALL_M_REMAINING_COLD_ITERATIONS
            if iterations is None else iterations)
    elif profile == "gemm_v2_k5120_n5120_cold":
        resolved_footprint_mb = (
            GEMM_V2_K5120_N5120_COLD_FOOTPRINT_MB
            if footprint_mb is None else footprint_mb)
        resolved_iterations = (
            GEMM_V2_K5120_N5120_COLD_ITERATIONS
            if iterations is None else iterations)
    elif profile == "gemm_v2_k5120_n1536_cold":
        resolved_footprint_mb = (
            GEMM_V2_K5120_N1536_COLD_FOOTPRINT_MB
            if footprint_mb is None else footprint_mb)
        resolved_iterations = (
            GEMM_V2_K5120_N1536_COLD_ITERATIONS
            if iterations is None else iterations)
    elif profile == "gemm_v2_key98499_cold":
        resolved_footprint_mb = (
            GEMM_V2_KEY98499_COLD_FOOTPRINT_MB
            if footprint_mb is None else footprint_mb)
        resolved_iterations = (
            GEMM_V2_KEY98499_COLD_ITERATIONS
            if iterations is None else iterations)
    elif profile == "gemm_k5120_n13824_source_order_cold":
        resolved_footprint_mb = (
            GEMM_K5120_N13824_SOURCE_ORDER_COLD_FOOTPRINT_MB
            if footprint_mb is None else footprint_mb)
        resolved_iterations = (
            GEMM_K5120_N13824_SOURCE_ORDER_COLD_ITERATIONS
            if iterations is None else iterations)
    elif profile == "gemm_p90_boundary_cold":
        resolved_footprint_mb = (
            GEMM_P90_BOUNDARY_COLD_FOOTPRINT_MB
            if footprint_mb is None else footprint_mb)
        resolved_iterations = (
            GEMM_P90_BOUNDARY_COLD_ITERATIONS
            if iterations is None else iterations)
    elif profile in (
            "gemm_max_ape_k4096_n11008_cold",
            "gemm_max_ape_k13824_n5120_cold",
            "gemm_large_contiguous_descriptor"):
        resolved_footprint_mb = (
            GEMM_MAX_APE_K4096_N11008_COLD_FOOTPRINT_MB
            if footprint_mb is None else footprint_mb)
        resolved_iterations = (
            GEMM_MAX_APE_K4096_N11008_COLD_ITERATIONS
            if iterations is None else iterations)
    elif profile in PROFILE_NAMES:
        resolved_footprint_mb = (
            DEFAULT_FOOTPRINT_MB if footprint_mb is None else footprint_mb)
        resolved_iterations = iterations
    else:
        raise ValueError(f"unknown benchmark profile: {profile!r}")
    return resolved_footprint_mb, resolved_iterations


def _stop_child(signum, _frame):
    if _ACTIVE_PROCESS is not None:
        try:
            os.killpg(_ACTIVE_PROCESS.pid, signal.SIGTERM)
        except ProcessLookupError as error:
            raise SystemExit(128 + signum) from error
    raise SystemExit(128 + signum)


def main():
    global _ACTIVE_PROCESS

    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--profile", choices=PROFILE_NAMES, default="canonical")
    parser.add_argument("--trials", type=_positive_integer, default=TRIALS)
    parser.add_argument(
        "--footprint-mb", type=_positive_integer)
    parser.add_argument("--iterations", type=_positive_integer)
    parser.add_argument(
        "--child-timeout-seconds", type=_positive_integer, default=900)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")
    footprint_mb, requested_iterations = _execution_parameters(
        args.profile, args.footprint_mb, args.iterations)

    benchmark_root = Path(__file__).resolve().parent
    print(
        f"runner_pid={os.getpid()} device_id={args.device_id} "
        f"run_id={args.run_id} trials={args.trials} "
        f"profile={args.profile} "
        f"footprint_mb={footprint_mb} iterations={requested_iterations} "
        f"binary={args.binary.resolve()} "
        f"binary_sha256={_sha256(args.binary)} "
        f"source_snapshot_sha256={_source_snapshot_sha256(benchmark_root)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "run_id,trial,order_direction,order_position,process_pid,"
        "rows,columns,source_row_stride,mode,source_tile_span_bytes,copied_bytes,"
        "active_aic_groups,iterations,short_cycles,short_wall_ns,"
        "long_cycles,long_wall_ns,tick_ns,cycles_per_copy,ns_per_copy,"
        "bandwidth_gbs,validation",
        flush=True,
    )
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    signal.signal(signal.SIGINT, _stop_child)
    signal.signal(signal.SIGTERM, _stop_child)
    profile_cells = _cells(args.profile)
    for trial in range(args.trials):
        if trial % 2 == 0:
            order_direction = "forward"
            cells = profile_cells
        else:
            order_direction = "reverse"
            cells = tuple(reversed(profile_cells))
        command = [str(args.binary), "--batch", str(footprint_mb)]
        for rows, columns, source_row_stride, mode, active_groups in cells:
            copied_bytes = rows * columns * 2
            iterations = (
                requested_iterations if requested_iterations is not None
                else max(
                    20000,
                    footprint_mb * 1024 * 1024 // copied_bytes,
                )
            )
            command.extend((
                str(rows), str(columns), str(source_row_stride),
                str(mode), str(active_groups), str(iterations),
                # worker_group 1 = the disjoint per-AIC partitions these
                # timing cells are defined on.
                "1",
            ))
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            start_new_session=True,
        )
        _ACTIVE_PROCESS = process
        try:
            stdout, stderr = process.communicate(
                timeout=args.child_timeout_seconds)
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
            raise RuntimeError(
                f"child PID {process.pid} timed out after "
                f"{args.child_timeout_seconds} seconds") from error
        finally:
            _ACTIVE_PROCESS = None
        if stderr:
            print(
                f"child_pid={process.pid} trial={trial + 1} "
                f"stderr={stderr.rstrip()}",
                file=sys.stderr,
                flush=True,
            )
        if process.returncode != 0:
            raise RuntimeError(
                f"child PID {process.pid} failed with exit "
                f"{process.returncode}")
        output_lines = stdout.strip().splitlines()
        if len(output_lines) != len(cells):
            raise RuntimeError(
                f"child PID {process.pid} emitted "
                f"{len(output_lines)} data lines")
        for order_position, output_line in enumerate(output_lines):
            fields = output_line.split(",")
            if len(fields) != CHILD_FIELDS:
                raise RuntimeError(
                    f"child PID {process.pid} emitted an invalid raw schema")
            expected = cells[order_position]
            observed = tuple(int(fields[index]) for index in (0, 1, 2, 3, 6))
            if observed != expected or fields[-1] != "pass":
                raise RuntimeError(
                    f"child PID {process.pid} changed the requested order")
            print(
                f"{args.run_id},{trial + 1},{order_direction},"
                f"{order_position},{process.pid},{output_line}",
                flush=True,
            )


if __name__ == "__main__":
    main()
