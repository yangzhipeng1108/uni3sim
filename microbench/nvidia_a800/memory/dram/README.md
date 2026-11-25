# NVIDIA A800 DRAM mechanism benchmarks

## Direction-transition matched sequences

`direction_transition.cu` isolates the cost of changing the program-issued stream direction. It launches independent `extern "C"` read and write phase kernels in six four-phase sequences. `RRWW`, `WWRR`, `RWRW`, and `WRWR` have the same per-phase bytes and launch geometry but different order; `RRRR` and `WWWW` are no-transition startup controls. `M` repeats the four-label pattern, using a fresh direction-local window for every launch rather than replaying an address range. Thus every mixed sequence uses exactly the same read windows `R0..R(2M-1)` and write windows `W0..W(2M-1)` with the same deterministic per-occurrence values; only program order changes.

The declared matrix is:

- phase footprint `q`: 64 or 256 MiB;
- pattern repetitions `M`: 1, 2, or 4;
- grid: 108, 216, or 432 blocks, with 256 threads per block;
- address roles: allocation A for reads and B for writes (`ab`), or the swapped assignment (`ba`).

Every phase footprint exceeds the A800 40-MiB L2. Read and write roles use different allocations, and direction-local occurrence selects a non-overlapping window in that allocation, so no timed launch aliases or reuses a source/destination window. Reads are coalesced 128-bit `ld.global.cg` operations (L1 bypass), and writes are coalesced 128-bit `st.global.wb` operations (explicit write-back policy). The timed read symbol is strictly load-only: each volatile inline load feeds a register-only XOR invariant whose impossible-on-valid-data branch traps, preventing `ptxas` from deleting the load without adding global output. Independent full-buffer read validation and deterministic write validation are synchronized outside the event interval. Each trial prepares all direction-role slots in a fixed order, independent of the selected sequence. An untimed 64-MiB launch in each address-role assignment loads both phase kernels before any trial; the subsequent full preparation pass overwrites those warmup windows.

The full sweep is trial-first. Within each fixed `(trial,q,M,grid,address_roles)` group, sequences use a compile-time cyclic permutation: trial 0 starts at `RRWW`, trials 1 through 5 rotate the first label through `WWRR`, `RWRW`, `WRWR`, `RRRR`, and `WWWW`, and trial 6 repeats the trial-0 order. This counterbalancing is fixed without consulting measurements.

Build and inspect the host-only configuration contract:

```bash
/usr/local/cuda-12.2/bin/nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo \
  -o direction_transition direction_transition.cu
./direction_transition --describe
```

Run all declared cases or one counter-capture case only after confirming the assigned A800 is idle:

```bash
CUDA_VISIBLE_DEVICES=3 ./direction_transition
CUDA_VISIBLE_DEVICES=3 ./direction_transition --single RWRW 64 2 216 ab
```

Stdout is one CSV row per trial:

```text
sequence,q_mib,M,grid,address_roles,trial,sequence_order_slot,phase_launches,event_ns,logical_read_bytes,logical_write_bytes,program_read_to_write,program_write_to_read,program_transition_count,buffer_validation
```

The transition columns are counts of adjacent labels in the launched program, including repetition boundaries. They must not be described as controller-visible transitions. NCU duration and elapsed-cycle evidence are disabled for this benchmark: NCU is allowed only for non-duration counters and SASS evidence that verify traffic, cache policy, and controller behavior.
