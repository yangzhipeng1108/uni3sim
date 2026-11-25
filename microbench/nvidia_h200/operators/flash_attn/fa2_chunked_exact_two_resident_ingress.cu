// NVIDIA H200 owner copy of
// microbench/nvidia_a800/operators/flash_attn/fa2_chunked_exact_two_resident_ingress.cu.
// Unchanged resource class: 82,944 B (81 KiB) dynamic shared memory still
// admits exactly two resident 128-thread CTAs on a 228-KiB H200 SM
// (3 x (82,944 + 1,024 reserved) B > 233,472 B), checked at runtime.  The
// included ingress fragment is the H200 owner copy (132 SMs, sm_90).
// The C1 flash_fwd_splitkv NCU launch observation fixes 82,944 B/CTA and two
// resident 128-thread CTAs per GA100 SM.  Reuse the independent K/V-only
// fragment with that exact resource topology; it contains no Attention work.
#define FA2_INGRESS_DYNAMIC_SHARED_BYTES (81 * 1024)
#define FA2_INGRESS_EXPECTED_RESIDENT_CTAS 2
#define FA2_INGRESS_SHARED_BUFFER_COUNT 1
#include "fa2_chunked_one_resident_ingress.cu"
