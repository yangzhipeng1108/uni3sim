// The C1 flash_fwd_splitkv NCU launch observation fixes 82,944 B/CTA and two
// resident 128-thread CTAs per GA100 SM.  Reuse the independent K/V-only
// fragment with that exact resource topology; it contains no Attention work.
#define FA2_INGRESS_DYNAMIC_SHARED_BYTES (81 * 1024)
#define FA2_INGRESS_EXPECTED_RESIDENT_CTAS 2
#define FA2_INGRESS_SHARED_BUFFER_COUNT 1
#include "fa2_chunked_one_resident_ingress.cu"
