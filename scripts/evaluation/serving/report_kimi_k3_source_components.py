#!/usr/bin/env python3
"""Publish independent K3 source components; no complete model/runtime claim."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.simulation.model_operator_probe import (
    write_kimi_k3_source_components, write_kimi_k3_nope_prefill_components, write_kimi_k3_nope_decode_components,
    write_kimi_k3_native_latent_output_components, write_kimi_k3_marlin_routed_output_components,
    write_kimi_k3_native_marlin_moe_components, write_kimi_k3_nope_prefill_layer_components,
    write_kimi_k3_kda_fresh_prefill_components, write_kimi_k3_dense_kda_prefill_layer_components,
    write_kimi_k3_fresh_prefill_decoder_stack_components, write_kimi_k3_fresh_prefill_logits_components,
    write_kimi_k3_fresh_prefill_forward_logits_components, write_kimi_k3_single_decode_forward_logits_components,
    write_kimi_k3_fresh_prefill_greedy_step_components, write_kimi_k3_single_decode_greedy_step_components,
    write_kimi_k3_situ_offline_compile, write_kimi_k3_nope_mixed_components, write_kimi_k3_kda_mixed_components, write_kimi_k3_mixed_greedy_step_components,
    write_kimi_k3_kda_context_prefill_components, write_kimi_k3_nope_context_prefill_components,
    write_kimi_k3_context_prefill_greedy_step_components, write_kimi_k3_nope_context_continuation_components,
    write_kimi_k3_context_continuation_greedy_step_components)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--component", choices=("existing", "nope-prefill", "nope-decode", "nope-mixed", "kda-mixed", "kda-context-prefill", "nope-context-prefill", "nope-context-continuation", "context-prefill-greedy-step",
                                               "context-continuation-greedy-step", "mixed-greedy-step", "native-latent-output", "marlin-routed-output",
                                               "native-marlin-moe", "nope-prefill-layer",
                                               "kda-fresh-prefill", "dense-kda-prefill-layer",
                                               "fresh-prefill-decoder-stack", "fresh-prefill-logits",
                                               "fresh-prefill-forward-logits", "single-decode-forward-logits",
                                               "fresh-prefill-greedy-step", "single-decode-greedy-step", "situ-offline-compile"), default="existing",
                        help="select one source branch without repeating prior component evaluation")
    parser.add_argument("--backend-supports-out", choices=("true", "false"),
                        help="required declared backend capability for continuation components")
    args = parser.parse_args()
    writers = {"existing": write_kimi_k3_source_components,
               "nope-prefill": write_kimi_k3_nope_prefill_components,
               "nope-decode": write_kimi_k3_nope_decode_components,
               "nope-mixed": write_kimi_k3_nope_mixed_components,
               "kda-mixed": write_kimi_k3_kda_mixed_components,
               "kda-context-prefill": write_kimi_k3_kda_context_prefill_components,
               "nope-context-prefill": write_kimi_k3_nope_context_prefill_components,
               "nope-context-continuation": write_kimi_k3_nope_context_continuation_components,
               "context-continuation-greedy-step": write_kimi_k3_context_continuation_greedy_step_components,
               "context-prefill-greedy-step": write_kimi_k3_context_prefill_greedy_step_components,
               "mixed-greedy-step": write_kimi_k3_mixed_greedy_step_components,
               "native-latent-output": write_kimi_k3_native_latent_output_components,
               "marlin-routed-output": write_kimi_k3_marlin_routed_output_components,
               "native-marlin-moe": write_kimi_k3_native_marlin_moe_components,
               "nope-prefill-layer": write_kimi_k3_nope_prefill_layer_components,
               "kda-fresh-prefill": write_kimi_k3_kda_fresh_prefill_components,
               "dense-kda-prefill-layer": write_kimi_k3_dense_kda_prefill_layer_components,
               "fresh-prefill-decoder-stack": write_kimi_k3_fresh_prefill_decoder_stack_components,
               "fresh-prefill-logits": write_kimi_k3_fresh_prefill_logits_components,
               "fresh-prefill-forward-logits": write_kimi_k3_fresh_prefill_forward_logits_components,
               "single-decode-forward-logits": write_kimi_k3_single_decode_forward_logits_components,
               "fresh-prefill-greedy-step": write_kimi_k3_fresh_prefill_greedy_step_components,
               "single-decode-greedy-step": write_kimi_k3_single_decode_greedy_step_components,
               "situ-offline-compile": write_kimi_k3_situ_offline_compile}
    writer = writers[args.component]
    continuation = args.component in ("nope-context-continuation", "context-continuation-greedy-step")
    if continuation:
        if args.backend_supports_out is None:
            parser.error("continuation component requires --backend-supports-out true|false")
        result = writer(args.model, args.source_root, args.output, backend_supports_out=args.backend_supports_out == "true")
    else:
        if args.backend_supports_out is not None:
            parser.error("--backend-supports-out applies only to continuation components")
        result = writer(args.model, args.source_root, args.output)
    label = "offline compiler instances" if args.component == "situ-offline-compile" else "source calls/components"
    print(f"{args.output}: {result['component_count']} {label}; full model/runtime=false")


if __name__ == "__main__":
    main()
