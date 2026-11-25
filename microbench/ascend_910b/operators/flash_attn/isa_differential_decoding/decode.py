#!/usr/bin/env python3
"""Report per-opcode-prefix count deltas between two AI-core objects.

The class whose delta equals the repetition-count difference is the operation
under test.  Instructions are four bytes, little-endian; register fields sit in
the low bits, so aggregate on a high-bit prefix.
"""

import argparse
import collections
import re
import subprocess

OBJDUMP = ("/usr/local/Ascend/cann-8.5.1/aarch64-linux/"
           "ccec_compiler/bin/llvm-objdump")
_LINE = re.compile(r"\s*([0-9a-f]{4,16})\s+((?:[0-9a-f]{8}\s+)+)")


def words(obj, section=".text"):
    """Return the section's instructions as little-endian hex words."""
    dumped = subprocess.run(
        [OBJDUMP, "-s", "--section=" + section, obj],
        capture_output=True, text=True, check=True).stdout
    out = []
    for line in dumped.splitlines():
        matched = _LINE.match(line)
        if matched is None:
            continue
        for word in matched.group(2).split():
            out.append(bytes.fromhex(word)[::-1].hex())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("low_object")
    parser.add_argument("high_object")
    parser.add_argument("--section", default=".text")
    parser.add_argument("--prefix-nibbles", type=int, default=3)
    args = parser.parse_args()
    low = words(args.low_object, args.section)
    high = words(args.high_object, args.section)
    print(f"words {len(low)} -> {len(high)}  delta {len(high) - len(low)}")
    counted_low = collections.Counter(
        word[:args.prefix_nibbles] for word in low)
    counted_high = collections.Counter(
        word[:args.prefix_nibbles] for word in high)
    for key in sorted(set(counted_low) | set(counted_high)):
        delta = counted_high[key] - counted_low[key]
        if delta:
            print(f"   {key}  {delta:+5d}")


if __name__ == "__main__":
    main()
