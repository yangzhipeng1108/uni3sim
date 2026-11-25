#!/usr/bin/env python3
"""Count and locate an identified opcode class inside a shipped kernel function.

The positional histogram is the robust output: it does not depend on having
enumerated every encoding of the operation.
"""

import argparse
import re
import subprocess

from decode import OBJDUMP, _LINE


def function_words(obj, start, size, section=".text"):
    dumped = subprocess.run(
        [OBJDUMP, "-s", "--section=" + section, obj],
        capture_output=True, text=True, check=True).stdout
    by_address = {}
    for line in dumped.splitlines():
        matched = _LINE.match(line)
        if matched is None:
            continue
        base = int(matched.group(1), 16)
        for index, word in enumerate(matched.group(2).split()):
            by_address[base + index * 4] = bytes.fromhex(word)[::-1].hex()
    return [(address, by_address[address])
            for address in range(start, start + size, 4)
            if address in by_address]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("object")
    parser.add_argument("--start", required=True,
                        type=lambda text: int(text, 0))
    parser.add_argument("--size", required=True,
                        type=lambda text: int(text, 0))
    parser.add_argument("--prefix", required=True,
                        help="high-nibble opcode prefix, e.g. 414")
    parser.add_argument("--buckets", type=int, default=8)
    args = parser.parse_args()
    listed = function_words(args.object, args.start, args.size)
    selected = [(address, word) for address, word in listed
                if word.startswith(args.prefix)]
    buckets = [0] * args.buckets
    for address, _ in selected:
        index = (address - args.start) * args.buckets // args.size
        buckets[min(args.buckets - 1, index)] += 1
    share = 100.0 * len(selected) / len(listed) if listed else 0.0
    print(f"instructions {len(listed)}")
    print(f"class {args.prefix}: {len(selected)} ({share:.1f}%)")
    print(f"per-bucket: {buckets}")


if __name__ == "__main__":
    main()
