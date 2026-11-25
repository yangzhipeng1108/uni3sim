"""Generate the probe's inline PTX helper from the pinned FA3 descriptor ABI."""

from pathlib import Path
import sys


def generate(reference: Path, output: Path) -> None:
    source = reference.read_text()
    marker = "__device__ __forceinline__ void wgmma_ss_n64_kmajor"
    if source.count(marker) != 1:
        raise ValueError("FA3 descriptor helper boundary changed")
    preamble = source[:source.index(marker)]
    registers = ", ".join(f"%{index}" for index in range(64))
    outputs = ", ".join(f'"+f"(d[{index}])' for index in range(64))
    inputs = ", ".join(
        f'"l"({side}[{step}])' for step in range(4) for side in ("a", "b"))
    inputs += ', "r"(scale_d)'
    lines = [preamble]
    for wait in (0, 1):
        instructions = [
            '"{\\n.reg .pred p;\\nsetp.ne.b32 p, %72, 0;\\n"'
        ]
        for step in range(4):
            a_index = 64 + step * 2
            b_index = a_index + 1
            instructions.append(
                '"wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 '
                f'{{{registers}}}, %{a_index}, %{b_index}, p, 1, 1, 0, 0;\\n"')
        instructions += [
            '"wgmma.commit_group.sync.aligned;\\n"',
            f'"wgmma.wait_group.sync.aligned {wait};\\n}}"',
        ]
        lines.append(
            f"\n__device__ __forceinline__ void wgmma_ss_n128_group_wait{wait}(\n"
            "    float (&d)[64], const uint64_t (&a)[4], const uint64_t (&b)[4], int scale_d) {\n"
            "  asm volatile(\n    " + "\n    ".join(instructions) + "\n"
            "    : " + outputs + "\n"
            "    : " + inputs + ");\n}\n"
        )
    output.write_text("".join(lines))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: wgmma_ss_n128_shortk_issue_gen.py <fa3.cu> <output.cuh>")
    generate(Path(sys.argv[1]), Path(sys.argv[2]))
