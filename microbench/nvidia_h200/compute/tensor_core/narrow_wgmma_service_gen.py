"""Generate a whole-loop PTX control for compiler accumulator boundaries.

The paired legacy probe exposes each MMA result to C++. This control keeps
all accumulators and the round loop inside one asm, with one fence per batch.
It measures an instruction window, never a complete matrix multiplication.
"""
from pathlib import Path
import sys


def generate(output: Path):
    pieces = ["template<int N,int Acc,int Pending> struct MatrixWindowRequest;\n"]
    for width in (8, 16, 32, 64):
        count = width // 2
        for chains in (1, 2, 4):
            total = count * chains
            outputs = ", ".join(
                f'"+f"(d[{chain}][{i}])'
                for chain in range(chains) for i in range(count))
            for pending in (0, 1, 3):
                ptx = ["{", ".reg .b32 r;", ".reg .pred p;",
                       f".reg .b64 a<{chains}>, b<{chains}>;",
                       "mov.u32 r, 0;", "setp.eq.u32 p, 0, 0;", "LOOP:",
                       "wgmma.fence.sync.aligned;"]
                for chain in range(chains):
                    ptx.extend((f"add.u64 a{chain}, %{total}, {chain * 512};",
                                f"add.u64 b{chain}, %{total + 1}, {chain * 512};"))
                for step in range(4):
                    for chain in range(chains):
                        registers = ", ".join(f"%{chain * count + i}" for i in range(count))
                        ptx.append(
                            f"wgmma.mma_async.sync.aligned.m64n{width}k16.f32.f16.f16 "
                            f"{{{registers}}}, a{chain}, b{chain}, p, 1, 1, 0, 0;")
                        if step < 3:
                            ptx.extend((f"add.u64 a{chain}, a{chain}, 2;",
                                        f"add.u64 b{chain}, b{chain}, 2;"))
                ptx.extend(("wgmma.commit_group.sync.aligned;",
                            f"wgmma.wait_group.sync.aligned {pending};",
                            "add.u32 r, r, 1;", f"setp.lt.u32 p, r, %{total + 2};",
                            "@p bra LOOP;", "wgmma.wait_group.sync.aligned 0;", "}"))
                # p stays true on every taken loop edge, including the first.
                assembly = "\n".join('"' + line + '\\n"' for line in ptx)
                pieces.append(
                    f"template<> struct MatrixWindowRequest<{width},{chains},{pending}> {{\n"
                    f"__device__ __forceinline__ static void issue(float (&d)[{chains}][{count}], "
                    "uint64_t a, uint64_t b, int rounds) {\nasm volatile(\n"
                    f"{assembly}\n: {outputs} : \"l\"(a), \"l\"(b), \"r\"(rounds) : \"memory\");\n}}}};\n")
    output.write_text("".join(pieces))


if __name__ == "__main__":
    generate(Path(sys.argv[1]))
