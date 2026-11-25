"""Generate standalone FP16 WGMMA widths using existing descriptor primitives."""
from pathlib import Path
import sys


def generate(source: Path, output: Path):
    original = source.read_text()
    helpers = original[original.index("__device__ __forceinline__ uint64_t make_desc"):
        original.index("__device__ __forceinline__ void wgmma_ss_n64_kmajor")]
    pieces = [helpers, "template<int N> struct MatrixRequest;\n"]
    for width in (8, 16, 32, 64, 96, 128):
        count = width // 2
        registers = ", ".join(f"%{i}" for i in range(count))
        constraints = ", ".join(f'"+f"(d[{i}])' for i in range(count))
        pieces.append(
            f"template<> struct MatrixRequest<{width}> {{\n"
            f"__device__ __forceinline__ static void issue(float (&d)[{count}], "
            "uint64_t a, uint64_t b, int accumulate) {\n"
            'asm volatile("{\\n.reg .pred p;\\n'
            f'setp.ne.b32 p, %{count + 2}, 0;\\n"\n'
            f'"wgmma.mma_async.sync.aligned.m64n{width}k16.f32.f16.f16 '
            f'{{{registers}}}, %{count}, %{count + 1}, p, 1, 1, 0, 0;\\n}}\\n"\n'
            f': {constraints} : "l"(a), "l"(b), "r"(accumulate));\n'
            "}};\n")
    output.write_text("".join(pieces))


if __name__ == "__main__":
    generate(Path(sys.argv[1]), Path(sys.argv[2]))
