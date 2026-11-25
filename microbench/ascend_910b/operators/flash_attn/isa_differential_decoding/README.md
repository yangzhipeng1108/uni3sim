# 910B AI-core ISA differential decoding

CANN 8.5.1 ships no AI-core instruction decoder. `llvm-objdump` registers the `hiipu64` target but prints `<not available>` for every instruction under every `--mcpu` tried (`dav-c220-cube`, `dav-c220-vec`, `dav-c220`, `dav-m200`, `dav-c100`); `msobjdump` only inspects fatbin/ELF structure; `msdebug` (lldb) resolves addresses but returns empty mnemonics; and `ccec -S` is rejected with `unsupported option '-S'`. This directory recovers specific opcode classes without a decoder, so statements about a shipped kernel's instruction mix can be made from ISA static semantics, which is an allowed zero-fit source.

## Method

Compile an AscendC probe twice with `ccec`, identical except for an unrolled repetition count `NREP`, then diff the `.text` word histograms. Instructions are a fixed four bytes and little-endian. The opcode class whose occurrence count changes by exactly `ΔNREP` is the operation under test; register fields occupy the low bits, so aggregate on a high-bit prefix rather than the full word.

`ccec` needs the same driver flags CMake uses, recoverable from any built microbench's `CMakeFiles/*/flags.make`. The essential ones beyond the include paths are `--cce-aicore-arch=dav-c220-cube --cce-aicore-only --cce-aicore-lang -std=c++17`; without `--cce-aicore-lang` the toolchain rejects `__gm__` and the AscendC headers fail to parse.

`build.sh` compiles one probe at one `NREP`. `decode.py` dumps a section, converts to little-endian words, and reports per-prefix count deltas between two objects. `count_in_kernel.py` counts and locates an identified class inside a named function of a shipped kernel object.

## Classes identified so far (CANN 8.5.1, dav-c220-cube, -O3)

| Operation | Instructions per iteration | Class |
|---|---:|---|
| `GlobalTensor<int64_t>::GetValue`, dependent index | 1 | full word `01c41100`; family = top 10 bits equal to `7` (top-12 prefix `01c`/`01d`/`01e`, register in low bits) |
| `GlobalTensor<int32_t>::GetValue` plus index arithmetic | 4 | top-12 prefixes `414`, `028`, `038`, plus `000` for the accumulate |

Both were confirmed by a clean `+16` delta at `ΔNREP = 16`.

## Scope and limits

A class is identified from what `ccec` emits for that construct at this optimisation level and architecture. Other addressing forms, widths or intrinsics may use encodings not yet identified, so a count of identified classes is a lower bound on the operation's true frequency. The positional distribution of a class inside a function is the more robust output, because it does not depend on having enumerated every encoding.

This directory decodes instruction classes only. It does not produce timing, and nothing here is a Forward parameter.
