"""Execute the genuine save-partial program on payload-free lane metadata.

No Torch/Triton import, numeric tensor, chosen position/slot value, physical
address or production formula is needed. Unknown value-dependent branches
and unsupported source operations fail rather than choosing a scenario.
"""

class _Index:
    def __init__(self, ledger):
        self.ledger = ledger

    def operation(self, name, constant):
        if type(constant) is not int or constant < 0 or name in ("floor_divide", "remainder") and constant == 0:
            raise ValueError("source address operation requires a nonnegative constant and positive divisor")
        self.ledger.integers.append(name)
        return _Index(self.ledger)

    def __mul__(self, value):
        return self.operation("multiply", value)

    def __floordiv__(self, value):
        return self.operation("floor_divide", value)

    def __mod__(self, value):
        return self.operation("remainder", value)

    def __lt__(self, value):
        if type(value) is not int or value != 0:
            raise ValueError("source branch needs an undeclared integer value")
        self.ledger.integers.append("compare_nonnegative_with_zero")
        return False  # Proven by the declared all-valid, nonnegative domain.

    def __bool__(self):
        raise ValueError("source branch needs an undeclared integer value")

    def __eq__(self, other):
        raise ValueError("source branch needs an undeclared integer value")

    __ne__ = __eq__
    __gt__ = __eq__
    __ge__ = __eq__
    __le__ = __eq__


class _Lanes:
    def __init__(self, width):
        self.width = width

    def __lt__(self, bound):
        if type(bound) is not int or bound <= 0:
            raise ValueError("source lane mask requires a positive declared bound")
        return _Lanes(min(self.width, bound))

    def __bool__(self):
        raise ValueError("source vector mask cannot select a scalar branch")

    def __eq__(self, other):
        raise ValueError("source lane comparison is outside the supported instruction subset")

    __ne__ = __eq__


class _Pointer:
    def __init__(self, ledger, role, dtype, element_bytes, *, lanes=None, column=0):
        self.ledger, self.role, self.dtype, self.element_bytes = ledger, role, dtype, element_bytes
        self.lanes, self.column = lanes, column

    def __add__(self, offset):
        lanes, column = self.lanes, self.column
        if isinstance(offset, _Index):
            if offset.ledger is not self.ledger:
                raise ValueError("source address belongs to another program")
        elif isinstance(offset, _Lanes):
            if lanes is not None:
                raise ValueError("source metadata port does not support multidimensional lane addresses")
            lanes = offset.width
        elif type(offset) is int:
            column += offset
        else:
            raise ValueError("source pointer uses an unsupported or data-dependent address")
        return _Pointer(self.ledger, self.role, self.dtype, self.element_bytes, lanes=lanes, column=column)


class _Values:
    def __init__(self, ledger, count, dtype, expression):
        self.ledger, self.count, self.dtype, self.expression = ledger, count, dtype, expression

    def __add__(self, other):
        if (not isinstance(other, _Values) or other.ledger is not self.ledger
                or other.count != self.count or other.dtype != self.dtype or self.dtype != "fp32"):
            raise ValueError("source add requires matching FP32 lane metadata")
        self.ledger.additions += self.count
        return _Values(self.ledger, self.count, self.dtype, ("add", self.expression, other.expression))

    def __bool__(self):
        raise ValueError("source branch needs undeclared payload values")

    def __eq__(self, other):
        raise ValueError("source comparison needs undeclared payload values")

    __ne__ = __eq__


class _Program:
    def __init__(self):
        self.reads, self.writes, self.integers, self.additions = [], [], [], 0


class _TL:
    """The small instruction subset actually used by SavePartialStatesKernel."""
    def __init__(self, program):
        self.program = program

    def program_id(self, axis):
        if axis != 0:
            raise ValueError("save-partial reference requires a one-dimensional token grid")
        return _Index(self.program)

    def arange(self, start, end):
        if start != 0 or type(end) is not int or end <= 0:
            raise ValueError("source lane range must have explicit positive width")
        return _Lanes(end)

    def count(self, pointer, mask):
        if not isinstance(pointer, _Pointer) or pointer.ledger is not self.program:
            raise ValueError("source memory access must use this program's typed pointer")
        if pointer.lanes is None:
            if mask is not None:
                raise ValueError("source scalar access cannot use a vector mask")
            return 1
        count = pointer.lanes if mask is None else mask.width
        if not 0 < count <= pointer.lanes:
            raise ValueError("source memory mask exceeds its declared lanes")
        return count

    def load(self, pointer, mask=None):
        count = self.count(pointer, mask)
        self.program.reads.append({"role": pointer.role, "dtype": pointer.dtype,
            "elements": count, "bytes": count * pointer.element_bytes})
        if pointer.dtype == "int64" and pointer.lanes is None:
            return _Index(self.program)
        if pointer.dtype != "fp32":
            raise ValueError("save-partial payload must remain FP32")
        return _Values(self.program, count, pointer.dtype, pointer.role)

    def store(self, pointer, values, mask=None):
        count = self.count(pointer, mask)
        if (not isinstance(values, _Values) or values.ledger is not self.program
                or values.dtype != pointer.dtype or values.count != count):
            raise ValueError("source store layout/dtype differs from its values")
        self.program.writes.append({"role": pointer.role, "dtype": pointer.dtype,
            "column_offset_elements": pointer.column, "elements": count,
            "bytes": count * pointer.element_bytes, "value_expression": values.expression})


class PartialStateSource:
    """Source-derived layout, lane work and requested payload per active row."""
    def __init__(self, source):
        self.source = source
        self.wrapper = source.forward("SavePartialStatesKernel", "__call__", metadata_globals={
            "next_power_of_2": lambda size: 1 << (size - 1).bit_length()})

    def analyze(self, arguments):
        if (any(arguments[name].dtype != "fp32" for name in ("kv", "score", "ape", "state_cache"))
                or arguments["positions"].dtype != "int64"):
            raise ValueError("save-partial abstract program requires FP32 payload/state and INT64 positions")
        grid, geometry = self.wrapper(None, **arguments)
        rows, = grid
        program = _Program()
        pointers = {role: _Pointer(program, role, "int64" if role in ("positions", "slots") else "fp32",
            8 if role in ("positions", "slots") else 4) for role in ("kv", "score", "ape", "positions", "state", "slots")}
        kernel = self.source.forward("SavePartialStatesKernel", "kernel", metadata_globals={"tl": _TL(program)})
        kernel(kv_ptr=pointers["kv"], kv_stride=geometry["kv_stride"],
            score_ptr=pointers["score"], score_stride=geometry["score_stride"],
            ape_ptr=pointers["ape"], ape_stride=geometry["ape_stride"],
            positions_ptr=pointers["positions"], state_cache_ptr=pointers["state"],
            state_cache_stride0=geometry["state_cache_stride0"], state_cache_stride1=geometry["state_cache_stride1"],
            slot_mapping_ptr=pointers["slots"], block_size=geometry["block_size"],
            HEAD_SIZE=geometry["HEAD_SIZE"], TRITON_BLOCK_SIZE=geometry["TRITON_BLOCK_SIZE"],
            STATE_WIDTH=geometry["STATE_WIDTH"], COMPRESS_RATIO=geometry["COMPRESS_RATIO"])
        views = [{"role": role, "dtype": arguments[role].dtype, "shape": list(arguments[role].shape),
            "strides_elements": list(arguments[role].strides), "offset_elements": arguments[role].storage_offset,
            "allocates": False} for role in ("kv", "score")]
        writes = []
        for write in program.writes:
            if write["value_expression"] == "kv":
                role = "kv_state"
            elif write["value_expression"] == ("add", "score", "ape"):
                role = "score_state"
            else:
                raise ValueError("source state value expression is outside the supported save-partial contract")
            writes.append({"role": role, "dtype": write["dtype"],
                "record_offset_elements": write["column_offset_elements"], "elements_per_row": write["elements"]})
        return {"split_views": views, "writes": writes, "source_read_bytes": rows * sum(v["bytes"] for v in program.reads),
            "source_write_bytes": rows * sum(v["bytes"] for v in program.writes),
            "score_ape_fp32_additions": rows * program.additions,
            "source_read_roles_per_row": program.reads, "source_write_roles_per_row": program.writes,
            "source_integer_operations_per_row": program.integers, "source_geometry": geometry,
            "operand_layouts": {role: {"shape": tuple(arguments[role].shape), "dtype": arguments[role].dtype,
                "strides_elements": arguments[role].strides, "offset_elements": arguments[role].storage_offset}
                for role in ("kv", "score", "state_cache")},
            "condition": "all active slots valid/nonnegative, INT64 positions/slots and FP32 projected KV/score/APE/state; no physical cache address or reuse binding"}
