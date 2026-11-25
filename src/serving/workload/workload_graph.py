"""Logical workload DAG containing compute, communication, and transfers."""

from src.kernel.prediction_cache import runtime_lru_cache

from collections import deque
from os.path import commonprefix

from dataclasses import dataclass, field, replace
import weakref
from types import FunctionType
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Tuple


class WorkloadKind(str, Enum):
    LINEAR = "linear"
    LM_HEAD = "lm_head"
    ROUTER = "router"
    FLASH_ATTN = "flash_attn"
    EAGER_ATTN = "eager_attn"
    FUSED_MOE = "fused_moe"
    NORM = "norm"
    ACTIVATION = "activation"
    ROTARY = "rotary"
    ELEMENTWISE = "elementwise"
    REDUCE = "reduce"
    GATHER = "gather"
    VECTOR = "vector"
    KV_UPDATE = "kv_update"
    COLLECTIVE = "collective"
    P2P = "p2p"
    PD_KV_TRANSFER = "pd_kv_transfer"
    AF_ACTIVATION_TRANSFER = "af_activation_transfer"
    MEMCPY = "memcpy"
    KV_OFFLOAD = "kv_offload"


@dataclass(frozen=True)
class WorkloadNode:
    node_id: str
    kind: WorkloadKind
    pool_name: str
    shape: Mapping[str, int]
    dtype: Optional[str] = None
    variant: Optional[str] = None
    rank: Optional[int] = None
    attributes: Mapping[str, object] = None

    def with_attributes(self, attributes: Mapping[str, object], *, node_id: str | None = None):
        """Derive annotations while sharing already validated immutable geometry."""
        if type(self) is not WorkloadNode:
            raise TypeError("annotation derivation requires a canonical WorkloadNode")
        if not isinstance(attributes, Mapping):
            raise TypeError("workload attributes must be a mapping")
        identity = self.node_id if node_id is None else node_id
        if not isinstance(identity, str) or not identity:
            raise ValueError("workload node_id must be a non-empty string")
        derived = object.__new__(WorkloadNode)
        derived.__dict__.update(self.__dict__)
        object.__setattr__(derived, "node_id", identity)
        object.__setattr__(derived, "attributes", MappingProxyType(dict(attributes)))
        if "_field_update_capsule" in derived.__dict__:
            del derived.__dict__["_field_update_capsule"]
        return derived

    def with_updates(self, shape_updates, attribute_updates, *, node_id=None, compose=False):
        """Derive owned fields and record their actual immutable write facts."""
        if type(compose) is not bool:
            raise TypeError("source write composition must be an explicit boolean")
        native = (type(self) is WorkloadNode and type(shape_updates) is dict
            and type(attribute_updates) is dict
            and all(type(key) is str and key for key in (*shape_updates, *attribute_updates))
            and all(type(value) is int and value >= 0 for value in shape_updates.values())
            and type(self.shape) is MappingProxyType and type(self.attributes) is MappingProxyType
            and all(type(key) is str and type(value) is int for key, value in self.shape.items())
            and all(type(key) is str for key in self.attributes))
        identity = self.node_id if node_id is None else node_id
        if not isinstance(identity, str) or not identity:
            raise ValueError("workload node_id must be a non-empty string")
        if not native:
            shape = dict(self.shape); shape.update(shape_updates)
            attributes = dict(self.attributes); attributes.update(attribute_updates)
            return replace(self, node_id=identity, shape=shape, attributes=attributes)
        shape = self.shape
        if shape_updates:
            values = dict(self.shape); values.update(shape_updates)
            shape = MappingProxyType(values)
        attributes = dict(self.attributes); attributes.update(attribute_updates)
        derived = object.__new__(WorkloadNode)
        derived.__dict__.update(self.__dict__)
        object.__setattr__(derived, "node_id", identity)
        object.__setattr__(derived, "shape", shape)
        object.__setattr__(derived, "attributes", MappingProxyType(attributes))
        # Keep the original recorded node alive only through its source owner.
        # Chained annotation passes must not lose their write facts when an
        # intermediate derived node is released.
        # Complete source bindings retain their immediate template parent.
        # Only explicit repeated-body/annotation composition flattens writes.
        lineage = _workload_node_field_updates_read(self) if compose else None
        parent, shapes, attrs = ((self, (), ()) if lineage is None else lineage)
        capsule = (_NODE_FIELD_UPDATE_MARKER, weakref.ref(parent), weakref.ref(derived),
            tuple(dict.fromkeys((*shapes, *shape_updates))),
            tuple(dict.fromkeys((*attrs, *attribute_updates))), shape, derived.attributes,
            (parent.kind, parent.pool_name, parent.dtype, parent.variant, parent.rank,
                parent.shape, parent.attributes))
        object.__setattr__(derived, "_field_update_capsule", capsule)
        return derived

    def __post_init__(self) -> None:
        for name in ("node_id", "pool_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"workload {name} must be a non-empty string")
        object.__setattr__(self, "kind", WorkloadKind(self.kind))
        if not isinstance(self.shape, Mapping):
            raise TypeError("workload shape must be a mapping")
        invalid_shape_keys = [
            name for name in self.shape if not isinstance(name, str) or not name
        ]
        if invalid_shape_keys:
            raise TypeError(
                "workload shape keys must be non-empty strings: "
                f"{invalid_shape_keys}")
        invalid_shape_types = {
            name: value for name, value in self.shape.items()
            if isinstance(value, bool) or not isinstance(value, int)
        }
        if invalid_shape_types:
            raise TypeError(
                f"workload shape dimensions must be integers: "
                f"{invalid_shape_types}")
        normalized_shape = dict(self.shape)
        if any(value < 0 for value in normalized_shape.values()):
            raise ValueError("workload shape dimensions cannot be negative")
        object.__setattr__(self, "shape", MappingProxyType(normalized_shape))
        for name in ("dtype", "variant"):
            value = getattr(self, name)
            if value is not None and (
                    not isinstance(value, str) or not value.strip()):
                raise ValueError(
                    f"workload {name} must be a non-empty string or null")
        if self.rank is not None:
            if isinstance(self.rank, bool) or not isinstance(self.rank, int):
                raise TypeError("workload rank must be an integer or null")
            if self.rank < 0:
                raise ValueError("workload rank cannot be negative")
        if self.attributes is not None and not isinstance(
                self.attributes, Mapping):
            raise TypeError("workload attributes must be a mapping or null")
        attributes = {} if self.attributes is None else dict(self.attributes)
        object.__setattr__(self, "attributes", MappingProxyType(attributes))


_NODE_FIELD_UPDATE_MARKER = object()


def _node_update_source_unchanged():
    return (WorkloadNode is _NODE_UPDATE_CANONICAL_TYPE
        and tuple(vars(WorkloadNode)) == _NODE_UPDATE_CLASS_KEYS
        and all(vars(WorkloadNode)[name] is member for name, member in _NODE_UPDATE_CLASS_MEMBERS)
        and all(globals()[name] is function and function.__code__ is code
            and function.__defaults__ == defaults and function.__kwdefaults__ == kwdefaults
            for name, function, code, defaults, kwdefaults in _NODE_UPDATE_FUNCTION_SOURCE)
        and all(function.__code__ is code and function.__defaults__ == defaults
            and function.__kwdefaults__ == kwdefaults for function, code, defaults, kwdefaults in _NODE_UPDATE_METHOD_SOURCE))


def _workload_node_field_updates_read(node):
    """Read an owned capsule in an already source-checked synchronous batch."""
    if type(node) is not WorkloadNode or "_field_update_capsule" not in node.__dict__:
        return None
    capsule = node.__dict__["_field_update_capsule"]
    if type(capsule) is not tuple or len(capsule) != 8 or capsule[0] is not _NODE_FIELD_UPDATE_MARKER:
        return None
    _, parent_reference, node_reference, shapes, attributes, shape, attrs, signature = capsule
    if (type(parent_reference) is not weakref.ReferenceType or type(node_reference) is not weakref.ReferenceType
            or type(shapes) is not tuple or type(attributes) is not tuple
            or node_reference() is not node or node.shape is not shape or node.attributes is not attrs):
        return None
    parent = parent_reference()
    if (type(parent) is not WorkloadNode
            or any(getattr(current, name) is not expected for current in (node, parent)
                for name, expected in zip(("kind", "pool_name", "dtype", "variant", "rank"), signature[:5]))
            or parent.shape is not signature[5] or parent.attributes is not signature[6]):
        return None
    return parent, shapes, attributes


def _workload_node_field_updates(node):
    """Return exact owner-issued field writes under the frozen-node contract."""
    if not _node_update_source_unchanged():return None
    return _workload_node_field_updates_read(node)


_NODE_UPDATE_CANONICAL_TYPE = WorkloadNode
_NODE_UPDATE_CLASS_KEYS = tuple(vars(WorkloadNode))
_NODE_UPDATE_CLASS_MEMBERS = tuple(vars(WorkloadNode).items())
_NODE_UPDATE_FUNCTION_SOURCE = tuple((name, globals()[name], globals()[name].__code__,
    globals()[name].__defaults__, globals()[name].__kwdefaults__) for name in (
        "_node_update_source_unchanged", "_workload_node_field_updates", "_workload_node_field_updates_read"))
_NODE_UPDATE_METHOD_SOURCE = tuple((member, member.__code__, member.__defaults__,
    None if member.__kwdefaults__ is None else dict(member.__kwdefaults__))
    for member in vars(WorkloadNode).values() if type(member) is FunctionType)


@dataclass(frozen=True)
class WorkloadEdge:
    source: str
    destination: str
    kind: str = "data"

    def __post_init__(self) -> None:
        for name in ("source", "destination", "kind"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"workload edge {name} must be a non-empty string")
        if self.kind not in ("data", "barrier", "ready", "completion"):
            raise ValueError(f"unknown workload edge kind: {self.kind!r}")


@dataclass(frozen=True)
class WorkloadBlockRepeat:
    """One materialized block template plus exact recurrence edges."""

    block_id: str
    node_ids: Tuple[str, ...]
    repeat_count: int
    feedback_edges: Tuple[WorkloadEdge, ...]
    trace_node_ids: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id:
            raise ValueError("workload block repeat ID must be non-empty")
        if not isinstance(self.node_ids, (list, tuple)):
            raise TypeError("workload block repeat node IDs must be a sequence")
        node_ids = tuple(self.node_ids)
        if (not node_ids
                or any(not isinstance(node_id, str) or not node_id
                       for node_id in node_ids)):
            raise ValueError(
                "workload block repeat node IDs must be non-empty strings")
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("workload block repeat node IDs must be unique")
        if (isinstance(self.repeat_count, bool)
                or not isinstance(self.repeat_count, int)
                or self.repeat_count < 2):
            raise ValueError(
                "workload block repeat count must be an integer >= 2")
        if not isinstance(self.feedback_edges, (list, tuple)):
            raise TypeError(
                "workload block repeat feedback edges must be a sequence")
        feedback_edges = tuple(self.feedback_edges)
        if any(not isinstance(edge, WorkloadEdge) for edge in feedback_edges):
            raise TypeError(
                "workload block repeat feedback requires WorkloadEdge objects")
        known = set(node_ids)
        if any(
                edge.source not in known or edge.destination not in known
                for edge in feedback_edges):
            raise ValueError(
                "workload block feedback edges must stay inside the template")
        if not isinstance(self.trace_node_ids, (list, tuple)):
            raise TypeError(
                "workload block repeat trace node IDs must be a sequence")
        trace_node_ids = tuple(
            (template_id, tuple(logical_ids))
            for template_id, logical_ids in self.trace_node_ids
        )
        template_ids = []
        logical_ids = []
        for template_id, occurrences in trace_node_ids:
            if template_id not in known:
                raise ValueError(
                    "workload block trace node is outside the template")
            if (len(occurrences) != self.repeat_count
                    or occurrences[0] != template_id
                    or any(not isinstance(node_id, str) or not node_id
                           for node_id in occurrences)):
                raise ValueError(
                    "workload block trace node occurrences must cover every "
                    "logical repeat")
            template_ids.append(template_id)
            logical_ids.extend(occurrences)
        if len(template_ids) != len(set(template_ids)):
            raise ValueError(
                "workload block trace template node IDs must be unique")
        if len(logical_ids) != len(set(logical_ids)):
            raise ValueError(
                "workload block logical trace node IDs must be unique")
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "feedback_edges", feedback_edges)
        object.__setattr__(self, "trace_node_ids", trace_node_ids)


@dataclass(frozen=True)
class WorkloadGraph:
    nodes: Tuple[WorkloadNode, ...]
    edges: Tuple[WorkloadEdge, ...]
    block_repeats: Tuple[WorkloadBlockRepeat, ...]
    _is_node_binding: bool = field(init=False, repr=False, compare=False, default=False)
    _nodes_by_id: Mapping[str, WorkloadNode] = field(
        init=False, repr=False, compare=False)
    _predecessors_by_id: Mapping[str, Tuple[str, ...]] = field(
        init=False, repr=False, compare=False)
    _topological_node_ids: Tuple[str, ...] = field(
        init=False, repr=False, compare=False)

    def __init__(self, nodes: Iterable[WorkloadNode],
                 edges: Iterable[WorkloadEdge] = (),
                 block_repeats: Iterable[WorkloadBlockRepeat] = ()):
        object.__setattr__(self, "_is_node_binding", False)
        node_tuple = tuple(nodes)
        edge_tuple = tuple(edges)
        block_repeat_tuple = tuple(block_repeats)
        if not node_tuple:
            raise ValueError("WorkloadGraph requires at least one node")
        ids = [node.node_id for node in node_tuple]
        if len(ids) != len(set(ids)):
            raise ValueError("workload node IDs must be unique")
        known = set(ids)
        for edge in edge_tuple:
            if edge.source not in known or edge.destination not in known:
                raise ValueError(f"workload edge references an unknown node: {edge}")
            if edge.source == edge.destination:
                raise ValueError("workload self-edges are not allowed")
        if any(
                not isinstance(repeat, WorkloadBlockRepeat)
                for repeat in block_repeat_tuple):
            raise TypeError(
                "WorkloadGraph block repeats require WorkloadBlockRepeat objects")
        block_ids = [repeat.block_id for repeat in block_repeat_tuple]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("workload block repeat IDs must be unique")
        repeated_node_ids = []
        for repeat in block_repeat_tuple:
            unknown = set(repeat.node_ids) - known
            if unknown:
                raise ValueError(
                    "workload block repeat references unknown nodes: "
                    f"{sorted(unknown)}")
            repeated_node_ids.extend(repeat.node_ids)
        if len(repeated_node_ids) != len(set(repeated_node_ids)):
            raise ValueError(
                "workload nodes cannot belong to multiple block repeats")
        predecessors = {node_id: [] for node_id in ids}
        for edge in edge_tuple:
            predecessors[edge.destination].append(edge.source)
        object.__setattr__(self, "nodes", node_tuple)
        object.__setattr__(self, "edges", edge_tuple)
        object.__setattr__(self, "block_repeats", block_repeat_tuple)
        object.__setattr__(
            self, "_nodes_by_id", MappingProxyType({
                node.node_id: node for node in node_tuple
            }))
        object.__setattr__(
            self, "_predecessors_by_id", MappingProxyType({
                node_id: tuple(source_ids)
                for node_id, source_ids in predecessors.items()
            }))
        object.__setattr__(
            self, "_topological_node_ids", self._compute_topological_order())

    def __getattribute__(self, name):
        if name in ("nodes", "edges", "block_repeats"):
            state = object.__getattribute__(self, "__dict__")
            if "_native_source_frame" in state:
                frame = state["_native_source_frame"]
                return frame.public_nodes() if name == "nodes" else frame.public_topology(name)
        return object.__getattribute__(self, name)

    @property
    def source_model_execution_version(self):
        """Optional validated caller context, separate from recorded nodes."""
        return self.__dict__["_source_model_execution_version"] if "_source_model_execution_version" in self.__dict__ else None

    def bind_model_execution_version(self, version):
        """Bind caller provenance without rewriting immutable source frames."""
        if not isinstance(version, str) or not version:
            raise ValueError("model source version must be explicit and nonempty")
        if self.source_model_execution_version is not None and self.source_model_execution_version != version:
            raise ValueError("workload graph source version disagrees with its actual model caller")
        if any("source_model_execution_version" in node.attributes
                and node.attributes["source_model_execution_version"] != version for node in self.nodes):
            raise ValueError("workload node source version disagrees with its actual model caller")
        object.__setattr__(self, "_source_model_execution_version", version)

    def with_nodes(self, nodes: Iterable[WorkloadNode]):
        """Bind validated nodes to the same complete, ordered dependency DAG."""
        if type(self) is not WorkloadGraph:
            raise TypeError("graph binding requires a canonical WorkloadGraph")
        nodes = tuple(nodes)
        if any(type(node) is not WorkloadNode for node in nodes):
            raise TypeError("graph binding requires canonical WorkloadNode objects")
        if (len(nodes) != len(self.nodes)
                or any(node.node_id != original.node_id
                    for node, original in zip(nodes, self.nodes))):
            raise ValueError("graph binding must preserve every node ID and its order")
        derived = object.__new__(WorkloadGraph)
        derived.__dict__.update(self.__dict__)
        object.__setattr__(derived, "edges", self.edges)
        object.__setattr__(derived, "block_repeats", self.block_repeats)
        if "_ep_source_binding" in derived.__dict__:
            del derived.__dict__["_ep_source_binding"]
        if "_native_source_frame" in derived.__dict__:
            del derived.__dict__["_native_source_frame"]
        object.__setattr__(derived, "nodes", nodes)
        object.__setattr__(derived, "_is_node_binding", True)
        object.__setattr__(derived, "_nodes_by_id", MappingProxyType({node.node_id: node for node in nodes}))
        return derived

    @property
    def logical_node_count(self) -> int:
        return len(self.nodes) + sum(
            (repeat.repeat_count - 1) * len(repeat.node_ids)
            for repeat in self.block_repeats
        )

    @property
    def logical_edge_count(self) -> int:
        repeated_edges = 0
        for repeat in self.block_repeats:
            node_ids = set(repeat.node_ids)
            internal_edge_count = sum(
                1 for edge in self.edges
                if edge.source in node_ids and edge.destination in node_ids
            )
            repeated_edges += (repeat.repeat_count - 1) * (
                internal_edge_count + len(repeat.feedback_edges))
        return len(self.edges) + repeated_edges

    def node(self, node_id: str) -> WorkloadNode:
        try:
            return self._nodes_by_id[node_id]
        except KeyError as exc:
            raise KeyError(node_id) from exc

    def predecessors(self, node_id: str) -> Tuple[str, ...]:
        if node_id in self._predecessors_by_id:
            return self._predecessors_by_id[node_id]
        return ()

    def topological_order(self) -> Tuple[str, ...]:
        return self._topological_node_ids

    def _compute_topological_order(self) -> Tuple[str, ...]:
        indegree = {node.node_id: 0 for node in self.nodes}
        successors = {node.node_id: [] for node in self.nodes}
        for edge in self.edges:
            indegree[edge.destination] += 1
            successors[edge.source].append(edge.destination)
        ready = deque(
            node.node_id for node in self.nodes if indegree[node.node_id] == 0)
        order = []
        while ready:
            current = ready.popleft()
            order.append(current)
            for successor in successors[current]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
        if len(order) != len(self.nodes):
            raise ValueError("WorkloadGraph must be acyclic")
        return tuple(order)


def _source_layer(node_id):
    labels = [part for part in node_id.split(":")
        if part.startswith("layer-") and part[6:].isascii() and part[6:].isdecimal()]
    return int(labels[0][6:]) if len(labels) == 1 else None


def _relative_layer_id(value, layer):
    parts = value.split(":")
    return ":".join("layer-" + str(int(part[6:]) - layer)
        if part.startswith("layer-") and part[6:].isascii() and part[6:].isdecimal()
        else part for part in parts)


_SOURCE_REPEAT_TOPOLOGY_CACHE_SIZE = 32


@runtime_lru_cache(maxsize=_SOURCE_REPEAT_TOPOLOGY_CACHE_SIZE)
def _source_repeat_structure(nodes, edges):
    """Compile only the exact relative DAG; physical contracts remain live."""
    labels = tuple(_source_layer(name) for name, _ in nodes)
    layers, incoming = {}, {}
    for index, label in enumerate(labels):
        if label is not None:
            if label not in layers:
                layers[label], incoming[label] = [], []
            layers[label].append(index)
    for source, destination, kind in edges:
        label = labels[destination]
        if label is not None:
            incoming[label].append((source, destination, kind))
    ordered = tuple(sorted(layers))
    return (labels, tuple((label, tuple(layers[label])) for label in ordered),
        tuple((label, (tuple((_relative_layer_id(nodes[index][0], label), nodes[index][1])
            for index in layers[label]), tuple(sorted((_relative_layer_id(nodes[source][0], label),
                _relative_layer_id(nodes[destination][0], label), kind)
                for source, destination, kind in incoming[label])))) for label in ordered))


@runtime_lru_cache(maxsize=_SOURCE_REPEAT_TOPOLOGY_CACHE_SIZE)
def _source_repeat_layout(nodes, edges, candidates):
    """Compile index-only recurrence layout for an exact DAG and live groups."""
    layer_labels, layer_entries, _ = _source_repeat_structure(nodes, edges)
    layers = dict(layer_entries)
    selected = []
    for group in candidates:
        members = set(group)
        for source, destination, _ in edges:
            left, right = layer_labels[source], layer_labels[destination]
            if left in members and ((right in members and right - left not in (0, 1))
                    or (right not in members and left != group[-1])):
                break
        else:
            selected.append(group)
    if not selected:
        return None

    owner, renamed, repeats = {}, {}, []
    edge_key = lambda edge: (nodes[edge[0]][0], nodes[edge[1]][0], edge[2])
    for number, group in enumerate(selected):
        template = layers[group[0]]
        for layer in group:
            for original, base in zip(layers[layer], template, strict=True):
                owner[original] = number
                renamed[original] = base
        feedback = set()
        for source, destination, kind in edges:
            left, right = layer_labels[source], layer_labels[destination]
            if left in group and right in group and right == left + 1:
                feedback.add((renamed[source], renamed[destination], kind))
        if not feedback:
            return None
        repeats.append((template, len(group), tuple(sorted(feedback, key=edge_key)),
            tuple((node, tuple(layers[layer][index] for layer in group))
                for index, node in enumerate(template))))

    retained = tuple(index for index in range(len(nodes))
        if index not in renamed or renamed[index] == index)
    compact_edges = set()
    for source, destination, kind in edges:
        source_group = owner[source] if source in owner else None
        destination_group = owner[destination] if destination in owner else None
        if source_group is not None and source_group == destination_group:
            if layer_labels[source] != layer_labels[destination] or renamed[source] != source:
                continue
        elif destination_group is not None and renamed[destination] != destination:
            continue
        source = renamed[source] if source in renamed else source
        destination = renamed[destination] if destination in renamed else destination
        compact_edges.add((source, destination, kind))
    return retained, tuple(sorted(compact_edges, key=edge_key)), tuple(repeats)


def compact_source_layer_repeats(graph: WorkloadGraph, node_contracts: Mapping) -> WorkloadGraph:
    """Compress equal live layer contracts using the exact source dependency DAG.

    The cached layout contains only relative node indices. Physical contracts
    are compared again on every invocation, and all source IDs, nodes, repeats,
    and logical edge counts are bound and checked against the current graph.
    """
    if graph.block_repeats or set(node_contracts) != {node.node_id for node in graph.nodes}:
        return graph
    names = tuple(node.node_id for node in graph.nodes)
    prefix = commonprefix(names)
    prefix = prefix[:prefix.rfind(":") + 1]
    if any(part.startswith("layer-") and part[6:].isascii() and part[6:].isdecimal()
            for part in prefix.split(":")):
        prefix = ""
    positions = {name: index for index, name in enumerate(names)}
    nodes = tuple((node.node_id[len(prefix):], node.rank) for node in graph.nodes)
    edges = tuple((positions[edge.source], positions[edge.destination], edge.kind) for edge in graph.edges)
    _, layer_entries, structure_entries = _source_repeat_structure(nodes, edges)
    if len(layer_entries) < 3:
        return graph
    ordered = tuple(label for label, _ in layer_entries)
    structures = dict(structure_entries)
    signatures = {layer: tuple(node_contracts[names[index]] for index in indices)
        for layer, indices in layer_entries}
    candidates, position = [], 0
    while position < len(ordered):
        start, stop = position, position + 1
        while (stop < len(ordered) and ordered[stop] == ordered[stop - 1] + 1
                and structures[ordered[stop]] == structures[ordered[start]]
                and signatures[ordered[stop]] == signatures[ordered[start]]):
            stop += 1
        if stop - start >= 2:
            candidates.append(tuple(ordered[start:stop]))
        position = stop
    layout = _source_repeat_layout(nodes, edges, tuple(candidates))
    if layout is None:
        return graph
    retained, compact_edges, repeat_layouts = layout
    repeats = tuple(WorkloadBlockRepeat(
        names[template[0]] + ":source-layer-repeat", tuple(names[index] for index in template),
        count, tuple(WorkloadEdge(names[source], names[destination], kind)
            for source, destination, kind in feedback),
        tuple((names[node], tuple(names[index] for index in occurrences))
            for node, occurrences in trace))
        for template, count, feedback, trace in repeat_layouts)
    compact = WorkloadGraph(tuple(graph.nodes[index] for index in retained),
        tuple(WorkloadEdge(names[source], names[destination], kind)
            for source, destination, kind in compact_edges), repeats)
    if (compact.logical_node_count != graph.logical_node_count
            or compact.logical_edge_count != graph.logical_edge_count):
        return graph
    return compact


class _SourceLayerCompaction:
    """One last validated source DAG layout; every physical contract stays live."""

    def __init__(self, graph, contracts, compact):
        self.predecessors = graph._predecessors_by_id
        from .traced_forward import _source_bound_topology
        self.edges = _source_bound_topology(graph)[0]
        self.ranks = tuple(node.rank for node in graph.nodes)
        layers = {}
        for node in graph.nodes:
            layer = _source_layer(node.node_id)
            if layer is not None:
                if layer not in layers:
                    layers[layer] = []
                layers[layer].append(node.node_id)
        self.layers = tuple(tuple(layers[layer]) for layer in sorted(layers))
        self.relations = self._relations(contracts)
        self.template = compact

    def _relations(self, contracts):
        signatures = tuple(tuple(contracts[name] for name in layer) for layer in self.layers)
        return tuple(left == right for left, right in zip(signatures, signatures[1:]))

    def bind(self, graph, contracts):
        from .traced_forward import _source_bound_topology
        if (type(graph) is not WorkloadGraph or graph.block_repeats
                or graph._predecessors_by_id is not self.predecessors
                or _source_bound_topology(graph)[0] is not self.edges
                or len(graph.nodes) != len(contracts)
                or any(node.rank != rank for node, rank in zip(graph.nodes, self.ranks, strict=True))
                or self._relations(contracts) != self.relations):
            return None
        # with_nodes preserves the complete verified compact topology, repeat
        # count and original logical trace IDs. Nodes are always current.
        return self.template.with_nodes(graph.node(node.node_id) for node in self.template.nodes)
