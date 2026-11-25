"""Independent NVIDIA H200 launch selection.

Every H200 launch request resolves within this package. A source/runtime or
compiler domain which is not bound remains an explicit unsupported request;
no selector or collective rules are projected from another hardware package.
"""

from .collective_rules import COLLECTIVE_RULES
