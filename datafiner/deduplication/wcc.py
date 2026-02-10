from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def weakly_connected_component(edges: Iterable[Tuple[int, int]], num_partitions: int = 1):
    """
    Compatibility shim: returns connected component ids using a union-find algorithm.
    """

    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for src, dst in edges:
        union(int(src), int(dst))

    components = []
    for node in list(parent.keys()):
        components.append((node, find(node)))

    return components
