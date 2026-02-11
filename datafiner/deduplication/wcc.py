"""Weakly connected component helper for dedup graph clustering.

This module provides a lightweight union-find fallback used by MinHash without
requiring distributed graph dependencies.
It keeps representative selection deterministic for stable dedup outputs across
runs.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def weakly_connected_component(edges: Iterable[Tuple[int, int]], num_partitions: int = 1):
    """
    Compute connected components with a union-find algorithm.

    Args:
        edges: Iterable of undirected graph edges `(src, dst)`.
        num_partitions: Compatibility argument retained for API parity.

    Returns:
        List of `(node, component_root)` pairs.

    Side effects:
        None.

    Assumptions:
        Component representative is the smallest root seen by union policy.
    """

    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        """Find canonical parent with path compression.

        Args:
            x: Node id.

        Returns:
            Root representative id.

        Side effects:
            Mutates `parent` for path compression.

        Assumptions:
            Unknown nodes initialize as self-parent roots.
        """
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        """Union two sets by choosing smaller representative as root.

        Args:
            a: First node id.
            b: Second node id.

        Returns:
            None.

        Side effects:
            Mutates `parent` mapping.

        Assumptions:
            Deterministic smallest-root policy keeps stable component ids.
        """
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
