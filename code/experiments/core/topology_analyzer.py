"""
Topology analyzer for social graph metrics and performativity measurement.

Computes: clustering coefficient, inter-cluster bridges, modularity,
average path length, Performativity Index (PI), and Topological
Performativity Metric (TPM) from Section 7.5 of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

try:
    import community as community_louvain
except ImportError:
    community_louvain = None


@dataclass
class TopologyMetrics:
    num_nodes: int
    num_edges: int
    clustering_coefficient: float
    inter_cluster_bridges: int
    modularity: float
    average_path_length: float
    num_communities: int
    density: float


@dataclass
class PerformativityResult:
    pi: float  # Performativity Index
    novel_connections: int
    organic_recurrences: int
    random_baseline_pi: float


class TopologyAnalyzer:
    """Analyzes social graph topology and computes performativity metrics."""

    def __init__(self, social_graph: nx.Graph):
        self.graph = social_graph
        self._partition: Optional[Dict] = None

    def _get_partition(self) -> Dict:
        """Louvain community partition (cached)."""
        if self._partition is None:
            if community_louvain is None:
                raise ImportError("python-louvain required: pip install python-louvain")
            if self.graph.number_of_nodes() == 0:
                self._partition = {}
            else:
                self._partition = community_louvain.best_partition(self.graph)
        return self._partition

    def clustering_coefficient(self) -> float:
        """Average clustering coefficient."""
        if self.graph.number_of_nodes() < 3:
            return 0.0
        return nx.average_clustering(self.graph)

    def inter_cluster_bridges(self) -> int:
        """Count edges connecting different Louvain communities."""
        partition = self._get_partition()
        if not partition:
            return 0
        count = 0
        for u, v in self.graph.edges():
            if partition.get(u, -1) != partition.get(v, -2):
                count += 1
        return count

    def modularity(self) -> float:
        """Modularity of the Louvain partition."""
        partition = self._get_partition()
        if not partition:
            return 0.0
        communities = {}
        for node, comm in partition.items():
            communities.setdefault(comm, set()).add(node)
        community_list = list(communities.values())
        if len(community_list) < 2:
            return 0.0
        return nx.community.modularity(self.graph, community_list)

    def average_path_length(self) -> float:
        """Average shortest path length on the largest connected component."""
        if self.graph.number_of_nodes() < 2:
            return 0.0
        if nx.is_connected(self.graph):
            return nx.average_shortest_path_length(self.graph)
        # Use largest connected component
        largest_cc = max(nx.connected_components(self.graph), key=len)
        subgraph = self.graph.subgraph(largest_cc)
        if subgraph.number_of_nodes() < 2:
            return 0.0
        return nx.average_shortest_path_length(subgraph)

    def num_communities(self) -> int:
        """Number of Louvain communities detected."""
        partition = self._get_partition()
        if not partition:
            return 0
        return len(set(partition.values()))

    def density(self) -> float:
        """Graph density."""
        return nx.density(self.graph)

    def get_metrics(self) -> TopologyMetrics:
        """Compute all topology metrics."""
        return TopologyMetrics(
            num_nodes=self.graph.number_of_nodes(),
            num_edges=self.graph.number_of_edges(),
            clustering_coefficient=self.clustering_coefficient(),
            inter_cluster_bridges=self.inter_cluster_bridges(),
            modularity=self.modularity(),
            average_path_length=self.average_path_length(),
            num_communities=self.num_communities(),
            density=self.density(),
        )

    def performativity_index(
        self,
        novel_connections: List[Tuple[str, str, int]],
        organic_transactions: List[Tuple[str, str, int]],
        horizon: int = 10,
    ) -> PerformativityResult:
        """
        Performativity Index: fraction of algorithmically novel connections
        that recur organically in subsequent cycles.

        For each pair (i,j) connected by optimizer in cycle t with no prior history:
        track if they transact in t+1..t+k WITHOUT optimizer mediation.
        PI = organic_recurrences / novel_connections

        Args:
            novel_connections: [(member_a, member_b, cycle_introduced)]
            organic_transactions: [(member_a, member_b, cycle)] - unmediated repeat transactions
            horizon: number of cycles to look ahead (k)
        """
        if not novel_connections:
            return PerformativityResult(0.0, 0, 0, 0.0)

        # Build lookup of organic transactions
        organic_set: Dict[Tuple[str, str], Set[int]] = {}
        for a, b, cycle in organic_transactions:
            key = (min(a, b), max(a, b))
            organic_set.setdefault(key, set()).add(cycle)

        recurrences = 0
        total_novel = len(novel_connections)

        for a, b, intro_cycle in novel_connections:
            key = (min(a, b), max(a, b))
            cycles = organic_set.get(key, set())
            # Check if any organic transaction in [intro_cycle+1, intro_cycle+horizon]
            if any(intro_cycle < c <= intro_cycle + horizon for c in cycles):
                recurrences += 1

        pi = recurrences / max(1, total_novel)

        # Random baseline: expected recurrence if connections were random
        total_possible_pairs = max(1, self.graph.number_of_nodes() * (self.graph.number_of_nodes() - 1) // 2)
        total_organic = len(organic_transactions)
        random_pi = total_organic / max(1, total_possible_pairs * horizon)

        return PerformativityResult(
            pi=pi,
            novel_connections=total_novel,
            organic_recurrences=recurrences,
            random_baseline_pi=random_pi,
        )

    @staticmethod
    def topological_performativity_metric(
        graph_a: nx.Graph,
        graph_b: nx.Graph,
    ) -> Dict[str, float]:
        """
        Compare inter-cluster bridges between two graphs (e.g., QUBO vs greedy).
        Returns ratio and difference metrics.
        """
        analyzer_a = TopologyAnalyzer(graph_a)
        analyzer_b = TopologyAnalyzer(graph_b)

        bridges_a = analyzer_a.inter_cluster_bridges()
        bridges_b = analyzer_b.inter_cluster_bridges()

        return {
            "bridges_a": bridges_a,
            "bridges_b": bridges_b,
            "ratio": bridges_a / max(1, bridges_b),
            "difference": bridges_a - bridges_b,
            "clustering_a": analyzer_a.clustering_coefficient(),
            "clustering_b": analyzer_b.clustering_coefficient(),
            "modularity_a": analyzer_a.modularity(),
            "modularity_b": analyzer_b.modularity(),
            "communities_a": analyzer_a.num_communities(),
            "communities_b": analyzer_b.num_communities(),
        }
