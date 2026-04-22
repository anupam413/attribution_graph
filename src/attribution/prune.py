"""
src/attribution/prune.py
-----------------------
Phase 3e: Indirect influence matrix computation and graph pruning.

The indirect influence matrix M answers:
  "If I change feature i at layer ℓ, how much does feature j at layer ℓ' change?"
  
This captures multi-hop paths through the graph. We compute it as:
  M = (I - W)^{-1}
  
where W is the direct influence (Jacobian) matrix.

Pruning strategies:
  1. Threshold-based: Remove edges with |weight| < threshold
  2. Top-k: Keep only top-k edges per node
  3. Indirect influence: Remove nodes with low total downstream influence
"""

from __future__ import annotations

import torch
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Optional

from src.attribution.graph import AttributionGraph


class IndirectInfluenceComputer:
    """
    Computes indirect influence matrix from attribution graph.
    
    Usage:
        computer = IndirectInfluenceComputer()
        M = computer.compute_influence_matrix(graph)
        
        # M[i, j] = total influence of node i on node j (direct + indirect)
    """
    
    def compute_influence_matrix(
        self, graph: AttributionGraph, damping: float = 0.01
    ) -> np.ndarray:
        """
        Compute indirect influence matrix M = (I - W)^{-1}.
        
        Args:
            graph: Attribution graph
            damping: Small value added to diagonal for numerical stability
            
        Returns:
            Influence matrix (n_nodes, n_nodes)
        """
        n_nodes = len(graph.nodes)
        
        if n_nodes == 0:
            return np.array([])
        
        # Build direct influence matrix W
        W = np.zeros((n_nodes, n_nodes))
        node_to_idx = {id(n): i for i, n in enumerate(graph.nodes)}
        
        for edge in graph.edges:
            i = node_to_idx[id(edge.from_node)]
            j = node_to_idx[id(edge.to_node)]
            W[j, i] = edge.weight  # W[j,i] = influence of i on j
        
        # Compute M = (I - W)^{-1}
        # Add damping to diagonal for stability
        I = np.eye(n_nodes)
        A = I - W + damping * I
        
        # Solve using sparse solver for efficiency
        try:
            A_sparse = csr_matrix(A)
            M = spsolve(A_sparse, I)
            
            if isinstance(M, np.matrix):
                M = np.asarray(M)
        except:
            # Fallback to dense solver
            M = np.linalg.solve(A, I)
        
        return M
    
    def compute_path_influence(
        self, 
        graph: AttributionGraph,
        source_nodes: list[int],
        target_nodes: list[int],
    ) -> dict:
        """
        Compute total influence from source nodes to target nodes.
        
        Args:
            graph: Attribution graph
            source_nodes: List of source node indices
            target_nodes: List of target node indices
            
        Returns:
            dict with influence scores
        """
        M = self.compute_influence_matrix(graph)
        
        if M.size == 0:
            return {'total_influence': 0.0}
        
        # Total influence from sources to targets
        influence = M[np.ix_(target_nodes, source_nodes)]
        
        return {
            'total_influence': float(np.sum(np.abs(influence))),
            'mean_influence': float(np.mean(np.abs(influence))),
            'max_influence': float(np.max(np.abs(influence))),
            'influence_matrix': influence,
        }


class GraphPruner:
    """
    Prune attribution graphs using various strategies.
    
    Usage:
        pruner = GraphPruner()
        
        # Threshold pruning
        pruned = pruner.prune_by_threshold(graph, threshold=0.05)
        
        # Top-k pruning
        pruned = pruner.prune_by_top_k(graph, k=10)
        
        # Indirect influence pruning
        pruned = pruner.prune_by_influence(graph, min_influence=0.1)
    """
    
    def prune_by_threshold(
        self, graph: AttributionGraph, threshold: float
    ) -> AttributionGraph:
        """Remove edges with |weight| < threshold."""
        return graph.filter_by_threshold(threshold)
    
    def prune_by_top_k(
        self, graph: AttributionGraph, k: int, per_layer: bool = False
    ) -> AttributionGraph:
        """
        Keep only top-k edges.
        
        Args:
            k: Number of edges to keep
            per_layer: If True, keep top-k per layer; else top-k globally
        """
        if per_layer:
            return self._prune_top_k_per_layer(graph, k)
        else:
            return self._prune_top_k_global(graph, k)
    
    def _prune_top_k_global(self, graph: AttributionGraph, k: int) -> AttributionGraph:
        """Keep top-k edges globally by absolute weight."""
        if len(graph.edges) == 0:
            return graph
        
        sorted_edges = sorted(graph.edges, key=lambda e: abs(e.weight), reverse=True)
        top_edges = sorted_edges[:min(k, len(sorted_edges))]
        
        # Keep nodes that have edges
        connected_nodes = set()
        for edge in top_edges:
            connected_nodes.add(id(edge.from_node))
            connected_nodes.add(id(edge.to_node))
        
        filtered_nodes = [n for n in graph.nodes if id(n) in connected_nodes]
        
        new_graph = AttributionGraph()
        new_graph.nodes = filtered_nodes
        new_graph.edges = top_edges
        return new_graph
    
    def _prune_top_k_per_layer(self, graph: AttributionGraph, k: int) -> AttributionGraph:
        """Keep top-k edges per layer."""
        # Group edges by layer
        edges_by_layer = {}
        for edge in graph.edges:
            layer = edge.from_node.layer
            if layer not in edges_by_layer:
                edges_by_layer[layer] = []
            edges_by_layer[layer].append(edge)
        
        # Keep top-k per layer
        pruned_edges = []
        for layer, edges in edges_by_layer.items():
            sorted_edges = sorted(edges, key=lambda e: abs(e.weight), reverse=True)
            pruned_edges.extend(sorted_edges[:k])
        
        # Rebuild graph
        connected_nodes = set()
        for edge in pruned_edges:
            connected_nodes.add(id(edge.from_node))
            connected_nodes.add(id(edge.to_node))
        
        filtered_nodes = [n for n in graph.nodes if id(n) in connected_nodes]
        
        new_graph = AttributionGraph()
        new_graph.nodes = filtered_nodes
        new_graph.edges = pruned_edges
        return new_graph
    
    def prune_by_influence(
        self,
        graph: AttributionGraph,
        min_influence: float,
        target_nodes: Optional[list[int]] = None,
    ) -> AttributionGraph:
        """
        Prune nodes by total indirect influence on target nodes.
        
        Args:
            graph: Attribution graph
            min_influence: Minimum total influence to keep a node
            target_nodes: List of node indices to measure influence on
                          (None = use all output nodes)
        """
        if len(graph.nodes) == 0:
            return graph
        
        computer = IndirectInfluenceComputer()
        M = computer.compute_influence_matrix(graph)
        
        if M.size == 0:
            return graph
        
        # Determine target nodes
        if target_nodes is None:
            # Use nodes from last layer as targets
            max_layer = max(n.layer for n in graph.nodes)
            target_nodes = [
                i for i, n in enumerate(graph.nodes) if n.layer == max_layer
            ]
        
        if len(target_nodes) == 0:
            return graph
        
        # Compute total influence of each node on targets
        influences = np.abs(M[target_nodes, :]).sum(axis=0)
        
        # Keep nodes above threshold
        keep_indices = np.where(influences >= min_influence)[0]
        kept_nodes = [graph.nodes[i] for i in keep_indices]
        
        # Keep edges between kept nodes
        kept_node_ids = {id(n) for n in kept_nodes}
        kept_edges = [
            e for e in graph.edges
            if id(e.from_node) in kept_node_ids and id(e.to_node) in kept_node_ids
        ]
        
        new_graph = AttributionGraph()
        new_graph.nodes = kept_nodes
        new_graph.edges = kept_edges
        return new_graph
    
    def prune_by_activation(
        self,
        graph: AttributionGraph,
        min_activation: float,
    ) -> AttributionGraph:
        """
        Prune nodes with activation below threshold.
        
        Args:
            graph: Attribution graph
            min_activation: Minimum activation to keep
        """
        # Keep nodes with activation above threshold
        kept_nodes = [n for n in graph.nodes if n.activation >= min_activation]
        kept_node_ids = {id(n) for n in kept_nodes}
        
        # Keep edges between kept nodes
        kept_edges = [
            e for e in graph.edges
            if id(e.from_node) in kept_node_ids and id(e.to_node) in kept_node_ids
        ]
        
        new_graph = AttributionGraph()
        new_graph.nodes = kept_nodes
        new_graph.edges = kept_edges
        return new_graph
    
    def prune_by_layer_range(
        self,
        graph: AttributionGraph,
        min_layer: int,
        max_layer: int,
    ) -> AttributionGraph:
        """
        Keep only nodes within a specific layer range.
        
        Args:
            graph: Attribution graph
            min_layer: Minimum layer (inclusive)
            max_layer: Maximum layer (inclusive)
        """
        kept_nodes = [
            n for n in graph.nodes
            if min_layer <= n.layer <= max_layer
        ]
        kept_node_ids = {id(n) for n in kept_nodes}
        
        kept_edges = [
            e for e in graph.edges
            if id(e.from_node) in kept_node_ids and id(e.to_node) in kept_node_ids
        ]
        
        new_graph = AttributionGraph()
        new_graph.nodes = kept_nodes
        new_graph.edges = kept_edges
        return new_graph
    
    def prune_to_path(
        self,
        graph: AttributionGraph,
        source_layer: int,
        target_layer: int,
        k_paths: int = 10,
    ) -> AttributionGraph:
        """
        Keep only nodes on the top-k paths from source to target layer.
        
        Args:
            graph: Attribution graph
            source_layer: Source layer
            target_layer: Target layer
            k_paths: Number of paths to keep
        """
        G = graph.to_networkx()
        
        # Find source and target nodes
        source_nodes = [i for i, n in enumerate(graph.nodes) if n.layer == source_layer]
        target_nodes = [i for i, n in enumerate(graph.nodes) if n.layer == target_layer]
        
        # Find all paths
        all_paths = []
        for source in source_nodes:
            for target in target_nodes:
                try:
                    paths = list(nx.all_simple_paths(G, source, target))
                    all_paths.extend(paths)
                except:
                    continue
        
        # Score paths by total edge weight
        path_scores = []
        for path in all_paths:
            score = 0
            for i in range(len(path) - 1):
                if G.has_edge(path[i], path[i+1]):
                    score += abs(G[path[i]][path[i+1]]['weight'])
            path_scores.append((path, score))
        
        # Keep top-k paths
        path_scores.sort(key=lambda x: x[1], reverse=True)
        top_paths = path_scores[:k_paths]
        
        # Collect nodes and edges from top paths
        kept_node_indices = set()
        kept_edge_pairs = set()
        
        for path, _ in top_paths:
            for node_idx in path:
                kept_node_indices.add(node_idx)
            for i in range(len(path) - 1):
                kept_edge_pairs.add((path[i], path[i+1]))
        
        # Build new graph
        kept_nodes = [graph.nodes[i] for i in kept_node_indices]
        kept_node_ids = {id(graph.nodes[i]) for i in kept_node_indices}
        
        node_idx_to_id = {i: id(n) for i, n in enumerate(graph.nodes)}
        
        kept_edges = []
        for edge in graph.edges:
            from_idx = [i for i, n in enumerate(graph.nodes) if id(n) == id(edge.from_node)][0]
            to_idx = [i for i, n in enumerate(graph.nodes) if id(n) == id(edge.to_node)][0]
            if (from_idx, to_idx) in kept_edge_pairs:
                kept_edges.append(edge)
        
        new_graph = AttributionGraph()
        new_graph.nodes = kept_nodes
        new_graph.edges = kept_edges
        return new_graph