"""
src/attribution/graph.py
-----------------------
Phase 3d: Attribution graph construction.

Builds a directed acyclic graph where:
  - Nodes = transcoder features + error nodes
  - Edges = Jacobian values (influence weights)
  
The graph represents how features in early layers influence
features in later layers, ultimately affecting the output logits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import networkx as nx
import numpy as np

from src.attribution.jacobian import JacobianResult, AttributionNode, AttributionEdge


@dataclass
class AttributionGraph:
    """
    A directed attribution graph.
    
    Nodes:  features and error nodes at different layers/positions
    Edges:  weighted by Jacobians (∂downstream/∂upstream)
    """
    nodes: list[AttributionNode] = field(default_factory=list)
    edges: list[AttributionEdge] = field(default_factory=list)
    
    def add_node(self, node: AttributionNode) -> None:
        """Add a node to the graph."""
        self.nodes.append(node)
    
    def add_edge(self, edge: AttributionEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph for analysis/visualization."""
        G = nx.DiGraph()
        
        # Add nodes
        for i, node in enumerate(self.nodes):
            G.add_node(
                i,
                layer=node.layer,
                type=node.node_type,
                index=node.index,
                pos=node.position,
                activation=node.activation,
                label=f"L{node.layer}_{node.node_type[0]}{node.index}_p{node.position}"
            )
        
        # Add edges
        node_to_idx = {id(n): i for i, n in enumerate(self.nodes)}
        for edge in self.edges:
            from_idx = node_to_idx[id(edge.from_node)]
            to_idx = node_to_idx[id(edge.to_node)]
            G.add_edge(from_idx, to_idx, weight=edge.weight)
        
        return G
    
    def filter_by_threshold(self, threshold: float) -> AttributionGraph:
        """Return a new graph with edges below threshold removed."""
        filtered_edges = [e for e in self.edges if abs(e.weight) >= threshold]
        
        # Keep only nodes that have edges
        connected_nodes = set()
        for edge in filtered_edges:
            connected_nodes.add(id(edge.from_node))
            connected_nodes.add(id(edge.to_node))
        
        filtered_nodes = [n for n in self.nodes if id(n) in connected_nodes]
        
        new_graph = AttributionGraph()
        new_graph.nodes = filtered_nodes
        new_graph.edges = filtered_edges
        return new_graph
    
    def get_nodes_by_layer(self, layer: int) -> list[AttributionNode]:
        """Get all nodes from a specific layer."""
        return [n for n in self.nodes if n.layer == layer]
    
    def get_active_features(self, layer: int, position: int) -> list[AttributionNode]:
        """Get active feature nodes at a specific layer and position."""
        return [
            n for n in self.nodes
            if n.layer == layer and n.position == position and n.node_type == "feature" and n.activation > 0
        ]
    
    def get_top_k_edges(self, k: int) -> list[AttributionEdge]:
        """Get top-k edges by absolute weight."""
        return sorted(self.edges, key=lambda e: abs(e.weight), reverse=True)[:k]
    
    def compute_metrics(self) -> dict:
        """Compute graph statistics."""
        G = self.to_networkx()
        
        metrics = {
            'n_nodes': len(self.nodes),
            'n_edges': len(self.edges),
            'n_features': len([n for n in self.nodes if n.node_type == "feature"]),
            'n_errors': len([n for n in self.nodes if n.node_type == "error"]),
            'density': nx.density(G) if len(self.nodes) > 0 else 0,
        }
        
        # Nodes per layer
        layers = {}
        for node in self.nodes:
            layers[node.layer] = layers.get(node.layer, 0) + 1
        metrics['nodes_per_layer'] = layers
        
        # Edge weight statistics
        if self.edges:
            weights = [abs(e.weight) for e in self.edges]
            metrics['edge_weight_mean'] = np.mean(weights)
            metrics['edge_weight_std'] = np.std(weights)
            metrics['edge_weight_max'] = np.max(weights)
            metrics['edge_weight_min'] = np.min(weights)
        
        return metrics


class GraphBuilder:
    """
    Builds attribution graphs from Jacobian results.
    
    Usage:
        builder = GraphBuilder()
        graph = builder.build_graph(
            jacobian_result=jacobians,
            feature_acts=output.feature_acts,
            error_acts=output.error_acts,
            threshold=0.01,
        )
    """
    
    def build_graph(
        self,
        jacobian_result: JacobianResult,
        feature_acts: list[torch.Tensor],
        error_acts: list[torch.Tensor],
        threshold: float = 0.01,
        top_k_per_node: Optional[int] = None,
        activation_threshold: float = 1e-6,
    ) -> AttributionGraph:
        """
        Build an attribution graph from Jacobians and activations.
        
        Args:
            jacobian_result: Computed Jacobians
            feature_acts: Feature activations per layer
            error_acts: Error node activations per layer
            threshold: Minimum edge weight to include
            top_k_per_node: If set, keep only top-k edges per downstream node
            activation_threshold: Minimum activation to include a feature node
            
        Returns:
            AttributionGraph
        """
        graph = AttributionGraph()
        n_layers = len(feature_acts)
        
        # Create nodes
        node_map = {}  # (layer, type, index, pos) -> AttributionNode
        
        for layer_idx in range(n_layers):
            features = feature_acts[layer_idx][0]  # (seq_len, n_features)
            errors = error_acts[layer_idx][0]      # (seq_len, d_model)
            
            seq_len, n_features = features.shape
            _, d_model = errors.shape
            
            # Feature nodes
            for pos in range(seq_len):
                for feat_idx in range(n_features):
                    act_val = features[pos, feat_idx].item()
                    if act_val > activation_threshold:  # Only active features
                        node = AttributionNode(
                            layer=layer_idx,
                            node_type="feature",
                            index=feat_idx,
                            position=pos,
                            activation=act_val,
                        )
                        graph.add_node(node)
                        node_map[(layer_idx, "feature", feat_idx, pos)] = node
            
            # Error nodes (if significant)
            for pos in range(seq_len):
                error_norm = errors[pos].norm().item()
                if error_norm > threshold:
                    # Create aggregated error node
                    node = AttributionNode(
                        layer=layer_idx,
                        node_type="error",
                        index=0,  # Aggregated error
                        position=pos,
                        activation=error_norm,
                    )
                    graph.add_node(node)
                    node_map[(layer_idx, "error", 0, pos)] = node
        
        # Create edges from Jacobians
        for layer_idx in range(n_layers - 1):
            f2f_jac = jacobian_result.feature_to_feature[layer_idx]
            # (n_features_downstream, n_features_upstream, seq_len)
            
            downstream_layer = layer_idx + 1
            
            n_downstream, n_upstream, seq_len_j = f2f_jac.shape
            
            for pos in range(min(seq_len_j, features.shape[0])):
                for downstream_feat in range(n_downstream):
                    # Get downstream node
                    downstream_key = (downstream_layer, "feature", downstream_feat, pos)
                    if downstream_key not in node_map:
                        continue
                    downstream_node = node_map[downstream_key]
                    
                    # Find upstream connections
                    for upstream_feat in range(n_upstream):
                        weight = f2f_jac[downstream_feat, upstream_feat, pos].item()
                        
                        if abs(weight) < threshold:
                            continue
                        
                        upstream_key = (layer_idx, "feature", upstream_feat, pos)
                        if upstream_key not in node_map:
                            continue
                        upstream_node = node_map[upstream_key]
                        
                        # Weight by activation (attribution = weight * activation)
                        attribution = weight * upstream_node.activation
                        
                        edge = AttributionEdge(
                            from_node=upstream_node,
                            to_node=downstream_node,
                            weight=attribution,
                        )
                        graph.add_edge(edge)
        
        # Prune to top-k if requested
        if top_k_per_node is not None:
            graph = self._prune_to_top_k(graph, top_k_per_node)
        
        return graph
    
    def _prune_to_top_k(
        self, graph: AttributionGraph, k: int
    ) -> AttributionGraph:
        """Keep only top-k incoming edges per node."""
        # Group edges by downstream node
        edges_by_node = {}
        for edge in graph.edges:
            node_id = id(edge.to_node)
            if node_id not in edges_by_node:
                edges_by_node[node_id] = []
            edges_by_node[node_id].append(edge)
        
        # Keep top-k per node
        pruned_edges = []
        for node_id, edges in edges_by_node.items():
            sorted_edges = sorted(edges, key=lambda e: abs(e.weight), reverse=True)
            pruned_edges.extend(sorted_edges[:k])
        
        # Rebuild graph
        new_graph = AttributionGraph()
        new_graph.nodes = graph.nodes
        new_graph.edges = pruned_edges
        return new_graph
    
    def build_from_activations(
        self,
        feature_acts: list[torch.Tensor],
        error_acts: list[torch.Tensor],
        transcoders: list,
        threshold: float = 0.01,
    ) -> AttributionGraph:
        """
        Build attribution graph directly from activations and transcoder weights.
        
        This is a simplified version that uses decoder/encoder weights
        as proxies for Jacobians.
        """
        graph = AttributionGraph()
        n_layers = len(feature_acts)
        
        # Create nodes (same as before)
        node_map = {}
        for layer_idx in range(n_layers):
            features = feature_acts[layer_idx][0]
            seq_len, n_features = features.shape
            
            for pos in range(seq_len):
                for feat_idx in range(n_features):
                    if features[pos, feat_idx] > 1e-6:
                        node = AttributionNode(
                            layer=layer_idx,
                            node_type="feature",
                            index=feat_idx,
                            position=pos,
                            activation=features[pos, feat_idx].item(),
                        )
                        graph.add_node(node)
                        node_map[(layer_idx, "feature", feat_idx, pos)] = node
        
        # Create edges using virtual weights
        for layer_idx in range(n_layers - 1):
            # Get virtual weights (decoder @ encoder)
            W_dec = transcoders[layer_idx].W_dec  # (d_model, n_features_from)
            W_enc = transcoders[layer_idx + 1].W_enc  # (n_features_to, d_model)
            virtual_weights = W_enc @ W_dec  # (n_features_to, n_features_from)
            
            # Create edges
            features_from = feature_acts[layer_idx][0]
            features_to = feature_acts[layer_idx + 1][0]
            
            seq_len = min(features_from.shape[0], features_to.shape[0])
            
            for pos in range(seq_len):
                for to_feat in range(virtual_weights.shape[0]):
                    to_key = (layer_idx + 1, "feature", to_feat, pos)
                    if to_key not in node_map:
                        continue
                    
                    for from_feat in range(virtual_weights.shape[1]):
                        from_key = (layer_idx, "feature", from_feat, pos)
                        if from_key not in node_map:
                            continue
                        
                        weight = virtual_weights[to_feat, from_feat].item()
                        # Weight by activation
                        attribution = weight * node_map[from_key].activation
                        
                        if abs(attribution) >= threshold:
                            edge = AttributionEdge(
                                from_node=node_map[from_key],
                                to_node=node_map[to_key],
                                weight=attribution,
                            )
                            graph.add_edge(edge)
        
        return graph