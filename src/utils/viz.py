"""
src/utils/viz.py
---------------
Visualization utilities for attribution graphs and transcoder analysis.

Reproduces all key visualizations from the paper:
  - Attribution graphs with features and error nodes
  - Feature activation heatmaps
  - Transcoder quality metrics (NMSE, L0, dead features)
  - Virtual weight matrices
  - Path analysis
"""

from typing import Optional, List, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import seaborn as sns

from src.attribution.graph import AttributionGraph


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def print_graph_summary(graph: AttributionGraph) -> None:
    """
    Print summary statistics of an attribution graph.
    
    Args:
        graph: Attribution graph to summarize
    """
    print(f"\n{'='*60}")
    print(f"ATTRIBUTION GRAPH SUMMARY")
    print(f"{'='*60}")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    
    # Nodes by type
    features = [n for n in graph.nodes if n.node_type == "feature"]
    errors = [n for n in graph.nodes if n.node_type == "error"]
    print(f"  Features: {len(features)}")
    print(f"  Errors: {len(errors)}")
    
    # Nodes by layer
    layers = {}
    for node in graph.nodes:
        layers[node.layer] = layers.get(node.layer, 0) + 1
    print(f"\nNodes per layer:")
    for layer in sorted(layers.keys()):
        print(f"  Layer {layer}: {layers[layer]}")
    
    # Edge weights
    if graph.edges:
        weights = [abs(e.weight) for e in graph.edges]
        print(f"\nEdge weights:")
        print(f"  Mean: {np.mean(weights):.4f}")
        print(f"  Std:  {np.std(weights):.4f}")
        print(f"  Max:  {np.max(weights):.4f}")
        print(f"  Min:  {np.min(weights):.4f}")
        print(f"  Median: {np.median(weights):.4f}")
    
    # Active features per layer
    print(f"\nActive features per layer:")
    for layer in sorted(set(n.layer for n in features)):
        layer_features = [n for n in features if n.layer == layer]
        active_at_pos = {}
        for n in layer_features:
            if n.position not in active_at_pos:
                active_at_pos[n.position] = 0
            active_at_pos[n.position] += 1
        if active_at_pos:
            mean_active = np.mean(list(active_at_pos.values()))
            print(f"  Layer {layer}: {mean_active:.1f} features/token (avg)")
    
    print(f"{'='*60}\n")


def plot_graph(
    graph: AttributionGraph,
    figsize: Tuple[int, int] = (16, 12),
    layout: str = "hierarchical",
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show_edge_weights: bool = False,
    show_activations: bool = True,
) -> None:
    """
    Plot attribution graph with proper styling.
    
    Args:
        graph: Attribution graph
        figsize: Figure size
        layout: Layout algorithm ('hierarchical', 'spring', 'kamada_kawai')
        save_path: If provided, save plot to this path
        title: Plot title
        show_edge_weights: Whether to display edge weights as labels
        show_activations: Whether to size nodes by activation
    """
    G = graph.to_networkx()
    
    if len(G.nodes()) == 0:
        print("Empty graph, nothing to plot")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Choose layout
    if layout == "hierarchical":
        # Custom hierarchical layout by layer
        pos = _hierarchical_layout(graph)
    elif layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Node colors by type
    node_colors = []
    for node_idx in G.nodes():
        node = graph.nodes[node_idx]
        if node.node_type == "feature":
            node_colors.append('#3498db')  # Blue for features
        else:
            node_colors.append('#e74c3c')  # Red for errors
    
    # Node sizes by activation
    if show_activations:
        node_sizes = []
        for node_idx in G.nodes():
            node = graph.nodes[node_idx]
            size = 100 + min(node.activation * 500, 1000)
            node_sizes.append(size)
    else:
        node_sizes = 200
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8,
        ax=ax,
    )
    
    # Draw edges with width based on weight
    edges = G.edges()
    weights = [abs(G[u][v]['weight']) for u, v in edges]
    
    if weights:
        max_weight = max(weights)
        edge_widths = [1 + (w / max_weight) * 3 for w in weights]
    else:
        edge_widths = 1.0
    
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.5,
        arrows=True,
        arrowsize=15,
        edge_color='gray',
        ax=ax,
    )
    
    # Draw labels
    labels = {}
    for node_idx in G.nodes():
        node = graph.nodes[node_idx]
        if node.node_type == "feature":
            labels[node_idx] = f"L{node.layer}F{node.index}"
        else:
            labels[node_idx] = f"L{node.layer}E"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    # Draw edge weights if requested
    if show_edge_weights and len(edges) < 100:  # Only for small graphs
        edge_labels = {
            (u, v): f"{G[u][v]['weight']:.2f}"
            for u, v in edges
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, ax=ax)
    
    # Legend
    feature_patch = mpatches.Patch(color='#3498db', label='Features')
    error_patch = mpatches.Patch(color='#e74c3c', label='Errors')
    ax.legend(handles=[feature_patch, error_patch], loc='upper right')
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')
    else:
        ax.set_title("Attribution Graph", fontsize=16, fontweight='bold')
    
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def _hierarchical_layout(graph: AttributionGraph) -> dict:
    """Create hierarchical layout based on layer structure."""
    pos = {}
    
    # Group nodes by layer
    layers = {}
    for i, node in enumerate(graph.nodes):
        if node.layer not in layers:
            layers[node.layer] = []
        layers[node.layer].append(i)
    
    # Position nodes
    for layer_idx, node_indices in layers.items():
        n_nodes = len(node_indices)
        for i, node_idx in enumerate(node_indices):
            x = layer_idx
            y = (i - n_nodes / 2) * 0.5
            pos[node_idx] = (x, y)
    
    return pos


def plot_feature_activations(
    feature_acts: List[torch.Tensor],
    tokens: Optional[List[str]] = None,
    layers_to_plot: Optional[List[int]] = None,
    top_k_features: int = 20,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot feature activation heatmap.
    
    Args:
        feature_acts: List of feature activation tensors per layer
        tokens: Token strings for x-axis labels
        layers_to_plot: Which layers to plot (None = all)
        top_k_features: How many top features to show
        figsize: Figure size
        save_path: Path to save figure
    """
    if layers_to_plot is None:
        layers_to_plot = list(range(len(feature_acts)))
    
    n_layers = len(layers_to_plot)
    fig, axes = plt.subplots(n_layers, 1, figsize=figsize)
    
    if n_layers == 1:
        axes = [axes]
    
    for ax_idx, layer_idx in enumerate(layers_to_plot):
        acts = feature_acts[layer_idx][0].detach().cpu().numpy()  # (seq_len, n_features)
        
        # Get top-k most active features
        mean_acts = acts.mean(axis=0)
        top_features = np.argsort(mean_acts)[-top_k_features:][::-1]
        
        # Plot heatmap
        acts_subset = acts[:, top_features].T
        
        im = axes[ax_idx].imshow(
            acts_subset,
            aspect='auto',
            cmap='YlOrRd',
            interpolation='nearest',
        )
        
        axes[ax_idx].set_ylabel(f'Layer {layer_idx}\nFeatures', fontsize=10)
        axes[ax_idx].set_yticks(range(len(top_features)))
        axes[ax_idx].set_yticklabels([f'F{f}' for f in top_features], fontsize=8)
        
        if ax_idx == n_layers - 1:
            axes[ax_idx].set_xlabel('Token Position', fontsize=10)
            if tokens:
                axes[ax_idx].set_xticks(range(len(tokens)))
                axes[ax_idx].set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
        else:
            axes[ax_idx].set_xticks([])
        
        # Colorbar
        plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04)
    
    plt.suptitle('Feature Activations by Layer', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_transcoder_metrics(
    metrics: dict,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot transcoder quality metrics.
    
    Args:
        metrics: Dictionary with 'nmse', 'l0', 'dead_frac', etc. per layer
        figsize: Figure size
        save_path: Path to save figure
    """
    # Extract per-layer metrics
    layers = []
    nmse_values = []
    l0_values = []
    dead_frac_values = []
    
    for key, value in metrics.items():
        if key.startswith('layer_'):
            layer_idx = int(key.split('_')[1])
            layers.append(layer_idx)
            layer_metrics = value
            nmse_values.append(layer_metrics.get('nmse', 0))
            l0_values.append(layer_metrics.get('l0', 0))
            dead_frac_values.append(layer_metrics.get('dead_frac', 0))
    
    if not layers:
        print("No per-layer metrics found")
        return
    
    # Sort by layer
    sorted_data = sorted(zip(layers, nmse_values, l0_values, dead_frac_values))
    layers, nmse_values, l0_values, dead_frac_values = zip(*sorted_data)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot NMSE
    axes[0, 0].plot(layers, nmse_values, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Layer', fontsize=12)
    axes[0, 0].set_ylabel('NMSE', fontsize=12)
    axes[0, 0].set_title('Normalized MSE by Layer', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Target (0.1)')
    axes[0, 0].legend()
    
    # Plot L0
    axes[0, 1].plot(layers, l0_values, 'o-', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_xlabel('Layer', fontsize=12)
    axes[0, 1].set_ylabel('L0 (avg active features)', fontsize=12)
    axes[0, 1].set_title('Sparsity by Layer', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot dead features
    axes[1, 0].plot(layers, [d * 100 for d in dead_frac_values], 'o-', 
                     linewidth=2, markersize=8, color='red')
    axes[1, 0].set_xlabel('Layer', fontsize=12)
    axes[1, 0].set_ylabel('Dead Features (%)', fontsize=12)
    axes[1, 0].set_title('Dead Features by Layer', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"""
    TRANSCODER QUALITY SUMMARY
    
    Mean NMSE: {np.mean(nmse_values):.4f}
    Mean L0: {np.mean(l0_values):.1f}
    Mean Dead %: {np.mean(dead_frac_values)*100:.1f}%
    
    Best Layer (NMSE): {layers[np.argmin(nmse_values)]}
    Worst Layer (NMSE): {layers[np.argmax(nmse_values)]}
    
    Sparsest Layer: {layers[np.argmin(l0_values)]}
    Densest Layer: {layers[np.argmax(l0_values)]}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                     verticalalignment='center')
    
    plt.suptitle('Transcoder Quality Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_attribution_visualization(
    graph: AttributionGraph,
    prompt: str,
    tokens: List[str],
    target_token: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 12),
) -> None:
    """
    Create comprehensive attribution visualization (paper-style).
    
    This creates a multi-panel figure showing:
      - Attribution graph
      - Feature activations by layer
      - Top contributing paths
      
    Args:
        graph: Attribution graph
        prompt: Input prompt
        tokens: Tokenized input
        target_token: Target token being predicted
        save_path: Path to save figure
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Attribution graph
    ax1 = fig.add_subplot(gs[0:2, 0])
    _plot_graph_on_axis(graph, ax1, title=f"Attribution for '{target_token}'")
    
    # Panel 2: Statistics
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    stats_text = f"""
    PROMPT: {prompt}
    TARGET: {target_token}
    
    Graph Statistics:
    • Nodes: {len(graph.nodes)}
    • Edges: {len(graph.edges)}
    • Features: {len([n for n in graph.nodes if n.node_type == 'feature'])}
    • Errors: {len([n for n in graph.nodes if n.node_type == 'error'])}
    """
    ax2.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    # Panel 3: Top paths
    ax3 = fig.add_subplot(gs[1, 1])
    _plot_top_paths(graph, ax3)
    
    # Panel 4: Layer-wise feature counts
    ax4 = fig.add_subplot(gs[2, :])
    _plot_layer_feature_counts(graph, ax4)
    
    plt.suptitle(f'Attribution Analysis: {prompt} → {target_token}',
                 fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def _plot_graph_on_axis(graph: AttributionGraph, ax, title: str = ""):
    """Helper to plot graph on a specific axis."""
    G = graph.to_networkx()
    
    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, 'Empty graph', ha='center', va='center')
        ax.set_title(title)
        return
    
    pos = _hierarchical_layout(graph)
    
    node_colors = ['#3498db' if graph.nodes[n].node_type == "feature" else '#e74c3c' 
                    for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100,
                            alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=10,
                            edge_color='gray', ax=ax)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')


def _plot_top_paths(graph: AttributionGraph, ax):
    """Plot top attribution paths."""
    G = graph.to_networkx()
    
    # Find paths from early to late layers
    source_nodes = [i for i, n in enumerate(graph.nodes) if n.layer == 0]
    max_layer = max(n.layer for n in graph.nodes)
    target_nodes = [i for i, n in enumerate(graph.nodes) if n.layer == max_layer]
    
    # Collect path strengths
    path_strengths = []
    for source in source_nodes[:5]:  # Limit for performance
        for target in target_nodes[:5]:
            try:
                paths = list(nx.all_simple_paths(G, source, target, cutoff=max_layer+1))
                for path in paths[:3]:  # Top 3 paths per pair
                    strength = sum(abs(G[path[i]][path[i+1]]['weight']) 
                                    for i in range(len(path)-1) if G.has_edge(path[i], path[i+1]))
                    path_strengths.append(strength)
            except:
                continue
    
    if path_strengths:
        ax.hist(path_strengths, bins=20, edgecolor='black')
        ax.set_xlabel('Path Strength', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Distribution of Path Strengths', fontsize=11, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No paths found', ha='center', va='center')
        ax.set_title('Top Paths', fontsize=11, fontweight='bold')


def _plot_layer_feature_counts(graph: AttributionGraph, ax):
    """Plot number of active features per layer."""
    layer_counts = {}
    for node in graph.nodes:
        if node.node_type == "feature":
            if node.layer not in layer_counts:
                layer_counts[node.layer] = 0
            layer_counts[node.layer] += 1
    
    if layer_counts:
        layers = sorted(layer_counts.keys())
        counts = [layer_counts[l] for l in layers]
        
        ax.bar(layers, counts, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('Active Features', fontsize=10)
        ax.set_title('Active Features per Layer', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No features', ha='center', va='center')
        ax.set_title('Layer Feature Counts', fontsize=11, fontweight='bold')


def plot_virtual_weights_matrix(
    virtual_weights: torch.Tensor,
    layer_from: int,
    layer_to: int,
    top_k: int = 50,
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot virtual weight matrix between two layers.
    
    Args:
        virtual_weights: (n_features_to, n_features_from) matrix
        layer_from: Source layer
        layer_to: Target layer
        top_k: Show only top-k features by weight
        figsize: Figure size
        save_path: Path to save figure
    """
    weights = virtual_weights.cpu().numpy()
    
    # Get top features
    row_sums = np.abs(weights).sum(axis=1)
    col_sums = np.abs(weights).sum(axis=0)
    
    top_rows = np.argsort(row_sums)[-top_k:]
    top_cols = np.argsort(col_sums)[-top_k:]
    
    weights_subset = weights[np.ix_(top_rows, top_cols)]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(weights_subset, cmap='RdBu_r', aspect='auto',
                    vmin=-np.abs(weights_subset).max(),
                    vmax=np.abs(weights_subset).max())
    
    ax.set_xlabel(f'Layer {layer_from} Features', fontsize=12)
    ax.set_ylabel(f'Layer {layer_to} Features', fontsize=12)
    ax.set_title(f'Virtual Weights: Layer {layer_from} → Layer {layer_to}',
                 fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Weight')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved virtual weights matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()