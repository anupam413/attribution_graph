"""
experiments/addition.py
----------------------
Reproduce the "36 + 59 = " arithmetic case study from the paper.

This experiment:
  1. Traces how the model computes addition step-by-step
  2. Identifies features for digit extraction, addition, and carry
  3. Shows the algorithmic circuit for multi-digit addition

Usage:
    python experiments/addition.py --model gpt2 --transcoder-path checkpoints/gpt2/plt_final.pt
    python experiments/addition.py --a 36 --b 59
"""

import argparse
from pathlib import Path

import torch
import numpy as np

from src.model.loader import ModelWrapper
from src.transcoder.per_layer import PerLayerTranscoder
from src.transcoder.cross_layer import CrossLayerTranscoder
from src.transcoder.train import TranscoderTrainer, CLTTrainer
from src.replacement_model.local import LocalReplacementModel
from src.attribution.graph import GraphBuilder
from src.attribution.prune import GraphPruner
from src.utils.viz import (
    print_graph_summary,
    plot_graph,
    plot_feature_activations,
    create_attribution_visualization,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", help="Model name")
    parser.add_argument("--transcoder-path", required=True, help="Path to trained transcoder")
    parser.add_argument("--clt", action="store_true", help="Use CLT instead of PLT")
    parser.add_argument("--a", type=int, default=36, help="First number")
    parser.add_argument("--b", type=int, default=59, help="Second number")
    parser.add_argument("--threshold", type=float, default=0.01, help="Edge threshold")
    parser.add_argument("--top-k", type=int, default=150, help="Keep top-k edges")
    parser.add_argument("--output-dir", default="outputs/addition", help="Output directory")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate expected result
    expected_result = args.a + args.b
    
    # -------------------------------------------------------------------------
    # 1. Load model and transcoder
    # -------------------------------------------------------------------------
    print(f"\n[1/5] Loading model: {args.model}")
    model_wrapper = ModelWrapper.load(args.model, device=device)
    base_model = model_wrapper.model

    base_model.set_use_attn_result(True)  # Enable attention weights for attribution
    
    print(f"\n[2/5] Loading transcoder: {args.transcoder_path}")
    
    if args.clt:
        # CLTTrainer.load_checkpoint returns CrossLayerTranscoder directly
        transcoder = CLTTrainer.load_checkpoint(args.transcoder_path, device=device)
    else:
        transcoders = []
        for layer_idx in range(model_wrapper.n_layers):
            tc_path = args.transcoder_path.replace("_final", f"_layer_{layer_idx}_final")
            tc_path = tc_path.replace("layer_0", f"layer_{layer_idx}")  # Handle layer_0 pattern
            
            if Path(tc_path).exists():
                trainer_or_tc = TranscoderTrainer.load_checkpoint(tc_path, device=device)
                # Handle both trainer and direct transcoder
                if hasattr(trainer_or_tc, 'transcoder'):
                    transcoders.append(trainer_or_tc.transcoder)
                else:
                    transcoders.append(trainer_or_tc)
            else:
                trainer_or_tc = TranscoderTrainer.load_checkpoint(args.transcoder_path, device=device)
                # Handle both trainer and direct transcoder
                if hasattr(trainer_or_tc, 'transcoder'):
                    tc = trainer_or_tc.transcoder
                else:
                    tc = trainer_or_tc
                transcoders = [tc for _ in range(model_wrapper.n_layers)]
                break
        transcoder = transcoders
    
    # -------------------------------------------------------------------------
    # 2. Test multiple formats
    # -------------------------------------------------------------------------
    print(f"\n[3/5] Testing addition: {args.a} + {args.b} = {expected_result}")
    
    test_prompts = [
        f"{args.a} + {args.b} =",
        f"calc: {args.a} + {args.b} =",
        f"What is {args.a} + {args.b}? The answer is",
        f"Q: {args.a} + {args.b} = ?\nA:",
    ]
    
    results = []
    
    for prompt in test_prompts:
        tokens = base_model.to_tokens(prompt)
        logits = base_model(tokens)
        probs = torch.softmax(logits[0, -1], dim=-1)
        
        top_probs, top_indices = probs.topk(5)
        top_tokens = [base_model.tokenizer.decode([idx]) for idx in top_indices]
        
        # Check if correct answer is in top-5
        correct_in_top5 = any(str(expected_result) in tok for tok in top_tokens)
        
        results.append({
            'prompt': prompt,
            'top_prediction': top_tokens[0],
            'correct_in_top5': correct_in_top5,
            'tokens': tokens,
        })
        
        print(f"\nPrompt: '{prompt}'")
        print(f"  Top prediction: '{top_tokens[0]}'")
        print(f"  Correct in top-5: {correct_in_top5}")
        print(f"  Top-5: {top_tokens}")
    
    # Choose best prompt for analysis
    best_result = max(results, key=lambda r: r['correct_in_top5'])
    prompt = best_result['prompt']
    tokens = best_result['tokens']
    
    print(f"\n{'='*60}")
    print(f"Using prompt for analysis: '{prompt}'")
    print(f"{'='*60}")
    
    # -------------------------------------------------------------------------
    # 3. Build attribution graph
    # -------------------------------------------------------------------------
    print(f"\n[4/5] Building attribution graph")
    
    replacement = LocalReplacementModel(base_model, transcoder, is_clt=args.clt)
    output = replacement(tokens)
    
    # Get target token
    target_logits = output.logits[0, -1]
    target_idx = target_logits.argmax().item()
    target_token = base_model.tokenizer.decode([target_idx])
    
    print(f"Target prediction: '{target_token}'")
    
    # Build graph
    builder = GraphBuilder()
    
    if args.clt:
        graph = builder.build_from_activations(
            output.feature_acts,
            output.error_acts,
            None,
            threshold=args.threshold,
        )
    else:
        graph = builder.build_from_activations(
            output.feature_acts,
            output.error_acts,
            transcoders,
            threshold=args.threshold,
        )
    
    print_graph_summary(graph)
    
    # Prune
    pruner = GraphPruner()
    pruned_graph = pruner.prune_by_top_k(graph, k=args.top_k)
    
    print(f"\nAfter pruning to top-{args.top_k} edges:")
    print_graph_summary(pruned_graph)
    
    # -------------------------------------------------------------------------
    # 4. Analyze arithmetic circuit
    # -------------------------------------------------------------------------
    print(f"\n[5/5] Analyzing arithmetic circuit")
    
    token_strs = [base_model.tokenizer.decode([t]) for t in tokens[0]]
    
    # Find position of each digit
    a_str = str(args.a)
    b_str = str(args.b)
    
    print(f"\nToken positions:")
    for i, tok in enumerate(token_strs):
        print(f"  {i}: '{tok}'")
    
    # Analyze features at different positions
    print(f"\n{'='*60}")
    print(f"FEATURES AT KEY POSITIONS")
    print(f"{'='*60}")
    
    # For each token position, show top features
    for pos, tok in enumerate(token_strs):
        print(f"\nPosition {pos} ('{tok}'):")
        
        # Show features from different layers
        for layer_idx in [0, model_wrapper.n_layers // 2, model_wrapper.n_layers - 1]:
            if layer_idx < len(output.feature_acts):
                features = output.feature_acts[layer_idx][0, pos]
                top_k = 3
                top_acts, top_indices = features.topk(top_k)
                
                if top_acts[0] > 0:
                    feature_strs = [f"F{idx.item()}({act.item():.3f})" 
                                     for act, idx in zip(top_acts, top_indices) if act > 0]
                    if feature_strs:
                        print(f"  L{layer_idx}: {', '.join(feature_strs)}")
    
    # -------------------------------------------------------------------------
    # 5. Visualize
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Creating visualizations")
    print(f"{'='*60}\n")
    
    # Graph visualization
    plot_graph(
        pruned_graph,
        figsize=(20, 14),
        layout="hierarchical",
        save_path=output_dir / f"addition_{args.a}_{args.b}_graph.png",
        title=f"Arithmetic Circuit: {args.a} + {args.b} = {target_token}",
        show_activations=True,
    )
    
    # Feature activations
    plot_feature_activations(
        output.feature_acts,
        tokens=token_strs,
        layers_to_plot=list(range(min(8, len(output.feature_acts)))),
        top_k_features=20,
        save_path=output_dir / f"addition_{args.a}_{args.b}_features.png",
    )
    
    # Comprehensive visualization
    create_attribution_visualization(
        pruned_graph,
        prompt=prompt,
        tokens=token_strs,
        target_token=target_token,
        save_path=output_dir / f"addition_{args.a}_{args.b}_comprehensive.png",
    )
    
    # Save analysis summary
    summary_path = output_dir / f"addition_{args.a}_{args.b}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"ARITHMETIC CIRCUIT ANALYSIS\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Problem: {args.a} + {args.b} = {expected_result}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Model prediction: {target_token}\n\n")
        f.write(f"Graph Statistics:\n")
        f.write(f"  Nodes: {len(pruned_graph.nodes)}\n")
        f.write(f"  Edges: {len(pruned_graph.edges)}\n")
        f.write(f"  Features: {len([n for n in pruned_graph.nodes if n.node_type == 'feature'])}\n\n")
        f.write(f"Results saved to: {output_dir}\n")
    
    print(f"Analysis summary saved to: {summary_path}")
    print(f"\nAll results saved to: {output_dir}\n")


if __name__ == "__main__":
    main()