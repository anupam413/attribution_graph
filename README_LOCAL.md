# Attribution Graphs Project

Implementation of the attribution graphs method from:
[Attribution Graphs: Mechanistic Interpretability for Large Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

## Overview

This codebase implements the full attribution graphs pipeline:
1. **Phase 2**: Base model loading and activation caching
2. **Phase 3a**: Per-layer transcoders (PLT)
3. **Phase 3b**: Cross-layer transcoders (CLT)
4. **Phase 3c**: Local replacement model
5. **Phase 3d**: Attribution graph construction
6. **Phase 3e**: Graph pruning and analysis
