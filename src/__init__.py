"""
Attribution Graphs Project
--------------------------
Implementation of the attribution graphs method from:
https://transformer-circuits.pub/2025/attribution-graphs/

Main components:
  - model: Base model loading and activation caching
  - transcoder: Sparse autoencoder transcoders (PLT and CLT)
  - replacement_model: Local replacement model with frozen attention
  - attribution: Jacobian computation and graph construction
  - utils: Visualization and helper functions
"""

__version__ = "0.1.0"