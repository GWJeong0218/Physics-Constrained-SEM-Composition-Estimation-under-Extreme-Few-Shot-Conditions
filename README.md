# Physics-Constrained SEM Composition Estimation under Extreme Few-Shot Conditions

This repository provides a reference implementation of the core components used in my proof-of-concept study on SEM-image-based composition estimation under extreme few-shot conditions.

The main goal is to examine whether simple physical constraints can reduce the effective hypothesis space in a data-scarce inverse characterization problem.

## Core Idea

Modern deep learning models can produce physically infeasible outputs when trained with very limited scientific imaging data.

This project explores a physics-constrained approach for SEM-based composition estimation by combining:

- Z-contrast-inspired synthetic SEM image generation
- SEM-style imaging corruptions and tone matching
- Sim-to-real backbone pretraining
- Leave-one-image-out evaluation under a small real SEM dataset
- Output-space constraints:
  - non-negativity
  - sum-to-one composition
  - optional element-support hard masking
  - optional leak penalty on absent elements

## Repository Structure

```text
physics-constrained-ml/
├── create_sim_images.py   # Synthetic SEM image generator
└── run_loio.py            # LOIO runner for few-shot SEM composition estimation
