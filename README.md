## Project Background
- While building AI agents, I wanted to validate an environment where users can easily leverage VLMs and broader multimodal capabilities.
- Vision-Language Models (VLMs) provide strong practical value by enabling workflows that use image data in addition to text.
- The goal is to make VLMs lighter and more accessible so that more people can use them comfortably in real-world settings.

## Project Objectives
- This project focuses on an **efficient, modular approach** that does **not require additional training**, making it highly practical.
- By introducing **vision token pruning**, **rater selection**, and **visual token recycling**, the model aims to **preserve semantic information while reducing computation**.
- Based on these requirements, **SparseVLM** was selected as the baseline.
- Ultimately, the project aims to develop a VLM that achieves **both efficiency and information preservation** through improved token selection strategies.

## Implementation Plan
- Add **spatiotemporal embeddings**.
- Improve **rater selection / pruning** (e.g., thresholding strategies).
- Explore **Graph Neural Networks (GNNs)** for **Visual Token Recycling**.

## Current Progress
- Completed a **pilot study** and preparing for the next phase.
- Due to security constraints, **evaluation (eval) code/results are not uploaded**.  
  - Recommended usage: run evaluation following the **LLaVA v1.5** evaluation guidelines.
- Implemented updates including:
  - Added **spatial embeddings**.
  - Modified **rater selection thresholds**.

