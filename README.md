# llm-syllogism

A systematic evaluation framework for analyzing syllogistic reasoning and structural biases in large language models (LLMs).

This repository contains the complete pipeline for my independent research project on **systematic biases in LLMs’ syllogistic reasoning**, later accepted by **AAAI 2026 Bridge: Logical and Symbolic Reasoning in Language Models**. It automatically generates controlled syllogistic reasoning tasks across all classical forms, queries multiple LLMs concurrently, parses and validates responses, and produces statistical analyses and visualizations.

---

## 🔍 Key Features

### 1. Automatic Dataset Generation
- Uses **WordNet** to generate *11,000 controlled syllogistic tasks* across **all 44 classical forms**.
- Maintains consistent semantic structure while varying lexical content.

### 2. Concurrent LLM Query Engine
- Built with **asyncio** + **aiohttp** for high-throughput inference.
- Supports **20–30 concurrent requests** across multiple LLM APIs.
- Unified interface for models such as GPT-4o, Gemini-2.0-Flash, LLaMA-3.3-70B, Qwen3-Max, and DeepSeek-Chat.

### 3. Robust Parsing & Error Handling
- Automatic JSON and text parsing with fallback strategies.
- Handles malformed outputs, retries, and rate limits.

### 4. Statistical Analysis & Visualization
- Computes accuracy matrices, syllogism-wise breakdowns, and **cross-model Pearson correlations**.
- Includes clustering and heatmaps for revealing shared failure modes.

### 5. Key Findings Enabled by the Framework
- **Validity asymmetry** (LLMs are more accurate on valid than invalid forms)
- **Middle-term position bias**
- **High cross-model consistency** in reasoning patterns

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt

### 2. Generate the syllogism dataset
```bash
python scripts/syllogism_generation.py

### 3. Format results into structured prompt–response pairs
```bash
python scripts/promptify.py

### 4. Query LLM APIs concurrently
```bash
python scripts/run_api.py
