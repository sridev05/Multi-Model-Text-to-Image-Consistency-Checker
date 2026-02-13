# Multi-Model Text-to-Image Consistency Checker

**A research-grade evaluation system that measures consistency between text prompts and generated images using multiple AI models (CLIP, BLIP, Grounding DINO, TIFA, OCR) with both web API and React frontend.**

## Overview

Multi-Model Text-to-Image Consistency Checker is a comprehensive evaluation platform designed for AI researchers and practitioners working with text-to-image generation models. The system evaluates how well generated images align with their source text prompts through a sophisticated multi-model pipeline.

## Key Features

- **Bidirectional Evaluation**: Forward (text→image) and backward (image→text→image) consistency scoring
- **Multi-Model Pipeline**: Integrates CLIP, BLIP, Grounding DINO, TIFA, and OCR for comprehensive analysis
- **CPU-Only Inference**: Deterministic results without GPU requirements
- **RESTful API**: Easy integration with existing workflows
- **Interactive Frontend**: React-based web interface for manual evaluation
- **Research-Grade Accuracy**: Final scores from 0-100 with categorical verdicts (MATCH/PARTIAL MATCH/MISMATCH)
- **Broad Compatibility**: Support for images from Stable Diffusion, DALL·E, Gemini, and other generators

## Evaluation Components

| Component | Model | Purpose |
|-----------|-------|---------|
| **CLIP** | `openai/clip-vit-base-patch32` | Semantic alignment between text and image |
| **BLIP** | `Salesforce/blip-image-captioning-base` | Dense image captioning for semantic grounding |
| **Grounding DINO** | `IDEA-Research/grounding-dino-tiny` | Object-level grounding and entity presence detection |
| **TIFA** | `google/flan-t5-small` | Atomic fact decomposition and faithfulness verification |
| **OCR** | PaddleOCR | Visible text extraction and comparison |

## Use Cases

- **AI Researchers**: Validate and benchmark text-to-image model performance
- **Content Creators**: Ensure prompt fidelity in generated images
- **Developers**: Integrate consistency checking into AI-powered applications
- **Quality Assurance**: Automated evaluation of large-scale image generation workflows

