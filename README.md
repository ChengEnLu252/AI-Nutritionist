# AI Nutritionist

An AI-powered nutrition assistant that combines Retrieval-Augmented Generation (RAG), image analysis, and personal health tracking to deliver dietary guidance and personalized coaching.

## Overview

AI Nutritionist is an end-to-end intelligent nutrition support system designed to help users:

- ask food and nutrition related questions through a RAG-based chatbot
- analyze uploaded food images with computer vision models
- record body weight, diet logs, and health goals
- receive personalized coaching suggestions based on recent health data

This project is not just a chatbot. It is a multi-module AI application that integrates LLMs, vector search, computer vision, and user data management into one practical system.

## Key Features

### 1. Nutrition Q&A with RAG
- Retrieves relevant nutrition knowledge from a food database
- Generates grounded answers with source references
- Reduces hallucination by using retrieval before generation

### 2. Food Image Analysis
- Supports uploaded food images
- Uses segmentation and vision models for food-related image understanding
- Enables multimodal interaction beyond text-only input

### 3. Personal Health Tracking
- Record body weight logs
- Record meal descriptions and calorie-related information
- Save and manage user goals

### 4. Personalized AI Coaching
- Summarizes recent diet and weight records
- Generates one-week health suggestions
- Produces goal-oriented feedback in a coaching style

## System Architecture

The system consists of four major modules:

1. **Frontend Interface**
   - Built with Gradio
   - Provides chat, image upload, and health record interaction

2. **RAG Pipeline**
   - Retrieves nutrition-related documents
   - Uses LLM-based answer generation with retrieved context

3. **Multimodal Image Module**
   - Integrates segmentation and vision models for image-based analysis

4. **User Data Layer**
   - Stores users, diet logs, weight logs, and goals
   - Supports personalized long-term coaching

## Tech Stack

- **Language:** Python
- **UI:** Gradio
- **LLM / Orchestration:** LangChain
- **Retrieval / Search:** RAG pipeline
- **Computer Vision:** SAM, ViT
- **Database:** Supabase
- **Data Processing:** Pandas
- **Deep Learning:** PyTorch

## Project Structure

```bash
AI-Nutritionist/
├── app.py
├── requirements.txt
├── notebooks/
├── src/
│   └── util/
├── test_image/
├── test_sam_minimal.py
└── README.md
