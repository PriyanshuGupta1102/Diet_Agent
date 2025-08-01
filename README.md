# 🍏 AI Diet Agent (Structured Output & Q&A)

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a sophisticated AI-powered diet assistant built following the architecture outlined in the `cura-backend` documentation. It is composed of two specialized agents: a conversational Q&A bot for general nutrition inquiries and a structured diet plan generator that outputs clean, predictable JSON.

The project leverages the LangChain framework to interact with Google's Gemini models, ensuring high-quality responses and robust functionality.

---

## ✨ Core Features

This project is divided into two main components:

### 1. Conversational Q&A Agent (`diet_agent.py`)
- **Context-Aware Conversations:** Remembers previous turns in the conversation to answer follow-up questions accurately.
- **Retrieval-Augmented Generation (RAG):** Answers questions based *only* on the information provided in a local `diet_data.txt` file, ensuring factual and controlled responses.
- **Efficient Knowledge Search:** Uses FAISS, a powerful local vector store, for fast and relevant information retrieval.

### 2. Structured Diet Plan Generator (`plan_generator.py`)
- **Guaranteed JSON Output:** Leverages Pydantic models to force the AI to return a diet plan in a strict, predefined JSON format, making it perfect for API integration.
- **Personalized Plan Generation:** Creates a detailed, one-day meal plan based on user-provided details like age, weight, goals, and dietary restrictions.
- **Reliable and Predictable:** Eliminates the risk of the AI returning unstructured text, making the output machine-readable and easy to parse.

---

## 🛠️ Tech Stack

- **Core Language:** Python 3.9+
- **AI Framework:** LangChain
- **LLM:** Google Gemini (`gemini-1.5-flash-latest`)
- **Embeddings:** Google AI Studio (`models/embedding-001`)
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Data Validation & Structuring:** Pydantic

---

## 📂 Project Structure
