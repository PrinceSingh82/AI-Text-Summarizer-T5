# AI Text Summarizer using T5 Transformer

An AI-powered text summarization web application built using
T5 Transformer, Hugging Face Transformers, PyTorch, and FastAPI.

## Features

- Text summarization using T5 Transformer
- Hugging Face Transformers
- FastAPI backend
- Simple web interface
- CPU/GPU device support
- Trained model included using Git LFS

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- T5 Transformer
- FastAPI
- HTML
- CSS
- JavaScript

## Project Structure

AI-Text-Summarizer-T5/
│
├── app.py
├── index.html
├── saved_summary_model/
├── .gitignore
└── .gitattributes

## How to Run

Install dependencies:

pip install torch transformers sentencepiece fastapi uvicorn jinja2

Run the application:

uvicorn app:app --reload

Open in browser:

http://127.0.0.1:8000

## Model

The project uses a fine-tuned T5 model for text summarization.
