# Docker Practice 🐳

A simple **Spam Classifier API** built with FastAPI and containerized using Docker.

This project is mainly for practicing **Docker, uv, FastAPI, and CI/CD concepts**.

## Tech Stack

* Python 3.13
* FastAPI
* Uvicorn
* scikit-learn
* NLTK
* uv
* Docker

## Project Structure

```text
docker-practice/
├── data/
├── models/
├── notebooks/
├── src/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── main.py
├── pyproject.toml
└── uv.lock
```

## Run Locally

Install dependencies:

```bash
uv sync
```

Run the API:

```bash
uv run uvicorn main:app --reload
```

API documentation:

```text
http://localhost:8000/docs
```

## Run with Docker

Build the image:

```bash
docker build -t spam-classifier:v1 .
```

Run the container:

```bash
docker run -p 8000:8000 spam-classifier:v1
```

Then open:

```text
http://localhost:8000/docs
```

## API

### `POST /predict`

Example request:

```json
{
  "text": "Congratulations! You won a free prize."
}
```

Example response:

```json
{
  "category": "Spam"
}
```

## Learning Goals

* Understand Docker images and containers
* Learn Dockerfile and image layers
* Manage Python dependencies with `uv`
* Containerize a FastAPI application
* Learn CI/CD with GitHub Actions
* Understand the basics of deploying ML applications

## Status

🚧 Work in Progress
