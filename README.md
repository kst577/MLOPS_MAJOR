# MLOPS Major Assignment

End-to-end MLOps pipeline using the Olivetti Faces dataset from scikit-learn.

## Project Overview

This project builds a complete automated MLOps pipeline:
- Train a Decision Tree classifier on face images
- Test model accuracy
- Automate training and testing with GitHub Actions (CI)
- Serve predictions via a Flask web app in Docker
- Deploy on Kubernetes with 3 replicas

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Initial project setup |
| `dev` | Model training (`train.py`), testing (`test.py`), and CI workflow |
| `docker_cicd` | Flask app, Dockerfile, Docker Hub, and Kubernetes deployment |

## Dataset

- **Olivetti Faces** — 400 grayscale face images (64×64 pixels) of 40 people
- Loaded via `sklearn.datasets.fetch_olivetti_faces`

## Model

- **DecisionTreeClassifier** from scikit-learn
- Saved as `savedmodel.pth` using joblib

## Links

- **GitHub:** https://github.com/kst577/MLOPS_MAJOR
- **Docker Hub:** _(to be added)_
