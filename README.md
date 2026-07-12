
<p align="center">
  <img src="assets/logo.png" width="280">
</p>

<h1 align="center">QUASAR</h1>

<h3 align="center">
Quantum Understanding and Assessment System for Reliable Analysis
</h3>

<p align="center">
<b>An AI-Assisted Quantum Circuit Reliability Analysis Platform</b>
</p>

---

## Overview

**QUASAR (Quantum Understanding and Assessment System for Reliable Analysis)** is an AI-assisted platform that evaluates the reliability of quantum circuits before execution on noisy quantum hardware.

The platform combines **Qiskit**, **machine learning**, **quantum noise simulation**, and **Explainable AI (SHAP & LIME)** to estimate circuit reliability, explain the reasoning behind each prediction, and generate actionable recommendations for improving circuit robustness.

Rather than focusing solely on circuit execution, QUASAR emphasizes **understanding**, **assessment**, and **reliability analysis**, enabling researchers, students, and developers to study how circuit characteristics influence expected performance under realistic quantum noise.

---

## Problem Statement

Current quantum computers operate in the Noisy Intermediate-Scale Quantum (NISQ) era, where decoherence, gate errors, measurement errors, and limited qubit fidelity can significantly reduce circuit performance.

Although existing quantum frameworks provide tools for circuit design and simulation, interpreting circuit reliability often requires multiple workflows and considerable expertise. There is a need for an integrated platform that predicts reliability, explains the prediction, and recommends improvements before deployment.

---

## Our Solution

QUASAR provides a complete analysis pipeline capable of:

- Parsing OpenQASM quantum circuits
- Extracting structural and operational circuit features
- Simulating realistic quantum noise
- Predicting reliability using a trained machine learning model
- Explaining predictions using SHAP and LIME
- Generating recommendations for improving circuit reliability
- Presenting results through an interactive Streamlit dashboard

---

## Key Features

### Quantum Circuit Parsing

Supports OpenQASM-based circuit analysis using Qiskit.

### Feature Extraction

Computes circuit-level metrics including gate counts, depth, width, parameterized operations, entangling gates, fidelity estimates, and additional reliability-related features.

### Noise Simulation

Evaluates circuit behaviour under realistic noisy conditions.

### Machine Learning Prediction

Predicts circuit reliability using a trained Gradient Boosting model.

### Explainable AI

Provides both global and local explanations using SHAP and LIME.

### Recommendation Engine

Suggests practical improvements to increase predicted circuit reliability.

### Interactive Dashboard

Visualizes predictions, explanations, reports, and supporting analytics through Streamlit.

---

## Repository Structure

```text
src/
assets/
circuits/
datasets/
models/
plots/
reports/

app.py
main.py
generate_dataset.py
requirements.txt
```

---

## Technology Stack

- Python
- Qiskit
- Qiskit Aer
- Scikit-learn
- SHAP
- LIME
- Streamlit
- Plotly
- Pandas
- NumPy

---

## Authors

- Khatwang Madhav Yippili

---

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.
