# IoMT-compromise


Python implementation and empirical evaluation for the AI-Guardian framework.

AI-Guardian combines deep perception restoration via a 1D Denoising Autoencoder (DAE) with real-time Control Barrier Function (CBF) Quadratic Programming (QP) projection 
to guarantee patient safety under microarchitectural credential compromise and hardware perturbation attacks.

## Repository Overview

- `src/models.py`: PyTorch implementations of the 1D Denoising Autoencoder (DAE) and ECG Diagnostic Classifier.
- `src/data.py`: PhysioNet MIT-BIH Arrhythmia database streaming and preprocessing functions.
- `scripts/generate_figures.py`: End-to-end benchmarking script generating publication-quality figures (Figures 1–4).

## Installation

### Prerequisites
- Python 3.9+
- PyTorch 2.0+

### Setup

```bash
# Clone repository
git clone [https://github.com/](https://github.com/)<your-username>/ai-guardian-iomt.git
cd ai-guardian-iomt

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
