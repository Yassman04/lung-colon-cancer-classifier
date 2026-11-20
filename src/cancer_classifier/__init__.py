"""Lung and colon cancer histopathology image classifier (PyTorch)."""

CLASS_NAMES = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]

CLASS_LABELS = {
    "colon_aca": "Colon adenocarcinoma",
    "colon_n": "Benign colon tissue",
    "lung_aca": "Lung adenocarcinoma",
    "lung_n": "Benign lung tissue",
    "lung_scc": "Lung squamous cell carcinoma",
}

__version__ = "1.0.0"
