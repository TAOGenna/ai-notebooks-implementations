import numpy as np

def pick_top_k(model, X, k=2, class_idx=1):
    """Return row indices for the top-k examples by P(y=class_idx)."""
    proba = model.predict_proba(X)          # shape: [n, n_classes]
    scores = proba[:, class_idx]
    return np.argsort(-scores)[:k].tolist()
