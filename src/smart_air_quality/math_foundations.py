"""Mathematical foundations module for Smart Air Quality Prediction System.
Implements syllabus-grounded mathematical techniques directly on atmospheric data:
1. Linear Algebra: Covariance matrix, Eigenvalues/Eigenvectors, Principal Component Analysis (PCA).
2. Probability & Statistics: Welch's t-test hypothesis testing, 95% Confidence Intervals, moments.
3. Optimization: Gradient Descent variants (SGD, Momentum, RMSprop, Adam) on AQI regression loss.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def compute_air_quality_pca(df: pd.DataFrame, pollutant_cols: list[str] | None = None) -> dict:
    """Compute PCA via eigenvalue decomposition of the covariance matrix on atmospheric criteria pollutants."""
    if pollutant_cols is None:
        pollutant_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
    
    # Filter available columns
    available_cols = [c for c in pollutant_cols if c in df.columns]
    valid_sub = df[available_cols].dropna().copy()
    
    if len(valid_sub) < 50:
        return {"error": "Insufficient valid data for PCA"}
    
    X = valid_sub.to_numpy()
    n_samples, n_features = X.shape
    
    # Standardize (zero mean, unit variance)
    mean_vec = np.mean(X, axis=0)
    std_vec = np.std(X, axis=0)
    std_vec[std_vec == 0] = 1.0
    X_std = (X - mean_vec) / std_vec
    
    # Covariance Matrix: Sigma = (1 / (N - 1)) * X^T * X
    cov_matrix = np.cov(X_std, rowvar=False)
    
    # Eigenvalue & Eigenvector Decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort descending
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]
    
    # Explained variance ratios
    total_var = np.sum(eigenvalues)
    var_explained = eigenvalues / total_var
    cum_var_explained = np.cumsum(var_explained)
    
    # Project data to top 2 PCs
    projected_2d = X_std @ eigenvectors[:, :2]
    
    loadings = pd.DataFrame(
        eigenvectors[:, :3],
        index=available_cols,
        columns=["PC1_Loading", "PC2_Loading", "PC3_Loading"]
    ).reset_index().rename(columns={"index": "Pollutant"})
    
    return {
        "features": available_cols,
        "n_samples": n_samples,
        "cov_matrix": cov_matrix.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "var_explained": var_explained.tolist(),
        "cum_var_explained": cum_var_explained.tolist(),
        "loadings": loadings.to_dict(orient="records"),
        "projected_sample": projected_2d[:500].tolist()
    }


def perform_seasonal_hypothesis_test(df: pd.DataFrame) -> dict:
    """Perform Welch's two-sample t-test evaluating Winter vs Monsoon AQI disparities."""
    df_clean = df.dropna(subset=["Date", "AQI"]).copy()
    df_clean["Month"] = pd.to_datetime(df_clean["Date"]).dt.month
    
    winter_aqi = df_clean.loc[df_clean["Month"].isin([12, 1, 2]), "AQI"].dropna()
    monsoon_aqi = df_clean.loc[df_clean["Month"].isin([6, 7, 8, 9]), "AQI"].dropna()
    
    if len(winter_aqi) < 30 or len(monsoon_aqi) < 30:
        return {"error": "Insufficient observations for hypothesis testing"}
    
    # Welch's t-test (unequal variances assumed)
    t_stat, p_val = stats.ttest_ind(winter_aqi, monsoon_aqi, equal_var=False)
    
    # Mann-Whitney U test (non-parametric rank-sum test)
    u_stat, u_pval = stats.mannwhitneyu(winter_aqi, monsoon_aqi, alternative="two-sided")
    
    # 95% Confidence Intervals for means
    def get_ci(data):
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = stats.t.interval(0.95, len(data) - 1, loc=mean, scale=sem)
        return float(mean), float(ci[0]), float(ci[1]), float(np.std(data))
    
    w_mean, w_low, w_high, w_std = get_ci(winter_aqi)
    m_mean, m_low, m_high, m_std = get_ci(monsoon_aqi)
    
    return {
        "winter_n": int(len(winter_aqi)),
        "winter_mean": round(w_mean, 2),
        "winter_ci_95": [round(w_low, 2), round(w_high, 2)],
        "winter_std": round(w_std, 2),
        "monsoon_n": int(len(monsoon_aqi)),
        "monsoon_mean": round(m_mean, 2),
        "monsoon_ci_95": [round(m_low, 2), round(m_high, 2)],
        "monsoon_std": round(m_std, 2),
        "mean_difference": round(w_mean - m_mean, 2),
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(p_val),
        "statistically_significant": bool(p_val < 0.001),
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p": float(u_pval)
    }


def simulate_optimizers_on_aqi_regression(
    df: pd.DataFrame,
    epochs: int = 60,
    learning_rate: float = 0.05
) -> dict:
    """Simulate and compare gradient descent optimization algorithms on the real atmospheric regression loss surface.
    
    Compares:
    1. Batch Gradient Descent (BGD)
    2. Stochastic Gradient Descent (SGD)
    3. Momentum Optimizer (Polyak momentum, beta=0.9)
    4. RMSprop Optimizer (beta=0.99)
    5. Adam Optimizer (beta1=0.9, beta2=0.999)
    """
    sub = df[["PM2.5", "PM10", "NO2", "CO", "AQI"]].dropna().copy()
    if len(sub) < 200:
        # Fallback dummy data if dataset is small
        np.random.seed(42)
        X = np.random.randn(500, 3)
        y = 2.5 * X[:, 0] - 1.8 * X[:, 1] + 3.0 * X[:, 2] + np.random.randn(500) * 0.5
    else:
        # Take a balanced slice for fast interactive optimization
        X = sub[["PM2.5", "PM10", "NO2", "CO"]].to_numpy()[:1000]
        y = sub["AQI"].to_numpy()[:1000]
        # Standardize features
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        # Add intercept column
        X = np.hstack([np.ones((len(X), 1)), X])
        y = (y - np.mean(y)) / (np.std(y) + 1e-8)

    n_samples, n_features = X.shape
    
    def compute_loss(w):
        res = X @ w - y
        return 0.5 * np.mean(res ** 2)

    def compute_grad(w, indices=None):
        if indices is None:
            res = X @ w - y
            return (X.T @ res) / n_samples
        else:
            Xi = X[indices]
            yi = y[indices]
            res = Xi @ w - yi
            return (Xi.T @ res) / len(indices)

    # Common initialization
    w_init = np.ones(n_features) * 2.0
    
    # 1. Batch Gradient Descent
    w_bgd = w_init.copy()
    loss_bgd = []
    for _ in range(epochs):
        loss_bgd.append(compute_loss(w_bgd))
        grad = compute_grad(w_bgd)
        w_bgd -= learning_rate * grad

    # 2. Stochastic Gradient Descent (mini-batch size = 32)
    w_sgd = w_init.copy()
    loss_sgd = []
    batch_size = 32
    np.random.seed(42)
    for _ in range(epochs):
        loss_sgd.append(compute_loss(w_sgd))
        for _ in range(n_samples // batch_size):
            idx = np.random.choice(n_samples, batch_size, replace=False)
            grad = compute_grad(w_sgd, idx)
            w_sgd -= learning_rate * grad

    # 3. Momentum (beta=0.9)
    w_mom = w_init.copy()
    v_mom = np.zeros_like(w_mom)
    loss_mom = []
    beta = 0.9
    for _ in range(epochs):
        loss_mom.append(compute_loss(w_mom))
        grad = compute_grad(w_mom)
        v_mom = beta * v_mom + learning_rate * grad
        w_mom -= v_mom

    # 4. RMSprop (beta=0.99, eps=1e-8)
    w_rms = w_init.copy()
    s_rms = np.zeros_like(w_rms)
    loss_rms = []
    beta_rms = 0.99
    eps = 1e-8
    for _ in range(epochs):
        loss_rms.append(compute_loss(w_rms))
        grad = compute_grad(w_rms)
        s_rms = beta_rms * s_rms + (1 - beta_rms) * (grad ** 2)
        w_rms -= (learning_rate / (np.sqrt(s_rms) + eps)) * grad

    # 5. Adam (beta1=0.9, beta2=0.999, eps=1e-8)
    w_adam = w_init.copy()
    m_adam = np.zeros_like(w_adam)
    v_adam = np.zeros_like(w_adam)
    loss_adam = []
    beta1 = 0.9
    beta2 = 0.999
    for t in range(1, epochs + 1):
        loss_adam.append(compute_loss(w_adam))
        grad = compute_grad(w_adam)
        m_adam = beta1 * m_adam + (1 - beta1) * grad
        v_adam = beta2 * v_adam + (1 - beta2) * (grad ** 2)
        m_hat = m_adam / (1 - beta1 ** t)
        v_hat = v_adam / (1 - beta2 ** t)
        w_adam -= (learning_rate / (np.sqrt(v_hat) + eps)) * m_hat

    return {
        "epochs": list(range(1, epochs + 1)),
        "BGD": [float(v) for v in loss_bgd],
        "SGD": [float(v) for v in loss_sgd],
        "Momentum": [float(v) for v in loss_mom],
        "RMSprop": [float(v) for v in loss_rms],
        "Adam": [float(v) for v in loss_adam],
        "final_loss": {
            "BGD": round(float(loss_bgd[-1]), 5),
            "SGD": round(float(loss_sgd[-1]), 5),
            "Momentum": round(float(loss_mom[-1]), 5),
            "RMSprop": round(float(loss_rms[-1]), 5),
            "Adam": round(float(loss_adam[-1]), 5)
        }
    }
