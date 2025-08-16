
"""Enhanced classical ML algorithms with accurate Big O complexity analysis and optimizations."""

from typing import Dict, Tuple
import math
from ..registry import register_model

BYTES_FP32 = 4
BYTES_FP64 = 8
BYTES_INT32 = 4


def _dataset_mem_bytes(n: int, p: int, dtype_bytes: int = BYTES_FP32) -> int:
    """Calculate dataset memory requirements."""
    return n * p * dtype_bytes


def _get_complexity_class(n: int, p: int, operation: str) -> str:
    """Get Big O complexity classification for the operation."""
    complexities = {
        "linear": "O(n·p)",
        "quadratic_n": "O(n²·p)",
        "quadratic_p": "O(n·p²)",
        "cubic_p": "O(p³)",
        "cubic_n": "O(n³)",
        "loglinear_n": "O(n·p·log(n))",
        "kernel": "O(n²·p + n³)",
        "svd": "O(min(n²·p, n·p²))"
    }
    return complexities.get(operation, "O(?)")


@register_model("linear_regression_normal")
def linreg_normal(spec: Dict) -> Dict:
    """
    Linear regression using normal equations.
    Time Complexity: O(p³ + n·p²)
    Space Complexity: O(n·p + p²)
    
    Algorithm steps:
    1. Compute X^T X: O(n·p²)
    2. Cholesky decomposition: O(p³/3)
    3. Forward/backward substitution: O(p²)
    """
    n, p = spec["n"], spec["p"]
    
    # Detailed FLOP counting
    xtx_flops = n * p * p                    # X^T X computation
    cholesky_flops = p**3 // 3               # Cholesky decomposition
    solve_flops = p**2                       # Triangular solve
    total_flops = xtx_flops + cholesky_flops + solve_flops
    
    params = p + 1  # weights + bias
    mem = _dataset_mem_bytes(n, p) + p * p * BYTES_FP32  # Data + Gram matrix
    
    return {
        "params": params,
        "train_flops": total_flops,
        "infer_flops_per_sample": p,
        "dataset_mem_bytes": mem,
        "complexity_class": _get_complexity_class(n, p, "cubic_p"),
        "algorithm": "Normal Equations",
        "numerical_stability": "Poor for ill-conditioned X^T X",
        "scalability": "Limited by p³ term, good for p << n",
        "flop_breakdown": {
            "gram_matrix": xtx_flops,
            "cholesky": cholesky_flops,
            "solve": solve_flops
        }
    }


@register_model("linear_regression_qr")
def linreg_qr(spec: Dict) -> Dict:
    """
    Linear regression using QR decomposition (more stable).
    Time Complexity: O(n·p²)
    Space Complexity: O(n·p)
    """
    n, p = spec["n"], spec["p"]
    
    # QR decomposition FLOPs (Householder)
    qr_flops = 2 * n * p**2 - (2/3) * p**3
    solve_flops = p**2
    total_flops = qr_flops + solve_flops
    
    params = p + 1
    mem = _dataset_mem_bytes(n, p) + n * p * BYTES_FP32  # Data + Q factor
    
    return {
        "params": params,
        "train_flops": total_flops,
        "infer_flops_per_sample": p,
        "dataset_mem_bytes": mem,
        "complexity_class": _get_complexity_class(n, p, "quadratic_p"),
        "algorithm": "QR Decomposition",
        "numerical_stability": "Excellent",
        "scalability": "Good for tall matrices (n >> p)",
        "flop_breakdown": {
            "qr_decomposition": qr_flops,
            "back_substitution": solve_flops
        }
    }


@register_model("logistic_regression_newton")
def logreg_newton(spec: Dict) -> Dict:
    """
    Logistic regression using Newton-Raphson method.
    Time Complexity: O(k·(n·p² + p³)) where k is iterations
    Space Complexity: O(n·p + p²)
    """
    n, p = spec["n"], spec["p"]
    k = spec.get("iterations", 5)
    
    # Per iteration costs
    gradient_flops = n * p                   # Gradient computation
    hessian_flops = n * p**2                # Hessian computation
    cholesky_flops = p**3 // 3              # Hessian decomposition
    solve_flops = p**2                       # Newton step
    
    per_iter_flops = gradient_flops + hessian_flops + cholesky_flops + solve_flops
    total_flops = k * per_iter_flops
    
    params = p + 1
    mem = _dataset_mem_bytes(n, p) + p * p * BYTES_FP32  # Data + Hessian
    
    return {
        "params": params,
        "train_flops": total_flops,
        "infer_flops_per_sample": p + 10,  # p operations + sigmoid
        "dataset_mem_bytes": mem,
        "complexity_class": f"O({k}·(n·p² + p³))",
        "algorithm": "Newton-Raphson",
        "convergence": "Quadratic (near optimum)",
        "iterations": k,
        "scalability": "Cubic in p, may require regularization",
        "flop_breakdown": {
            "gradient_per_iter": gradient_flops,
            "hessian_per_iter": hessian_flops,
            "solve_per_iter": cholesky_flops + solve_flops,
            "total_iterations": k
        }
    }


@register_model("logistic_regression_sgd")
def logreg_sgd(spec: Dict) -> Dict:
    """
    Logistic regression using Stochastic Gradient Descent.
    Time Complexity: O(k·n·p) where k is epochs
    Space Complexity: O(n·p)
    """
    n, p = spec["n"], spec["p"]
    epochs = spec.get("epochs", 100)
    batch_size = spec.get("batch_size", 32)
    
    steps_per_epoch = math.ceil(n / batch_size)
    total_steps = epochs * steps_per_epoch
    flops_per_step = batch_size * p  # Gradient computation per mini-batch
    total_flops = total_steps * flops_per_step
    
    params = p + 1
    mem = _dataset_mem_bytes(n, p)
    
    return {
        "params": params,
        "train_flops": total_flops,
        "infer_flops_per_sample": p + 10,
        "dataset_mem_bytes": mem,
        "complexity_class": f"O({epochs}·n·p)",
        "algorithm": "Stochastic Gradient Descent",
        "convergence": "Linear (with proper learning rate)",
        "epochs": epochs,
        "batch_size": batch_size,
        "scalability": "Excellent for large datasets",
        "memory_efficient": True
    }


@register_model("svm_linear")
def svm_linear(spec: Dict) -> Dict:
    """
    Linear SVM using coordinate descent or SMO.
    Time Complexity: O(k·n·p) average case
    Space Complexity: O(n·p)
    """
    n, p = spec["n"], spec["p"]
    iters = spec.get("iterations", 1000)
    
    # SMO-style algorithm
    flops_per_iter = n * p  # Simplified kernel evaluations for linear case
    total_flops = iters * flops_per_iter
    
    params = p + 1
    mem = _dataset_mem_bytes(n, p) + n * BYTES_FP32  # Data + alpha coefficients
    
    return {
        "params": params,
        "train_flops": total_flops,
        "infer_flops_per_sample": p,
        "dataset_mem_bytes": mem,
        "complexity_class": _get_complexity_class(n, p, "linear"),
        "algorithm": "Sequential Minimal Optimization (SMO)",
        "kernel": "Linear",
        "iterations": iters,
        "scalability": "Good for sparse data"
    }


@register_model("svm_rbf")
def svm_rbf(spec: Dict) -> Dict:
    """
    RBF SVM - WARNING: O(n³) scaling makes this impractical for large n.
    Time Complexity: O(n²·p + n³)
    Space Complexity: O(n²)
    """
    n, p = spec["n"], spec["p"]
    gamma = spec.get("gamma", 1.0 / p)
    
    # Kernel matrix computation and QP solving
    kernel_flops = n**2 * p  # RBF kernel evaluations
    qp_flops = n**3          # Quadratic programming (worst case)
    total_flops = kernel_flops + qp_flops
    
    # Support vectors (worst case: all points)
    support_vectors = min(n, spec.get("max_support_vectors", n))
    params = support_vectors
    
    # Kernel matrix is the memory bottleneck
    kernel_mem = n**2 * BYTES_FP32
    data_mem = _dataset_mem_bytes(n, p)
    total_mem = data_mem + kernel_mem
    
    return {
        "params": params,
        "train_flops": total_flops,
        "infer_flops_per_sample": support_vectors * p,
        "dataset_mem_bytes": total_mem,
        "complexity_class": _get_complexity_class(n, p, "kernel"),
        "algorithm": "RBF Kernel SVM",
        "kernel": "Radial Basis Function",
        "scalability_warning": f"O(n³) scaling! Impractical for n > {int(1000)} on most systems",
        "memory_bottleneck": "Kernel matrix O(n²)",
        "support_vectors": support_vectors,
        "gamma": gamma,
        "flop_breakdown": {
            "kernel_computation": kernel_flops,
            "quadratic_programming": qp_flops
        }
    }


@register_model("decision_tree")
def decision_tree(spec: Dict) -> Dict:
    """
    Decision Tree using CART algorithm.
    Time Complexity: O(n·p·log(n)) average, O(n²·p) worst case
    Space Complexity: O(n·p + tree_size)
    """
    n, p = spec["n"], spec["p"]
    max_depth = spec.get("max_depth", int(math.log2(max(2, n))))
    min_samples_split = spec.get("min_samples_split", 2)
    
    # CART algorithm analysis
    avg_depth = min(max_depth, math.log2(max(2, n)))
    splits_considered = n * p * avg_depth  # Approximate
    total_flops = splits_considered * 10    # Cost per split evaluation
    
    # Tree size estimation
    max_nodes = min(2**(max_depth + 1) - 1, n)  # Can't have more nodes than samples
    actual_nodes = min(max_nodes, n // min_samples_split)
    
    params = actual_nodes * 3  # Each node: feature_id, threshold, prediction
    mem = _dataset_mem_bytes(n, p) + actual_nodes * 3 * BYTES_FP32
    
    return {
        "params": params,
        "train_flops": total_flops,
        "infer_flops_per_sample": avg_depth * 2,  # Path traversal
        "dataset_mem_bytes": mem,
        "complexity_class": _get_complexity_class(n, p, "loglinear_n"),
        "algorithm": "CART (Classification and Regression Trees)",
        "max_depth": max_depth,
        "actual_depth": avg_depth,
        "tree_nodes": actual_nodes,
        "interpretability": "High",
        "overfitting_risk": "High without pruning/regularization"
    }


@register_model("random_forest")
def random_forest(spec: Dict) -> Dict:
    """
    Random Forest ensemble.
    Time Complexity: O(T·n·p·log(n)) where T is number of trees
    Space Complexity: O(T·tree_size + n·p)
    """
    n, p = spec["n"], spec["p"]
    n_trees = spec.get("n_estimators", 100)
    max_depth = spec.get("max_depth", int(math.log2(max(2, n))))
    max_features = spec.get("max_features", int(math.sqrt(p)))  # Feature subsampling
    
    # Each tree is trained on sqrt(p) features
    single_tree = decision_tree({
        "n": n, 
        "p": max_features, 
        "max_depth": max_depth
    })
    
    total_flops = n_trees * single_tree["train_flops"]
    total_params = n_trees * single_tree["params"]
    
    # Bootstrap sampling adds minimal cost
    bootstrap_flops = n_trees * n
    total_flops += bootstrap_flops
    
    mem = _dataset_mem_bytes(n, p) + total_params * BYTES_FP32
    
    return {
        "params": total_params,
        "train_flops": total_flops,
        "infer_flops_per_sample": n_trees * single_tree["infer_flops_per_sample"],
        "dataset_mem_bytes": mem,
        "complexity_class": f"O({n_trees}·n·p·log(n))",
        "algorithm": "Random Forest (Bootstrap Aggregation)",
        "n_estimators": n_trees,
        "max_features": max_features,
        "parallelizable": True,
        "variance_reduction": "Excellent",
        "interpretability": "Medium",
        "single_tree_complexity": single_tree["complexity_class"]
    }


@register_model("xgboost_like")
def xgboost_like(spec: Dict) -> Dict:
    """
    Gradient Boosting (XGBoost-style) with histogram-based optimization.
    Time Complexity: O(T·n·p·log(n)) with histogram optimization
    Space Complexity: O(n·p + T·tree_size)
    """
    n, p = spec["n"], spec["p"]
    n_trees = spec.get("n_estimators", 100)
    max_depth = spec.get("max_depth", 6)
    n_bins = spec.get("n_bins", 256)  # Histogram bins for optimization
    
    # Histogram-based tree building
    histogram_flops = n * p * n_bins      # Build histograms
    tree_building_flops = p * n_bins * max_depth  # Tree construction from histograms
    gradient_flops = n * 5                # Gradient/Hessian computation per tree
    
    per_tree_flops = histogram_flops + tree_building_flops + gradient_flops
    total_flops = n_trees * per_tree_flops
    
    nodes_per_tree = min(2**max_depth - 1, n // 2)
    total_params = n_trees * nodes_per_tree * 3
    
    mem = _dataset_mem_bytes(n, p) + total_params * BYTES_FP32
    mem += n_bins * p * BYTES_FP32  # Histogram storage
    
    return {
        "params": total_params,
        "train_flops": total_flops,
        "infer_flops_per_sample": n_trees * max_depth * 2,
        "dataset_mem_bytes": mem,
        "complexity_class": f"O({n_trees}·n·p·log(n))",
        "algorithm": "Gradient Boosting with Histogram Optimization",
        "n_estimators": n_trees,
        "max_depth": max_depth,
        "histogram_bins": n_bins,
        "optimization": "Second-order (Newton-Raphson)",
        "regularization": "L1 + L2 built-in",
        "parallelizable": "Tree-level and feature-level",
        "memory_efficient": "Histogram-based splits"
    }


@register_model("kmeans")
def kmeans(spec: Dict) -> Dict:
    """
    K-Means clustering using Lloyd's algorithm.
    Time Complexity: O(k·n·p·i) where i is iterations
    Space Complexity: O(n·p + k·p)
    """
    n, p = spec["n"], spec["p"]
    k = spec.get("k", 10)
    max_iters = spec.get("max_iter", 300)
    tol = spec.get("tol", 1e-4)
    
    # Lloyd's algorithm costs per iteration
    assignment_flops = k * n * p          # Distance calculations
    centroid_update_flops = n * p          # Mean computation
    convergence_check_flops = k * p        # Centroid movement check
    
    per_iter_flops = assignment_flops + centroid_update_flops + convergence_check_flops
    
    # Typical convergence is much faster than max_iters
    expected_iters = min(max_iters, max(10, int(math.log(n))))
    total_flops = expected_iters * per_iter_flops
    
    params = k * p  # Centroid coordinates
    mem = _dataset_mem_bytes(n, p) + k * p * BYTES_FP32 + n * BYTES_INT32  # Data + centroids + assignments
    
    return {
        "params": params,
        "train_flops": total_flops,
        "infer_flops_per_sample": k * p,  # Distance to all centroids
        "dataset_mem_bytes": mem,
        "complexity_class": f"O(k·n·p·{expected_iters})",
        "algorithm": "Lloyd's Algorithm (K-Means)",
        "k_clusters": k,
        "expected_iterations": expected_iters,
        "max_iterations": max_iters,
        "tolerance": tol,
        "initialization": "K-Means++",
        "convergence": "Local minimum (not global)",
        "scalability": "Linear in n, p, k"
    }


@register_model("pca_svd")
def pca_svd(spec: Dict) -> Dict:
    """
    Principal Component Analysis using SVD.
    Time Complexity: O(min(n²·p, n·p²))
    Space Complexity: O(n·p + p²)
    """
    n, p = spec["n"], spec["p"]
    n_components = spec.get("n_components", min(n, p))
    
    # SVD complexity depends on matrix dimensions
    if n > p:
        # Tall matrix: compute X^T X and its eigendecomposition
        flops = n * p**2 + p**3  # X^T X computation + eigendecomposition
        complexity_regime = "p << n (tall matrix)"
    else:
        # Wide matrix: compute X X^T and its eigendecomposition  
        flops = n**2 * p + n**3  # X X^T computation + eigendecomposition
        complexity_regime = "n << p (wide matrix)"
    
    # Principal components transformation
    transform_flops = n * n_components * p
    total_flops = flops + transform_flops
    
    params = n_components * p  # Principal component vectors
    mem = _dataset_mem_bytes(n, p) + n_components * p * BYTES_FP32
    
    return {
        "params": params,
        "train_flops": total_flops,
        "infer_flops_per_sample": n_components * p,
        "dataset_mem_bytes": mem,
        "complexity_class": _get_complexity_class(n, p, "svd"),
        "algorithm": "Singular Value Decomposition",
        "n_components": n_components,
        "complexity_regime": complexity_regime,
        "variance_preservation": "Optimal (captures maximum variance)",
        "orthogonality": "Guaranteed",
        "interpretability": "Medium (linear combinations)",
        "centering_required": True
    }

@register_model("naive_bayes")
def naive_bayes(spec: Dict) -> Dict:
    n, p = spec["n"], spec["p"]
    flops = n * p
    params = p
    mem = _dataset_mem_bytes(n, p)
    return {"params": params, "train_flops": flops, "infer_flops_per_sample": p, "dataset_mem_bytes": mem}
