
"""Enhanced deep learning models with accurate FLOP calculations and memory analysis."""

from typing import Dict, Tuple
import math
from ..registry import register_model

BYTES_FP16 = 2
BYTES_FP32 = 4
BYTES_BF16 = 2
BYTES_INT8 = 1
BYTES_INT4 = 0.5


def get_dtype_bytes(precision: str) -> float:
    """Get bytes per parameter for different precisions."""
    precision_map = {
        "fp32": BYTES_FP32,
        "fp16": BYTES_FP16, 
        "bf16": BYTES_BF16,
        "int8": BYTES_INT8,
        "int4": BYTES_INT4
    }
    return precision_map.get(precision.lower(), BYTES_FP32)


def get_optimizer_multiplier(optimizer: str, dtype_bytes: float) -> float:
    """Get memory multiplier for different optimizers."""
    # Parameter memory multipliers for different optimizers
    if optimizer.lower() in ["adam", "adamw"]:
        # params + momentum + variance (+ master weights for mixed precision)
        return 3.0 if dtype_bytes < BYTES_FP32 else 2.0
    elif optimizer.lower() == "sgd":
        # params + momentum
        return 2.0 if dtype_bytes < BYTES_FP32 else 1.5
    elif optimizer.lower() == "rmsprop":
        # params + variance (+ master weights for mixed precision)
        return 2.5 if dtype_bytes < BYTES_FP32 else 2.0
    else:
        return 2.0  # Default conservative estimate


def calculate_attention_flops(seq_len: int, d_model: int, n_heads: int) -> int:
    """Calculate FLOPs for multi-head attention mechanism."""
    d_k = d_model // n_heads
    
    # Q, K, V projections: 3 * seq_len * d_model * d_model
    qkv_proj = 3 * seq_len * d_model * d_model
    
    # Attention scores: seq_len * seq_len * d_k * n_heads  
    attention_scores = seq_len * seq_len * d_k * n_heads
    
    # Attention weights * values: seq_len * seq_len * d_k * n_heads
    attention_values = seq_len * seq_len * d_k * n_heads
    
    # Output projection: seq_len * d_model * d_model
    output_proj = seq_len * d_model * d_model
    
    return qkv_proj + attention_scores + attention_values + output_proj


def calculate_ffn_flops(seq_len: int, d_model: int, d_ff: int = None) -> int:
    """Calculate FLOPs for feed-forward network."""
    if d_ff is None:
        d_ff = 4 * d_model  # Standard transformer FFN expansion
    
    # Two linear layers: up-projection and down-projection
    up_proj = seq_len * d_model * d_ff
    down_proj = seq_len * d_ff * d_model
    
    return up_proj + down_proj


@register_model("mlp")
def mlp(spec: Dict) -> Dict:
    """
    Multi-Layer Perceptron with detailed analysis.
    Time Complexity: O(batch_size * sum(layer_i * layer_{i+1}))
    """
    n = spec.get("n", 10000)
    d_in = spec["d_in"]
    layers = spec.get("layers", [128, 128])
    d_out = spec.get("d_out", 1)
    epochs = spec.get("epochs", 20)
    batch_size = spec.get("batch_size", 64)
    precision = spec.get("precision", "fp32")
    optimizer = spec.get("optimizer", "adam")
    dropout = spec.get("dropout", 0.0)
    
    dtype_bytes = get_dtype_bytes(precision)
    
    # Network architecture
    dims = [d_in] + layers + [d_out]
    
    # Parameter counting
    weights = sum(dims[i] * dims[i+1] for i in range(len(dims)-1))
    biases = sum(dims[1:])  # No bias for input layer
    total_params = weights + biases
    
    # FLOP calculation (per sample)
    # Forward pass: 2 * weight_operations (MAC = 2 FLOPs)
    forward_flops = 2 * weights
    
    # Backward pass: roughly 2x forward (gradient computation + weight updates)
    backward_flops = 2 * forward_flops
    
    flops_per_sample = forward_flops + backward_flops
    
    # Training computation
    steps_per_epoch = math.ceil(n / batch_size)
    total_steps = steps_per_epoch * epochs
    train_flops = flops_per_sample * batch_size * total_steps
    
    # Memory analysis
    optimizer_mult = get_optimizer_multiplier(optimizer, dtype_bytes)
    param_mem = total_params * dtype_bytes * optimizer_mult
    
    # Activation memory (layer outputs during forward/backward)
    max_layer_size = max(dims)
    activation_mem = max_layer_size * batch_size * dtype_bytes * 2  # Forward + backward
    
    # Additional memory for intermediate computations
    gradient_mem = total_params * dtype_bytes
    total_mem = param_mem + activation_mem + gradient_mem
    
    return {
        "params": total_params,
        "train_flops": train_flops,
        "infer_flops_per_sample": forward_flops,
        "model_mem_bytes": total_mem,
        "activation_mem_bytes": activation_mem,
        "complexity_class": "O(∑(layer_i × layer_{i+1}))",
        "algorithm": "Backpropagation",
        "architecture": dims,
        "optimizer": optimizer,
        "precision": precision,
        "dropout_rate": dropout,
        "parallelizable": "Data parallel, limited model parallel",
        "memory_breakdown": {
            "parameters": param_mem,
            "activations": activation_mem,
            "gradients": gradient_mem,
            "optimizer_states": (optimizer_mult - 1) * total_params * dtype_bytes
        },
        "flop_breakdown": {
            "forward": forward_flops,
            "backward": backward_flops,
            "total_per_sample": flops_per_sample
        }
    }


@register_model("simple_cnn")
def simple_cnn(spec: Dict) -> Dict:
    """
    Convolutional Neural Network with detailed convolution analysis.
    """
    n = spec.get("n", 50000)
    H, W = spec.get("image_h", 224), spec.get("image_w", 224)
    c_in = spec.get("c_in", 3)
    channels = spec.get("channels", [64, 128, 256])
    kernel_size = spec.get("kernel", 3)
    stride = spec.get("stride", 2)
    padding = spec.get("padding", 1)
    epochs = spec.get("epochs", 10)
    batch_size = spec.get("batch_size", 64)
    precision = spec.get("precision", "fp32")
    optimizer = spec.get("optimizer", "adam")
    
    dtype_bytes = get_dtype_bytes(precision)
    
    # Track feature map dimensions and compute FLOPs
    h, w, c = H, W, c_in
    total_params = 0
    total_conv_flops = 0
    layer_info = []
    
    for i, c_out in enumerate(channels):
        # Convolution parameters
        conv_params = (kernel_size * kernel_size * c * c_out) + c_out  # weights + bias
        total_params += conv_params
        
        # Output dimensions (assuming same padding for simplicity)
        h_out = (h + 2 * padding - kernel_size) // stride + 1
        w_out = (w + 2 * padding - kernel_size) // stride + 1
        
        # Convolution FLOPs: output_size * kernel_ops
        conv_flops = h_out * w_out * c_out * (kernel_size * kernel_size * c * 2)  # 2 for MAC
        total_conv_flops += conv_flops
        
        layer_info.append({
            "layer": f"conv_{i+1}",
            "input_shape": (h, w, c),
            "output_shape": (h_out, w_out, c_out),
            "params": conv_params,
            "flops": conv_flops
        })
        
        # Update for next layer
        h, w, c = h_out, w_out, c_out
    
    # Classifier head (global average pooling + fully connected)
    pooled_features = c  # After global average pooling
    num_classes = spec.get("num_classes", 1000)
    classifier_params = pooled_features * num_classes + num_classes
    classifier_flops = pooled_features * num_classes * 2  # 2 for MAC
    
    total_params += classifier_params
    forward_flops = total_conv_flops + classifier_flops
    
    # Training computation
    backward_flops = 2 * forward_flops  # Approximate
    flops_per_sample = forward_flops + backward_flops
    
    steps_per_epoch = math.ceil(n / batch_size)
    train_flops = flops_per_sample * batch_size * steps_per_epoch * epochs
    
    # Memory analysis
    optimizer_mult = get_optimizer_multiplier(optimizer, dtype_bytes)
    param_mem = total_params * dtype_bytes * optimizer_mult
    
    # Feature map memory (largest intermediate representation)
    max_feature_mem = max(h * w * ch * batch_size * dtype_bytes 
                         for h, w, ch in [(layer["output_shape"]) for layer in layer_info])
    activation_mem = max_feature_mem + H * W * c_in * batch_size * dtype_bytes  # Input + largest feature map
    
    return {
        "params": total_params,
        "train_flops": train_flops,
        "infer_flops_per_sample": forward_flops,
        "model_mem_bytes": param_mem + activation_mem,
        "activation_mem_bytes": activation_mem,
        "complexity_class": "O(∑(H_i × W_i × C_in × C_out × K²))",
        "algorithm": "Convolutional Neural Network",
        "architecture": {
            "input_shape": (H, W, c_in),
            "conv_layers": layer_info,
            "classifier_params": classifier_params
        },
        "precision": precision,
        "optimizer": optimizer,
        "receptive_field": kernel_size + (len(channels) - 1) * (kernel_size - 1) * stride,
        "parameter_breakdown": {
            "convolution": total_params - classifier_params,
            "classifier": classifier_params
        },
        "flop_breakdown": {
            "convolution": total_conv_flops,
            "classifier": classifier_flops,
            "forward_total": forward_flops
        }
    }


@register_model("transformer_encoder")
def transformer_encoder(spec: Dict) -> Dict:
    """
    Transformer Encoder with detailed attention and FFN analysis.
    """
    L = spec.get("n_layers", 12)
    d_model = spec.get("d_model", 768)
    n_heads = spec.get("n_heads", 12)
    d_ff = spec.get("d_ff", 4 * d_model)
    seq_len = spec.get("seq_len", 512)
    vocab_size = spec.get("vocab_size", 30000)
    n_tokens = spec.get("n_tokens", 10_000_000)
    epochs = spec.get("epochs", 1)
    batch_size = spec.get("batch_size", 32)
    precision = spec.get("precision", "fp16")
    optimizer = spec.get("optimizer", "adamw")
    
    dtype_bytes = get_dtype_bytes(precision)
    
    # Parameter counting
    # Embedding layer
    embedding_params = vocab_size * d_model + seq_len * d_model  # Token + positional embeddings
    
    # Transformer layers
    per_layer_params = (
        4 * d_model * d_model +  # Q, K, V, O projections
        2 * d_model * d_ff +     # FFN layers
        4 * d_model              # Layer norm parameters (2 per layer)
    )
    transformer_params = L * per_layer_params
    
    # Output layer (language modeling head)
    output_params = d_model * vocab_size
    
    total_params = embedding_params + transformer_params + output_params
    
    # FLOP calculation per token
    attention_flops = calculate_attention_flops(seq_len, d_model, n_heads)
    ffn_flops = calculate_ffn_flops(seq_len, d_model, d_ff)
    layer_norm_flops = seq_len * d_model * 2  # 2 layer norms per layer
    
    per_layer_flops = attention_flops + ffn_flops + layer_norm_flops
    per_token_flops = L * per_layer_flops + (vocab_size * d_model * 2)  # + output projection
    
    # Training computation
    forward_flops = per_token_flops
    backward_flops = 2 * forward_flops  # Approximate
    total_per_token_flops = forward_flops + backward_flops
    
    train_flops = total_per_token_flops * n_tokens * epochs
    
    # Memory analysis
    optimizer_mult = get_optimizer_multiplier(optimizer, dtype_bytes)
    param_mem = total_params * dtype_bytes * optimizer_mult
    
    # Attention memory (key-value cache + attention scores)
    kv_cache_mem = 2 * L * seq_len * d_model * batch_size * dtype_bytes  # K, V for all layers
    attention_scores_mem = L * n_heads * seq_len * seq_len * batch_size * dtype_bytes
    
    # Activation memory (intermediate representations)
    activation_mem = seq_len * d_model * batch_size * dtype_bytes * L * 2  # Forward + backward
    
    total_activation_mem = kv_cache_mem + attention_scores_mem + activation_mem
    
    return {
        "params": total_params,
        "train_flops": train_flops,
        "infer_flops_per_token": forward_flops // seq_len,  # Per token inference
        "model_mem_bytes": param_mem + total_activation_mem,
        "activation_mem_bytes": total_activation_mem,
        "complexity_class": "O(L × (seq_len² × d_model + seq_len × d_model²))",
        "algorithm": "Transformer (Self-Attention)",
        "architecture": {
            "n_layers": L,
            "d_model": d_model,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "seq_len": seq_len,
            "vocab_size": vocab_size
        },
        "precision": precision,
        "optimizer": optimizer,
        "attention_pattern": "O(seq_len²) - quadratic scaling bottleneck",
        "parallelizable": "Excellent for data parallel, good for model parallel",
        "parameter_breakdown": {
            "embeddings": embedding_params,
            "transformer_layers": transformer_params,
            "output_head": output_params
        },
        "flop_breakdown": {
            "attention_per_layer": attention_flops,
            "ffn_per_layer": ffn_flops,
            "layer_norm_per_layer": layer_norm_flops,
            "total_per_token": per_token_flops
        },
        "memory_breakdown": {
            "parameters": param_mem,
            "kv_cache": kv_cache_mem,
            "attention_scores": attention_scores_mem,
            "activations": activation_mem
        }
    }


@register_model("transformer_decoder") 
def transformer_decoder(spec: Dict) -> Dict:
    """
    Transformer Decoder (GPT-style) for autoregressive generation.
    """
    # Similar to encoder but with causal attention and slightly different structure
    encoder_result = transformer_encoder(spec)
    
    # Modify for decoder-specific characteristics
    seq_len = spec.get("seq_len", 512)
    n_heads = spec.get("n_heads", 12)
    d_model = spec.get("d_model", 768)
    L = spec.get("n_layers", 12)
    
    # Causal attention has roughly half the computation due to masking
    causal_attention_savings = 0.5
    attention_flops = encoder_result["flop_breakdown"]["attention_per_layer"] * causal_attention_savings
    
    # Update the result
    encoder_result.update({
        "algorithm": "Transformer Decoder (GPT-style)",
        "attention_pattern": "O(seq_len²) - causal masked attention",
        "generation_flops_per_token": encoder_result["infer_flops_per_token"],
        "autoregressive": True,
        "flop_breakdown": {
            **encoder_result["flop_breakdown"],
            "attention_per_layer": attention_flops
        }
    })
    
    # Adjust total FLOPs for causal masking
    encoder_result["train_flops"] *= causal_attention_savings
    encoder_result["infer_flops_per_token"] *= causal_attention_savings
    
    return encoder_result


@register_model("resnet50")
def resnet50(spec: Dict) -> Dict:
    """
    ResNet-50 architecture with residual connections.
    """
    # ResNet-50 specific architecture
    n = spec.get("n", 1000000)  # ImageNet size
    input_shape = (224, 224, 3)
    num_classes = spec.get("num_classes", 1000)
    batch_size = spec.get("batch_size", 64)
    epochs = spec.get("epochs", 90)
    precision = spec.get("precision", "fp32")
    optimizer = spec.get("optimizer", "sgd")
    
    dtype_bytes = get_dtype_bytes(precision)
    
    # ResNet-50 has ~25.6M parameters
    total_params = 25_600_000
    
    # FLOPs for ResNet-50 inference (well-established)
    forward_flops = 4.1e9  # 4.1 GFLOPs for single inference
    
    # Training computation
    backward_flops = 2 * forward_flops
    flops_per_sample = forward_flops + backward_flops
    
    steps_per_epoch = math.ceil(n / batch_size)
    train_flops = flops_per_sample * batch_size * steps_per_epoch * epochs
    
    # Memory analysis
    optimizer_mult = get_optimizer_multiplier(optimizer, dtype_bytes)
    param_mem = total_params * dtype_bytes * optimizer_mult
    
    # Peak activation memory occurs in early layers (large feature maps)
    peak_activation_mem = 224 * 224 * 64 * batch_size * dtype_bytes  # First conv output
    
    return {
        "params": total_params,
        "train_flops": train_flops,
        "infer_flops_per_sample": forward_flops,
        "model_mem_bytes": param_mem + peak_activation_mem,
        "activation_mem_bytes": peak_activation_mem,
        "complexity_class": "O(H × W × C × K²) per layer",
        "algorithm": "Residual Neural Network",
        "architecture": "ResNet-50",
        "residual_connections": True,
        "skip_connections": "Identity mappings",
        "depth": 50,
        "precision": precision,
        "optimizer": optimizer,
        "batch_normalization": True,
        "regularization": "Batch norm + Weight decay"
    }
