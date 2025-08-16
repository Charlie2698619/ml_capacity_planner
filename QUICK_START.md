# 🚀 ML Capacity Planner - Quick Start Guide

Welcome to the **ML Capacity Planner**! This tool helps you estimate runtime, resource requirements, and scalability characteristics for machine learning models across different hardware configurations.

## 🎯 What This Tool Does

- **Runtime Estimation**: Calculate training and inference time for your ML models
- **Resource Planning**: Analyze memory, compute, and bandwidth requirements  
- **Bottleneck Analysis**: Identify performance limitations (memory vs compute bound)
- **Scalability Assessment**: Evaluate multi-GPU and distributed training scenarios
- **Cost Optimization**: Compare different hardware options and their trade-offs
- **Hardware Recommendations**: Get suggestions based on your model requirements

## 📦 Installation

### Using UV (Recommended)

```bash
# Clone or navigate to the project directory
cd ml_capacity_planner

# Install with UV
uv sync

# Install in development mode with dev dependencies
uv sync --extra dev
```

### Traditional Pip Installation

```bash
# Install the package
pip install -e .

# Or install dependencies manually
pip install -r requirements.txt
```

## 🚀 Quick Start (2 Minutes)

### 1. Generate Example Configurations

```bash
# Create example configuration files
uv run ml-capacity examples

# This creates:
# examples/linear_regression.yaml    - Classical ML example
# examples/deep_learning.yaml        - Neural network example  
# examples/transformer.yaml          - Large language model example
```

### 2. Run Your First Analysis

```bash
# Analyze a deep learning model
uv run ml-capacity plan examples/deep_learning.yaml

# Generate detailed HTML report
uv run ml-capacity plan examples/transformer.yaml --output report.html --verbose

# JSON output for programmatic access
uv run ml-capacity plan examples/linear_regression.yaml --format json --output results.json
```

### 3. Explore Available Models and Hardware

```bash
# List all supported models
uv run ml-capacity models --list

# List all hardware profiles
uv run ml-capacity hardware --list

# Get details on specific hardware
uv run ml-capacity hardware --details A100_80GB_fp16
```

## 📝 Configuration File Format

ML Capacity Planner uses YAML configuration files with the following structure:

```yaml
# Model configuration
model:
  name: "mlp"                    # Model type (see supported models below)
  spec:                          # Model-specific parameters
    n: 50000                     # Number of training samples
    d_in: 784                    # Input dimensions
    d_out: 10                    # Output dimensions
    layers: [512, 256, 128]      # Hidden layer architecture
    batch_size: 64               # Training batch size
    epochs: 20                   # Number of training epochs
    precision: "fp16"            # Numerical precision (fp32, fp16, bf16, int8)
  optimization: "adamw"          # Optimizer (adam, sgd, rmsprop)
  learning_rate: 0.001          # Learning rate

# Hardware configuration  
hardware:
  name: "RTX_3060_fp16"         # Hardware profile name
  efficiency: 0.4               # Hardware efficiency factor (0.0-1.0)
  parallel_devices: 1           # Number of devices for parallel training

# Scalability analysis (optional)
scalability:
  enabled: true                 # Enable multi-device analysis
  device_counts: [1, 2, 4, 8]   # Device counts to analyze
  batch_sizes: [32, 64, 128]    # Batch sizes to test
  data_parallel: true           # Data parallelism analysis
  model_parallel: false         # Model parallelism analysis

# Report configuration
report:
  format: "markdown"            # Output format (markdown, json, html)
  include_plots: true           # Include performance visualizations
  include_recommendations: true # Include optimization suggestions
  detailed_breakdown: true      # Include detailed FLOP/memory breakdowns
```

## 🧮 Supported Models

### Classical Machine Learning

| Model | Description | Complexity | Best For |
|-------|-------------|------------|----------|
| `linear_regression_normal` | Normal equations | O(p³ + n·p²) | Small p, well-conditioned |
| `linear_regression_qr` | QR decomposition | O(n·p²) | Tall matrices, numerical stability |
| `logistic_regression_newton` | Newton-Raphson | O(k·(n·p² + p³)) | Small datasets, fast convergence |
| `logistic_regression_sgd` | Stochastic gradient descent | O(k·n·p) | Large datasets |
| `svm_linear` | Linear SVM | O(k·n·p) | Text classification, sparse data |
| `svm_rbf` | RBF kernel SVM | O(n²·p + n³) | Small datasets only (⚠️ scales poorly) |
| `decision_tree` | CART algorithm | O(n·p·log(n)) | Interpretable models |
| `random_forest` | Bootstrap aggregation | O(T·n·p·log(n)) | Robust ensemble method |
| `xgboost_like` | Gradient boosting | O(T·n·p·log(n)) | High-performance ML |
| `kmeans` | Lloyd's algorithm | O(k·n·p·i) | Clustering, unsupervised |
| `pca_svd` | Principal components | O(min(n²·p, n·p²)) | Dimensionality reduction |

### Deep Learning

| Model | Description | Complexity | Best For |
|-------|-------------|------------|----------|
| `mlp` | Multi-layer perceptron | O(∑layer_i × layer_{i+1}) | Tabular data, classification |
| `simple_cnn` | Convolutional network | O(∑H×W×C_in×C_out×K²) | Image processing |
| `transformer_encoder` | Transformer encoder | O(L×(seq²×d + seq×d²)) | NLP, sequence modeling |
| `transformer_decoder` | GPT-style decoder | O(L×(seq²×d + seq×d²)) | Text generation |
| `resnet50` | Residual network | O(H×W×C×K²) per layer | Image classification |

## 🖥️ Hardware Profiles

### CPU Profiles

- `CPU_4C_2.5GHz_basic` - Entry-level 4-core CPU
- `CPU_8C_3.5Ghz_baseline` - Mid-range 8-core CPU  
- `CPU_16C_4.0GHz_high_end` - High-end 16-core CPU

### GPU Profiles

- `GTX_1660Ti_fp32` - Budget gaming GPU
- `RTX_3060_fp16` - Mid-range gaming GPU with Tensor Cores
- `RTX_4090_fp16` - High-end consumer GPU
- `A100_40GB_fp16` - Data center GPU (40GB)
- `A100_80GB_fp16` - Data center GPU (80GB)
- `H100_80GB_fp16` - Latest generation data center GPU

### Specialized Hardware

- `TPU_v4_pod` - Google TPU pod for massive scale
- `M1_Pro` - Apple Silicon with unified memory
- `M2_Ultra` - High-end Apple Silicon

## 📊 Understanding the Reports

### Key Metrics Explained

- **Parameters**: Total number of trainable model parameters
- **Training FLOPs**: Total floating-point operations for complete training
- **Training Time**: Estimated wall-clock time for training
- **Memory Usage**: Peak memory requirements during training
- **Bottleneck Analysis**: Whether you're compute-bound or memory-bound
- **Arithmetic Intensity**: FLOPs per byte of memory transferred
- **Scalability**: How well the model scales to multiple devices

### Optimization Recommendations

The tool provides actionable recommendations such as:

- 🚀 Use mixed precision training (FP16/BF16) for faster training
- 💾 Implement gradient checkpointing to reduce memory usage
- ⚡ Enable data parallelism for better hardware utilization
- 📏 Consider sequence length optimizations for transformers
- 💰 Switch to spot instances for cost savings

## 🔧 Advanced Usage

### Custom Hardware Profiles

You can add custom hardware profiles by modifying `ml_capacity_planner/hardware.py`:

```python
register_hardware("my_custom_gpu", {
    "type": "gpu",
    "peak_flops": 50e12,        # Peak FP32 performance
    "peak_flops_fp16": 100e12,  # Peak FP16 performance  
    "mem_gb": 64,               # System memory
    "vram_gb": 24,              # GPU memory
    "bandwidth_gbps": 900,      # Memory bandwidth
    "cost_per_hour": 2.00,      # Cloud cost estimate
    # ... other specifications
})
```

### Custom Model Calculators

Add new models by creating calculators in `ml_capacity_planner/calculators/`:

```python
@register_model("my_custom_model")
def my_custom_model(spec: Dict) -> Dict:
    n, p = spec["n"], spec["p"]
    
    # Calculate FLOPs and parameters
    flops = n * p * math.log(p)  # Example complexity
    params = p * p
    
    return {
        "params": params,
        "train_flops": flops,
        "infer_flops_per_sample": p,
        "complexity_class": "O(n·p·log(p))",
        "algorithm": "My Custom Algorithm"
    }
```

### Batch Analysis

Analyze multiple configurations:

```bash
# Analyze all examples
for config in examples/*.yaml; do
    echo "Analyzing $config"
    uv run ml-capacity plan "$config" --output "reports/$(basename $config .yaml).md"
done
```

### Integration with CI/CD

Use the JSON output for automated analysis:

```bash
# Generate JSON report
uv run ml-capacity plan config.yaml --format json --output results.json

# Extract key metrics programmatically
python -c "
import json
with open('results.json') as f:
    data = json.load(f)
    print(f'Training time: {data[\"computation\"][\"estimated_train_time_human\"]}')
    print(f'Memory usage: {data[\"memory\"][\"utilization_percent\"]:.1f}%')
"
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed with `uv sync`
2. **Configuration Validation Fails**: Use `uv run ml-capacity validate config.yaml` to check syntax
3. **Memory Requirements Exceed Available**: Consider gradient checkpointing or model sharding
4. **Very Long Training Times**: Check if you're using the right hardware profile and efficiency factor

### Getting Help

```bash
# Show help for main command
uv run ml-capacity --help

# Show help for specific subcommand
uv run ml-capacity plan --help

# Validate your configuration
uv run ml-capacity validate your-config.yaml

# Check available models and hardware
uv run ml-capacity models --list
uv run ml-capacity hardware --list
```

## 📈 Performance Tips

### For Classical ML
- Linear algebra operations benefit from optimized BLAS libraries
- Sparse data structures can dramatically reduce memory usage
- Consider approximate algorithms for very large datasets

### For Deep Learning
- Use mixed precision (FP16/BF16) when available
- Gradient accumulation allows larger effective batch sizes
- Data parallelism scales well for most models
- Model parallelism is needed for very large models (>1B parameters)

### Hardware Selection
- Memory bandwidth often limits performance more than compute
- GPU memory is usually the bottleneck for large models
- Consider total cost of ownership, not just raw performance

## 🚀 What's Next?

- Explore the example configurations in the `examples/` directory
- Try analyzing your own models by creating custom configurations
- Experiment with different hardware profiles to find optimal setups
- Use the scalability analysis to plan multi-GPU deployments
- Check out the detailed reports for optimization recommendations

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- Additional model calculators (CNNs, RNNs, etc.)
- More hardware profiles (latest GPUs, TPUs, etc.)  
- Enhanced optimization recommendations
- Visualization and plotting capabilities
- Integration with ML frameworks (PyTorch, TensorFlow)

## 📚 References

- [Roofline Model for Performance Analysis](https://en.wikipedia.org/wiki/Roofline_model)
- [Deep Learning Hardware Guide](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/)
- [Transformer Architecture Analysis](https://arxiv.org/abs/1706.03762)
- [Efficient Training Methods](https://arxiv.org/abs/1910.02054)

Happy capacity planning! 🎯
