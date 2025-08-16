## Model

- **Name**: transformer_encoder
- **Algorithm**: Transformer (Self-Attention)
- **Parameters**: 131444736
- **Complexity Class**: O(L × (seq_len² × d_model + seq_len × d_model²))
- **Architecture**: {'n_layers': 12, 'd_model': 768, 'n_heads': 12, 'd_ff': 3072, 'seq_len': 512, 'vocab_size': 30000}

## Computation

- **Train Flops**: 145121697792000000000
- **Train Flops Human**: 145.12 EFLOPs
- **Infer Flops Per Sample**: 0
- **Infer Flops Per Token**: 94480272
- **Estimated Train Time Seconds**: 14884276.696615385
- **Estimated Train Time Human**: 172.27 days

## Memory

- **Model Bytes**: 4412547072.0
- **Activation Bytes**: 3623878656
- **Dataset Bytes**: 0
- **Total Required Bytes**: 8036425728.0
- **Available Bytes**: 80000000000.0
- **Utilization Percent**: 10.045532159999999

## Hardware

- **Name**: Unknown
- **Type**: gpu
- **Peak Flops**: 19500000000000.0
- **Memory Gb**: 80
- **Bandwidth Gbps**: 1935
- **Cost Per Hour**: 3.5
- **Efficiency Factor**: 0.5

## Analysis

- **Bottlenecks**: {'primary': 'compute', 'reason': 'High arithmetic intensity (9028995146.83 FLOP/byte), compute-bound', 'memory_utilization': 0.1004553216, 'arithmetic_intensity': 9028995146.833515, 'machine_balance': 10.077519379844961, 'memory_bound': False}
- **Recommendations**: ['⚡ Compute bottleneck detected. Consider:', '   • Multi-GPU data parallelism', '   • Mixed precision training for higher throughput', '   • Model parallelism for very large models']
- **Flop Breakdown**: {'attention_per_layer': 1610612736, 'ffn_per_layer': 2415919104, 'layer_norm_per_layer': 786432, 'total_per_token': 48373899264}
- **Memory Breakdown**: {'parameters': 788668416.0, 'kv_cache': 603979776, 'attention_scores': 2415919104, 'activations': 603979776}

## Metadata

- **Config**: {'model': {'name': 'transformer_encoder', 'spec': {'batch_size': 32, 'd_model': 768, 'epochs': 1, 'n_heads': 12, 'n_layers': 12, 'n_tokens': 1000000000, 'precision': 'fp16', 'seq_len': 512, 'vocab_size': 30000}, 'optimization': <OptimizationType.ADAM: 'adam'>, 'learning_rate': 0.001}, 'hardware': {'name': 'A100_80GB_fp16', 'efficiency': 0.5, 'parallel_devices': 8, 'memory_limit_gb': None}, 'scalability': {'enabled': True, 'device_counts': [1, 2, 4, 8], 'batch_sizes': [32, 64, 128, 256], 'data_parallel': True, 'model_parallel': True}, 'report': {'format': 'markdown', 'include_plots': True, 'include_recommendations': True, 'detailed_breakdown': False}, 'timestamp': '2025-08-16T10:49:55.365199'}
- **Timestamp**: 2025-08-16T10:49:55.365199

## Scalability

- **Device Counts**: [1, 2, 4, 8]
- **Estimated Times**: [29768553.39323077, 16372704.366276924, 9674779.8528, 6325817.59606154]
- **Speedups**: [1.0, 1.8181818181818181, 3.0769230769230766, 4.705882352941176]
- **Efficiency**: [1.0, 0.9090909090909091, 0.7692307692307692, 0.588235294117647]
- **Cost Analysis**: [28941.649132307695, 31835.814045538464, 37624.143872, 49200.80352492309]
