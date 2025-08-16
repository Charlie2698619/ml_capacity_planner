
"""Enhanced hardware profiles with detailed specifications and scalability analysis."""

import psutil
from typing import Dict, Any, List
from .registry import register_hardware


def get_system_info() -> Dict[str, Any]:
    """Get current system hardware information."""
    return {
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "cpu_freq": psutil.cpu_freq().max if psutil.cpu_freq() else None,
    }


# Enhanced CPU profiles with more detailed specifications
register_hardware("CPU_4C_2.5GHz_basic", {
    "type": "cpu",
    "cores": 4,
    "threads": 8,
    "base_freq_ghz": 2.5,
    "peak_flops": 80e9,           # 80 GFLOPS effective
    "mem_gb": 16,
    "vram_gb": 0,
    "bandwidth_gbps": 20,         # DDR4-2400
    "cache_mb": 8,
    "tdp_watts": 65,
    "cost_per_hour": 0.05,        # Estimated cloud cost
    "memory_channels": 2,
    "pcie_lanes": 16,
})

register_hardware("CPU_8C_3.5Ghz_baseline", {
    "type": "cpu",
    "cores": 8,
    "threads": 16,
    "base_freq_ghz": 3.5,
    "peak_flops": 120e9,          # 120 GFLOPS effective
    "mem_gb": 32,
    "vram_gb": 0,
    "bandwidth_gbps": 25,         # DDR4-3200
    "cache_mb": 16,
    "tdp_watts": 95,
    "cost_per_hour": 0.10,
    "memory_channels": 2,
    "pcie_lanes": 20,
})

register_hardware("CPU_16C_4.0GHz_high_end", {
    "type": "cpu",
    "cores": 16,
    "threads": 32,
    "base_freq_ghz": 4.0,
    "peak_flops": 300e9,          # 300 GFLOPS effective
    "mem_gb": 64,
    "vram_gb": 0,
    "bandwidth_gbps": 40,         # DDR5-5600
    "cache_mb": 32,
    "tdp_watts": 125,
    "cost_per_hour": 0.25,
    "memory_channels": 4,
    "pcie_lanes": 24,
})

# Enhanced GPU profiles with more comprehensive specs
register_hardware("GTX_1660Ti_fp32", {
    "type": "gpu",
    "architecture": "Turing",
    "compute_capability": "7.5",
    "cuda_cores": 1536,
    "rt_cores": 0,
    "tensor_cores": 0,
    "peak_flops": 5.4e12,         # 5.4 TFLOPS FP32
    "peak_flops_fp16": 10.8e12,   # Simulated FP16
    "mem_gb": 32,                 # System memory
    "vram_gb": 6,
    "bandwidth_gbps": 288,        # GDDR6
    "memory_bus_width": 192,
    "tdp_watts": 120,
    "cost_per_hour": 0.15,
    "nvlink": False,
    "pcie_version": "3.0",
})

register_hardware("RTX_3060_fp16", {
    "type": "gpu",
    "architecture": "Ampere",
    "compute_capability": "8.6",
    "cuda_cores": 3584,
    "rt_cores": 28,
    "tensor_cores": 112,
    "peak_flops": 13e12,          # 13 TFLOPS FP32
    "peak_flops_fp16": 26e12,     # 26 TFLOPS FP16 with Tensor Cores
    "mem_gb": 16,
    "vram_gb": 12,
    "bandwidth_gbps": 360,        # GDDR6
    "memory_bus_width": 192,
    "tdp_watts": 170,
    "cost_per_hour": 0.30,
    "nvlink": False,
    "pcie_version": "4.0",
})

register_hardware("RTX_4090_fp16", {
    "type": "gpu",
    "architecture": "Ada Lovelace",
    "compute_capability": "8.9",
    "cuda_cores": 16384,
    "rt_cores": 128,
    "tensor_cores": 512,
    "peak_flops": 83e12,          # 83 TFLOPS FP32
    "peak_flops_fp16": 166e12,    # 166 TFLOPS FP16 with Tensor Cores
    "mem_gb": 64,
    "vram_gb": 24,
    "bandwidth_gbps": 1008,       # GDDR6X
    "memory_bus_width": 384,
    "tdp_watts": 450,
    "cost_per_hour": 1.20,
    "nvlink": False,
    "pcie_version": "4.0",
})

register_hardware("A100_40GB_fp16", {
    "type": "gpu",
    "architecture": "Ampere",
    "compute_capability": "8.0",
    "cuda_cores": 6912,
    "rt_cores": 0,
    "tensor_cores": 432,
    "peak_flops": 19.5e12,        # 19.5 TFLOPS FP32
    "peak_flops_fp16": 78e12,     # 78 TFLOPS FP16 with sparsity
    "mem_gb": 256,
    "vram_gb": 40,
    "bandwidth_gbps": 1555,       # HBM2e
    "memory_bus_width": 5120,
    "tdp_watts": 400,
    "cost_per_hour": 2.50,
    "nvlink": True,
    "nvlink_bandwidth_gbps": 600,
    "pcie_version": "4.0",
})

register_hardware("A100_80GB_fp16", {
    "type": "gpu",
    "architecture": "Ampere",
    "compute_capability": "8.0",
    "cuda_cores": 6912,
    "rt_cores": 0,
    "tensor_cores": 432,
    "peak_flops": 19.5e12,        # 19.5 TFLOPS FP32
    "peak_flops_fp16": 156e12,    # 156 TFLOPS FP16 with sparsity (conservative)
    "mem_gb": 256,
    "vram_gb": 80,
    "bandwidth_gbps": 1935,       # HBM2e
    "memory_bus_width": 5120,
    "tdp_watts": 400,
    "cost_per_hour": 3.50,
    "nvlink": True,
    "nvlink_bandwidth_gbps": 600,
    "pcie_version": "4.0",
})

register_hardware("H100_80GB_fp16", {
    "type": "gpu",
    "architecture": "Hopper",
    "compute_capability": "9.0",
    "cuda_cores": 16896,
    "rt_cores": 0,
    "tensor_cores": 528,
    "peak_flops": 67e12,          # 67 TFLOPS FP32
    "peak_flops_fp16": 1000e12,   # 1000 TFLOPS FP16 with sparsity
    "mem_gb": 512,
    "vram_gb": 80,
    "bandwidth_gbps": 3350,       # HBM3
    "memory_bus_width": 5120,
    "tdp_watts": 700,
    "cost_per_hour": 8.00,
    "nvlink": True,
    "nvlink_bandwidth_gbps": 900,
    "pcie_version": "5.0",
    "transformer_engine": True,
})

# TPU profiles
register_hardware("TPU_v4_pod", {
    "type": "tpu",
    "version": "v4",
    "cores": 4096,
    "peak_flops": 1.1e15,         # 1.1 ExaFLOPS (BF16)
    "peak_flops_fp16": 2.2e15,    # Theoretical FP16
    "mem_gb": 1400,               # HBM
    "vram_gb": 1400,
    "bandwidth_gbps": 4800,       # Inter-chip bandwidth
    "tdp_watts": 200000,          # Entire pod
    "cost_per_hour": 32.00,
    "interconnect": "2D torus",
    "specialized_for": ["transformers", "convnets"],
})

# Apple Silicon profiles
register_hardware("M1_Pro", {
    "type": "soc",
    "architecture": "Apple Silicon",
    "cpu_cores": 10,
    "gpu_cores": 16,
    "neural_engine_tops": 15.8,
    "peak_flops": 5.2e12,         # Combined CPU+GPU
    "mem_gb": 32,                 # Unified memory
    "vram_gb": 32,                # Shared with system
    "bandwidth_gbps": 200,        # Unified memory bandwidth
    "tdp_watts": 30,
    "cost_per_hour": 0.08,
    "unified_memory": True,
})

register_hardware("M2_Ultra", {
    "type": "soc",
    "architecture": "Apple Silicon",
    "cpu_cores": 24,
    "gpu_cores": 76,
    "neural_engine_tops": 31.6,
    "peak_flops": 27.2e12,        # Combined CPU+GPU
    "mem_gb": 192,                # Unified memory
    "vram_gb": 192,               # Shared with system
    "bandwidth_gbps": 800,        # Unified memory bandwidth
    "tdp_watts": 215,
    "cost_per_hour": 0.50,
    "unified_memory": True,
})


def calculate_multi_gpu_specs(base_profile: Dict[str, Any], num_gpus: int) -> Dict[str, Any]:
    """Calculate specifications for multi-GPU setup."""
    if base_profile["type"] != "gpu":
        raise ValueError("Multi-GPU calculation only supports GPU profiles")
    
    multi_profile = base_profile.copy()
    multi_profile["num_devices"] = num_gpus
    multi_profile["peak_flops"] *= num_gpus
    multi_profile["peak_flops_fp16"] *= num_gpus
    multi_profile["vram_gb"] *= num_gpus  # Total VRAM across all GPUs
    multi_profile["cost_per_hour"] *= num_gpus
    multi_profile["tdp_watts"] *= num_gpus
    
    # Communication overhead factor (conservative estimate)
    communication_efficiency = max(0.7, 1.0 - 0.05 * (num_gpus - 1))
    multi_profile["communication_efficiency"] = communication_efficiency
    
    return multi_profile


def get_hardware_recommendations(
    model_memory_gb: float,
    compute_flops: float,
    budget_per_hour: float = None,
    power_limit_watts: float = None
) -> List[Dict[str, Any]]:
    """Get hardware recommendations based on requirements."""
    from .registry import HARDWARE_PROFILES
    
    recommendations = []
    
    for name, profile in HARDWARE_PROFILES.items():
        # Check memory requirements
        available_memory = profile.get("vram_gb", profile.get("mem_gb", 0))
        if available_memory < model_memory_gb:
            continue
            
        # Check budget constraints
        if budget_per_hour and profile.get("cost_per_hour", 0) > budget_per_hour:
            continue
            
        # Check power constraints
        if power_limit_watts and profile.get("tdp_watts", 0) > power_limit_watts:
            continue
        
        # Calculate performance score
        peak_flops = profile.get("peak_flops_fp16", profile.get("peak_flops", 0))
        memory_score = available_memory / model_memory_gb if model_memory_gb > 0 else 1
        compute_score = peak_flops / compute_flops if compute_flops > 0 else 1
        cost_efficiency = peak_flops / profile.get("cost_per_hour", 1) if profile.get("cost_per_hour") else 0
        
        recommendations.append({
            "name": name,
            "profile": profile,
            "memory_score": memory_score,
            "compute_score": compute_score,
            "cost_efficiency": cost_efficiency,
            "overall_score": (memory_score + compute_score + cost_efficiency) / 3
        })
    
    # Sort by overall score
    recommendations.sort(key=lambda x: x["overall_score"], reverse=True)
    return recommendations[:5]  # Top 5 recommendations


def get_hardware_profile(name: str) -> Dict[str, Any]:
    """Get a specific hardware profile by name."""
    from .registry import HARDWARE_PROFILES
    
    if name not in HARDWARE_PROFILES:
        raise ValueError(f"Hardware profile '{name}' not found. Available profiles: {list(HARDWARE_PROFILES.keys())}")
    
    return HARDWARE_PROFILES[name]


def list_hardware_profiles() -> List[str]:
    """List all available hardware profile names."""
    from .registry import HARDWARE_PROFILES
    return list(HARDWARE_PROFILES.keys())
