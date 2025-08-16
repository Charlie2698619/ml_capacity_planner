
"""Enhanced reporting module with comprehensive analysis and visualizations."""

import math
from typing import Dict, Any, List, Optional
from pathlib import Path
import json


def bytes_to_human(b: int) -> str:
    """Convert bytes to human-readable format."""
    if b == 0:
        return "0 B"
    
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = abs(b)
    i = int(math.floor(math.log(size, 1024)))
    i = min(i, len(units) - 1)
    
    size = size / (1024 ** i)
    return f"{size:.2f} {units[i]}"


def flops_to_human(f: float) -> str:
    """Convert FLOPs to human-readable format."""
    if f == 0:
        return "0 FLOPs"
    
    units = ["FLOPs", "KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs", "PFLOPs", "EFLOPs"]
    size = abs(f)
    i = int(math.floor(math.log(size, 1000)))
    i = min(i, len(units) - 1)
    
    size = size / (1000 ** i)
    return f"{size:.2f} {units[i]}"


def time_to_human(seconds: float) -> str:
    """Convert seconds to human-readable time format."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.2f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.2f} hours"
    else:
        return f"{seconds/86400:.2f} days"


def est_time_seconds(flops: float, peak_flops: float, efficiency: float = 0.25) -> float:
    """Estimate execution time given FLOP requirements and hardware specs."""
    eff_peak = peak_flops * efficiency
    if eff_peak <= 0:
        return float("inf")
    return flops / eff_peak


def calculate_bottlenecks(calc: Dict[str, Any], hw: Dict[str, Any], efficiency: float) -> Dict[str, Any]:
    """Analyze potential bottlenecks in the computation."""
    bottlenecks = {}
    
    # Memory bandwidth bottleneck
    model_mem = calc.get("model_mem_bytes", 0)
    activation_mem = calc.get("activation_mem_bytes", 0)
    total_mem_required = model_mem + activation_mem
    
    available_memory = hw.get("vram_gb", hw.get("mem_gb", 0)) * 1e9
    memory_utilization = total_mem_required / available_memory if available_memory > 0 else float('inf')
    
    # Compute intensity analysis
    train_flops = calc.get("train_flops", 0)
    memory_transfers = total_mem_required * 2  # Read + write
    arithmetic_intensity = train_flops / memory_transfers if memory_transfers > 0 else float('inf')
    
    # Roofline analysis
    peak_flops = hw.get("peak_flops", 0)
    bandwidth_bps = hw.get("bandwidth_gbps", 0) * 1e9
    machine_balance = peak_flops / bandwidth_bps if bandwidth_bps > 0 else float('inf')
    
    if arithmetic_intensity < machine_balance:
        bottlenecks["primary"] = "memory_bandwidth"
        bottlenecks["reason"] = f"Low arithmetic intensity ({arithmetic_intensity:.2f} FLOP/byte) vs machine balance ({machine_balance:.2f})"
    else:
        bottlenecks["primary"] = "compute"
        bottlenecks["reason"] = f"High arithmetic intensity ({arithmetic_intensity:.2f} FLOP/byte), compute-bound"
    
    bottlenecks.update({
        "memory_utilization": memory_utilization,
        "arithmetic_intensity": arithmetic_intensity,
        "machine_balance": machine_balance,
        "memory_bound": arithmetic_intensity < machine_balance
    })
    
    return bottlenecks


def generate_recommendations(calc: Dict[str, Any], hw: Dict[str, Any], bottlenecks: Dict[str, Any]) -> List[str]:
    """Generate optimization recommendations based on analysis."""
    recommendations = []
    
    # Memory recommendations
    if bottlenecks["memory_utilization"] > 0.9:
        recommendations.append("⚠️  Memory usage > 90%. Consider gradient checkpointing or model sharding.")
    elif bottlenecks["memory_utilization"] > 0.7:
        recommendations.append("💡 Memory usage > 70%. Consider mixed precision training.")
    
    # Bottleneck-specific recommendations
    if bottlenecks["primary"] == "memory_bandwidth":
        recommendations.extend([
            "🚀 Memory bandwidth bottleneck detected. Consider:",
            "   • Increasing batch size to improve memory utilization",
            "   • Using tensor cores (if available) for higher arithmetic intensity",
            "   • Implementing gradient accumulation to reduce memory transfers"
        ])
    else:
        recommendations.extend([
            "⚡ Compute bottleneck detected. Consider:",
            "   • Multi-GPU data parallelism",
            "   • Mixed precision training for higher throughput",
            "   • Model parallelism for very large models"
        ])
    
    # Model-specific recommendations
    algorithm = calc.get("algorithm", "")
    if "SVM" in algorithm and "RBF" in algorithm:
        n = calc.get("n", 0)
        if n > 10000:
            recommendations.append("⚠️  RBF SVM with O(n³) scaling. Consider linear SVM or kernel approximations.")
    
    if "Transformer" in algorithm:
        seq_len = calc.get("seq_len", 0)
        if seq_len > 1024:
            recommendations.append("📏 Long sequences detected. Consider sliding window attention or sparse attention patterns.")
    
    # Hardware-specific recommendations
    if hw.get("type") == "gpu" and hw.get("tensor_cores", 0) > 0:
        precision = calc.get("precision", "fp32")
        if precision == "fp32":
            recommendations.append("🔧 GPU has Tensor Cores. Consider FP16/BF16 for significant speedup.")
    
    # Cost optimization
    cost_per_hour = hw.get("cost_per_hour", 0)
    train_time_hours = bottlenecks.get("estimated_time", 0) / 3600
    if cost_per_hour > 0 and train_time_hours > 0:
        total_cost = cost_per_hour * train_time_hours
        if total_cost > 100:
            recommendations.append(f"💰 Estimated cost: ${total_cost:.2f}. Consider spot instances or model compression.")
    
    return recommendations


def create_scalability_analysis(calc: Dict[str, Any], hw: Dict[str, Any], device_counts: List[int]) -> Dict[str, Any]:
    """Analyze scalability across multiple devices."""
    base_time = est_time_seconds(calc.get("train_flops", 0), hw.get("peak_flops", 1), 0.25)
    
    scalability = {
        "device_counts": device_counts,
        "estimated_times": [],
        "speedups": [],
        "efficiency": [],
        "cost_analysis": []
    }
    
    for devices in device_counts:
        # Communication overhead (simplified model)
        if devices == 1:
            comm_overhead = 1.0
        else:
            # Assume 10% overhead per additional device (conservative)
            comm_overhead = 1.0 + 0.1 * (devices - 1)
        
        # Parallel efficiency
        parallel_efficiency = 1.0 / comm_overhead
        effective_speedup = devices * parallel_efficiency
        scaled_time = base_time / effective_speedup
        
        scalability["estimated_times"].append(scaled_time)
        scalability["speedups"].append(effective_speedup)
        scalability["efficiency"].append(parallel_efficiency)
        
        # Cost analysis
        cost_per_hour = hw.get("cost_per_hour", 0)
        if cost_per_hour > 0:
            total_cost = devices * cost_per_hour * (scaled_time / 3600)
            scalability["cost_analysis"].append(total_cost)
    
    return scalability


def summarize(
    model_name: str, 
    calc: Dict[str, Any], 
    hw: Dict[str, Any], 
    efficiency: float = 0.25,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """Generate comprehensive capacity planning report."""
    lines = []
    
    # Header
    lines.append(f"# 🚀 ML Capacity Planning Report: {model_name}")
    lines.append("")
    lines.append(f"**Generated on**: {config.get('timestamp', 'N/A') if config else 'N/A'}")
    lines.append(f"**Hardware Profile**: {hw.get('type', 'unknown').upper()} - {hw.get('architecture', 'N/A')}")
    lines.append("")
    
    # Model Overview
    lines.append("## 📊 Model Overview")
    lines.append("")
    lines.append(f"- **Algorithm**: {calc.get('algorithm', 'N/A')}")
    lines.append(f"- **Parameters**: {calc.get('params', 0):,}")
    lines.append(f"- **Complexity Class**: {calc.get('complexity_class', 'N/A')}")
    
    if "architecture" in calc:
        arch = calc["architecture"]
        if isinstance(arch, dict):
            lines.append("- **Architecture Details**:")
            for key, value in arch.items():
                lines.append(f"  - {key.replace('_', ' ').title()}: {value}")
        else:
            lines.append(f"- **Architecture**: {arch}")
    
    lines.append("")
    
    # Computational Requirements
    lines.append("## ⚡ Computational Requirements")
    lines.append("")
    
    if "train_flops" in calc:
        lines.append(f"- **Training FLOPs**: {flops_to_human(calc['train_flops'])}")
        train_time = est_time_seconds(calc["train_flops"], hw.get("peak_flops", 1), efficiency)
        lines.append(f"- **Estimated Training Time**: {time_to_human(train_time)} (efficiency={efficiency:.0%})")
    
    if "infer_flops_per_sample" in calc:
        lines.append(f"- **Inference FLOPs/sample**: {flops_to_human(calc['infer_flops_per_sample'])}")
    
    if "infer_flops_per_token" in calc:
        lines.append(f"- **Inference FLOPs/token**: {flops_to_human(calc['infer_flops_per_token'])}")
    
    lines.append("")
    
    # Memory Requirements
    lines.append("## 💾 Memory Requirements")
    lines.append("")
    
    if "model_mem_bytes" in calc:
        lines.append(f"- **Model + Optimizer Memory**: {bytes_to_human(calc['model_mem_bytes'])}")
    
    if "activation_mem_bytes" in calc:
        lines.append(f"- **Activation Memory**: {bytes_to_human(calc['activation_mem_bytes'])}")
    
    if "dataset_mem_bytes" in calc:
        lines.append(f"- **Dataset Memory**: {bytes_to_human(calc['dataset_mem_bytes'])}")
    
    # Total memory and utilization
    total_memory_required = (
        calc.get("model_mem_bytes", 0) + 
        calc.get("activation_mem_bytes", 0) + 
        calc.get("dataset_mem_bytes", 0)
    )
    available_memory = hw.get("vram_gb", hw.get("mem_gb", 0)) * 1e9
    
    lines.append(f"- **Total Memory Required**: {bytes_to_human(total_memory_required)}")
    lines.append(f"- **Available Memory**: {bytes_to_human(available_memory)}")
    
    if available_memory > 0:
        utilization = total_memory_required / available_memory * 100
        status = "✅" if utilization < 70 else "⚠️" if utilization < 90 else "❌"
        lines.append(f"- **Memory Utilization**: {utilization:.1f}% {status}")
    
    lines.append("")
    
    # Hardware Specifications
    lines.append("## 🖥️ Hardware Specifications")
    lines.append("")
    lines.append(f"- **Type**: {hw.get('type', 'N/A').upper()}")
    lines.append(f"- **Peak Performance**: {flops_to_human(hw.get('peak_flops', 0))}")
    lines.append(f"- **Memory**: {hw.get('vram_gb', hw.get('mem_gb', 0))} GB")
    lines.append(f"- **Bandwidth**: {hw.get('bandwidth_gbps', 0)} GB/s")
    
    if hw.get("cost_per_hour"):
        lines.append(f"- **Cost**: ${hw['cost_per_hour']:.2f}/hour")
    
    if hw.get("tdp_watts"):
        lines.append(f"- **Power**: {hw['tdp_watts']} W")
    
    lines.append("")
    
    # Bottleneck Analysis
    bottlenecks = calculate_bottlenecks(calc, hw, efficiency)
    lines.append("## 🔍 Bottleneck Analysis")
    lines.append("")
    lines.append(f"- **Primary Bottleneck**: {bottlenecks['primary'].replace('_', ' ').title()}")
    lines.append(f"- **Reason**: {bottlenecks['reason']}")
    lines.append(f"- **Arithmetic Intensity**: {bottlenecks['arithmetic_intensity']:.2f} FLOP/byte")
    lines.append(f"- **Machine Balance**: {bottlenecks['machine_balance']:.2f} FLOP/byte")
    
    bound_type = "Memory-bound" if bottlenecks["memory_bound"] else "Compute-bound"
    lines.append(f"- **Performance Characteristic**: {bound_type}")
    lines.append("")
    
    # Detailed Breakdowns
    if "flop_breakdown" in calc:
        lines.append("## 📈 FLOP Breakdown")
        lines.append("")
        for component, flops in calc["flop_breakdown"].items():
            if isinstance(flops, (int, float)):
                lines.append(f"- **{component.replace('_', ' ').title()}**: {flops_to_human(flops)}")
        lines.append("")
    
    if "memory_breakdown" in calc:
        lines.append("## 💿 Memory Breakdown")
        lines.append("")
        for component, memory in calc["memory_breakdown"].items():
            if isinstance(memory, (int, float)):
                lines.append(f"- **{component.replace('_', ' ').title()}**: {bytes_to_human(memory)}")
        lines.append("")
    
    # Recommendations
    recommendations = generate_recommendations(calc, hw, bottlenecks)
    if recommendations:
        lines.append("## 💡 Optimization Recommendations")
        lines.append("")
        for rec in recommendations:
            lines.append(rec)
        lines.append("")
    
    # Scalability Analysis (if enabled in config)
    if config and config.get("scalability", {}).get("enabled", False):
        scalability_config = config["scalability"]
        device_counts = scalability_config.get("device_counts", [1, 2, 4, 8])
        scalability = create_scalability_analysis(calc, hw, device_counts)
        
        lines.append("## 📏 Scalability Analysis")
        lines.append("")
        lines.append("| Devices | Est. Time | Speedup | Efficiency | Cost |")
        lines.append("|---------|-----------|---------|------------|------|")
        
        for i, devices in enumerate(device_counts):
            time_str = time_to_human(scalability["estimated_times"][i])
            speedup = scalability["speedups"][i]
            efficiency = scalability["efficiency"][i]
            cost = scalability["cost_analysis"][i] if scalability["cost_analysis"] else 0
            
            lines.append(f"| {devices} | {time_str} | {speedup:.2f}x | {efficiency:.1%} | ${cost:.2f} |")
        
        lines.append("")
    
    # Technical Details
    if calc.get("numerical_stability") or calc.get("scalability") or calc.get("convergence"):
        lines.append("## 🔬 Technical Details")
        lines.append("")
        
        if calc.get("numerical_stability"):
            lines.append(f"- **Numerical Stability**: {calc['numerical_stability']}")
        
        if calc.get("scalability"):
            lines.append(f"- **Scalability**: {calc['scalability']}")
        
        if calc.get("convergence"):
            lines.append(f"- **Convergence**: {calc['convergence']}")
        
        if calc.get("parallelizable"):
            lines.append(f"- **Parallelization**: {calc['parallelizable']}")
        
        lines.append("")
    
    # Warnings
    warnings = []
    if calc.get("scalability_warning"):
        warnings.append(calc["scalability_warning"])
    
    if bottlenecks["memory_utilization"] > 1.0:
        warnings.append("⚠️ Memory requirements exceed available memory!")
    
    if warnings:
        lines.append("## ⚠️ Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("*Report generated by ML Capacity Planner*")
    
    return "\n".join(lines)


def create_json_report(
    model_name: str,
    calc: Dict[str, Any],
    hw: Dict[str, Any],
    efficiency: float = 0.25,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create structured JSON report for programmatic access."""
    bottlenecks = calculate_bottlenecks(calc, hw, efficiency)
    recommendations = generate_recommendations(calc, hw, bottlenecks)
    
    report = {
        "model": {
            "name": model_name,
            "algorithm": calc.get("algorithm", "N/A"),
            "parameters": calc.get("params", 0),
            "complexity_class": calc.get("complexity_class", "N/A"),
            "architecture": calc.get("architecture", {})
        },
        "computation": {
            "train_flops": calc.get("train_flops", 0),
            "train_flops_human": flops_to_human(calc.get("train_flops", 0)),
            "infer_flops_per_sample": calc.get("infer_flops_per_sample", 0),
            "infer_flops_per_token": calc.get("infer_flops_per_token", 0),
            "estimated_train_time_seconds": est_time_seconds(calc.get("train_flops", 0), hw.get("peak_flops", 1), efficiency),
            "estimated_train_time_human": time_to_human(est_time_seconds(calc.get("train_flops", 0), hw.get("peak_flops", 1), efficiency))
        },
        "memory": {
            "model_bytes": calc.get("model_mem_bytes", 0),
            "activation_bytes": calc.get("activation_mem_bytes", 0),
            "dataset_bytes": calc.get("dataset_mem_bytes", 0),
            "total_required_bytes": (
                calc.get("model_mem_bytes", 0) + 
                calc.get("activation_mem_bytes", 0) + 
                calc.get("dataset_mem_bytes", 0)
            ),
            # Use the first non-zero memory field (vram_gb or mem_gb). If both are zero or missing, treat available as 0.
            "available_bytes": ((hw.get("vram_gb") or hw.get("mem_gb") or 0) * 1e9),
            "utilization_percent": (
                ( (calc.get("model_mem_bytes", 0) + calc.get("activation_mem_bytes", 0) + calc.get("dataset_mem_bytes", 0))
                  / ((hw.get("vram_gb") or hw.get("mem_gb") or 1) * 1e9) ) * 100
            ) if ((hw.get("vram_gb") or hw.get("mem_gb") or 0) * 1e9) > 0 else 0
        },
        "hardware": {
            "name": hw.get("name", "Unknown"),
            "type": hw.get("type", "unknown"),
            "peak_flops": hw.get("peak_flops", 0),
            "memory_gb": hw.get("vram_gb", hw.get("mem_gb", 0)),
            "bandwidth_gbps": hw.get("bandwidth_gbps", 0),
            "cost_per_hour": hw.get("cost_per_hour", 0),
            "efficiency_factor": efficiency
        },
        "analysis": {
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "flop_breakdown": calc.get("flop_breakdown", {}),
            "memory_breakdown": calc.get("memory_breakdown", {})
        },
        "metadata": {
            "config": config,
            "timestamp": config.get("timestamp") if config else None
        }
    }
    
    # Add scalability analysis if enabled
    if config and config.get("scalability", {}).get("enabled", False):
        scalability_config = config["scalability"]
        device_counts = scalability_config.get("device_counts", [1, 2, 4, 8])
        report["scalability"] = create_scalability_analysis(calc, hw, device_counts)
    
    return report
