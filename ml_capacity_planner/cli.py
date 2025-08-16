
"""Enhanced CLI with comprehensive features and error handling."""

import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
import yaml

from .registry import MODEL_CALCULATORS, HARDWARE_PROFILES
from . import hardware  # registers defaults
from .calculators import classical, deep  # register models
from .io import load_config, save_report, create_example_configs, validate_config_file
from .report import summarize, create_json_report
from .config import Config, YAMLConfig

console = Console()
app = typer.Typer(
    name="ml-capacity",
    help="🚀 Advanced ML Capacity Planning Tool",
    add_completion=False
)


def print_banner():
    """Print application banner."""
    banner = """
    ╔══════════════════════════════════════════╗
    ║        🚀 ML Capacity Planner           ║
    ║   Runtime • Resources • Scalability     ║
    ╚══════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")


@app.command()
def plan(
    config: Path = typer.Argument(..., help="Path to YAML/JSON configuration file"),
    output: Path = typer.Option("report.md", "--output", "-o", help="Output report path"),
    format: str = typer.Option("markdown", "--format", "-f", help="Output format (markdown, json, html)"),
    efficiency: float = typer.Option(0.25, "--efficiency", "-e", help="Hardware efficiency factor (0.0-1.0)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    validate_only: bool = typer.Option(False, "--validate", help="Only validate configuration"),
    show_hardware: bool = typer.Option(False, "--show-hardware", help="Show available hardware profiles"),
    show_models: bool = typer.Option(False, "--show-models", help="Show available model calculators"),
):
    """Generate capacity planning report from configuration file."""
    
    if not config.exists() and not show_hardware and not show_models:
        console.print(f"❌ Configuration file not found: {config}", style="red")
        raise typer.Exit(1)
    
    if show_hardware:
        display_hardware_profiles()
        return
    
    if show_models:
        display_model_calculators()
        return
    
    try:
        if verbose:
            print_banner()
            console.print(f"📄 Loading configuration from: {config}", style="blue")
        
        # Validate configuration
        if not validate_config_file(config):
            console.print("❌ Configuration validation failed", style="red")
            raise typer.Exit(1)
        
        if validate_only:
            console.print("✅ Configuration is valid", style="green")
            return
        
        # Load and process configuration
        cfg = load_config(config)
        
        if verbose:
            console.print("✅ Configuration loaded successfully", style="green")
        
        # Extract configuration components
        model_name = cfg.model.name
        model_spec = cfg.model.spec
        hw_name = cfg.hardware.name
        hw_efficiency = cfg.hardware.efficiency
        
        # Override efficiency if provided via CLI
        if efficiency != 0.25:
            hw_efficiency = efficiency
        
        # Validate model and hardware existence
        if model_name not in MODEL_CALCULATORS:
            available_models = list(MODEL_CALCULATORS.keys())
            console.print(f"❌ Unknown model calculator: {model_name}", style="red")
            console.print(f"Available models: {', '.join(available_models)}", style="yellow")
            raise typer.Exit(1)
        
        if hw_name not in HARDWARE_PROFILES:
            available_hw = list(HARDWARE_PROFILES.keys())
            console.print(f"❌ Unknown hardware profile: {hw_name}", style="red")
            console.print(f"Available hardware: {', '.join(available_hw)}", style="yellow")
            raise typer.Exit(1)
        
        if verbose:
            console.print(f"🧮 Computing for model: {model_name}", style="blue")
            console.print(f"🖥️  Using hardware: {hw_name}", style="blue")
        
        # Run calculations
        calc = MODEL_CALCULATORS[model_name](model_spec)
        hw = HARDWARE_PROFILES[hw_name]
        
        # Add timestamp to config for reporting
        config_dict = cfg.dict()
        config_dict["timestamp"] = datetime.now().isoformat()
        
        # Generate report based on format
        if format.lower() == "json":
            report_data = create_json_report(model_name, calc, hw, hw_efficiency, config_dict)
            save_report(output, report_data)
        else:
            # Use output file extension to determine format if not explicitly specified
            if output.suffix.lower() == ".json":
                report_data = create_json_report(model_name, calc, hw, hw_efficiency, config_dict)
                save_report(output, report_data)
            else:
                report_content = summarize(model_name, calc, hw, hw_efficiency, config_dict)
                save_report(output, report_content)
        
        if verbose:
            console.print(f"📊 Report saved to: {output}", style="green")
        
        # Display summary in terminal
        if not verbose:
            print_summary(model_name, calc, hw, hw_efficiency)
        else:
            # Show full report in terminal for verbose mode
            report_content = summarize(model_name, calc, hw, hw_efficiency, config_dict)
            console.print(report_content)
        
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def examples(
    output_dir: Path = typer.Option("examples", "--output", "-o", help="Output directory for examples"),
    list_only: bool = typer.Option(False, "--list", "-l", help="List available examples"),
):
    """Create example configuration files."""
    
    if list_only:
        examples = YAMLConfig.get_example_configs()
        console.print("📋 Available example configurations:", style="bold blue")
        
        for name, config in examples.items():
            model_name = config["model"]["name"]
            hw_name = config["hardware"]["name"]
            console.print(f"  • {name}: {model_name} on {hw_name}")
        return
    
    try:
        create_example_configs(output_dir)
        console.print(f"✅ Example configurations created in: {output_dir}", style="green")
        
        # List created files
        console.print("\n📄 Created files:", style="blue")
        for file in Path(output_dir).glob("*.yaml"):
            console.print(f"  • {file.name}")
            
    except Exception as e:
        console.print(f"❌ Error creating examples: {e}", style="red")
        raise typer.Exit(1)


@app.command() 
def validate(
    config: Path = typer.Argument(..., help="Path to configuration file to validate"),
):
    """Validate a configuration file."""
    
    try:
        if validate_config_file(config):
            console.print("✅ Configuration is valid", style="green")
        else:
            console.print("❌ Configuration validation failed", style="red")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"❌ Validation error: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def hardware(
    list_all: bool = typer.Option(False, "--list", "-l", help="List all hardware profiles"),
    show_details: Optional[str] = typer.Option(None, "--details", "-d", help="Show details for specific hardware"),
    show_recommendations: bool = typer.Option(False, "--recommend", "-r", help="Show hardware recommendations"),
):
    """Manage hardware profiles."""
    
    if list_all:
        display_hardware_profiles()
        return
    
    if show_details:
        display_hardware_details(show_details)
        return
    
    if show_recommendations:
        # This would require additional logic to get requirements
        console.print("💡 Hardware recommendations require model specifications", style="yellow")
        console.print("Use 'ml-capacity plan' with a configuration file for recommendations", style="blue")
        return
    
    # Default: show available hardware
    display_hardware_profiles()


@app.command()
def models(
    list_all: bool = typer.Option(False, "--list", "-l", help="List all model calculators"),
    show_details: Optional[str] = typer.Option(None, "--details", "-d", help="Show details for specific model"),
):
    """Manage model calculators."""
    
    if list_all or not show_details:
        display_model_calculators()
        return
    
    if show_details:
        display_model_details(show_details)
        return


def print_summary(model_name: str, calc: dict, hw: dict, efficiency: float):
    """Print a concise summary to console."""
    from .report import flops_to_human, bytes_to_human, est_time_seconds, time_to_human
    
    # Create summary table
    table = Table(title=f"🚀 Capacity Plan: {model_name}")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    
    table.add_row("Parameters", f"{calc.get('params', 0):,}")
    
    if "train_flops" in calc:
        table.add_row("Training FLOPs", flops_to_human(calc['train_flops']))
        train_time = est_time_seconds(calc["train_flops"], hw.get("peak_flops", 1), efficiency)
        table.add_row("Est. Training Time", time_to_human(train_time))
    
    if "infer_flops_per_sample" in calc:
        table.add_row("Inference FLOPs/sample", flops_to_human(calc['infer_flops_per_sample']))
    
    if "model_mem_bytes" in calc:
        table.add_row("Model Memory", bytes_to_human(calc['model_mem_bytes']))
    
    table.add_row("Hardware", f"{hw.get('type', 'unknown').upper()}")
    table.add_row("Peak Performance", flops_to_human(hw.get('peak_flops', 0)))
    
    console.print(table)


def display_hardware_profiles():
    """Display available hardware profiles in a table."""
    table = Table(title="🖥️ Available Hardware Profiles")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Type", style="green")
    table.add_column("Performance", style="magenta")
    table.add_column("Memory", style="yellow")
    table.add_column("Cost/Hour", style="red")
    
    from .report import flops_to_human
    
    for name, profile in HARDWARE_PROFILES.items():
        hw_type = profile.get("type", "unknown").upper()
        performance = flops_to_human(profile.get("peak_flops", 0))
        memory = f"{profile.get('vram_gb', profile.get('mem_gb', 0))} GB"
        cost = f"${profile.get('cost_per_hour', 0):.2f}" if profile.get('cost_per_hour') else "N/A"
        
        table.add_row(name, hw_type, performance, memory, cost)
    
    console.print(table)


def display_hardware_details(hw_name: str):
    """Display detailed information for specific hardware."""
    if hw_name not in HARDWARE_PROFILES:
        console.print(f"❌ Hardware profile not found: {hw_name}", style="red")
        return
    
    profile = HARDWARE_PROFILES[hw_name]
    
    # Create details panel
    details = []
    for key, value in profile.items():
        if key == "peak_flops":
            from .report import flops_to_human
            details.append(f"Peak Performance: {flops_to_human(value)}")
        elif key.endswith("_gb"):
            details.append(f"{key.replace('_', ' ').title()}: {value} GB")
        elif key.endswith("_gbps"):
            details.append(f"{key.replace('_', ' ').title()}: {value} GB/s")
        elif key.endswith("_watts"):
            details.append(f"{key.replace('_', ' ').title()}: {value} W")
        else:
            details.append(f"{key.replace('_', ' ').title()}: {value}")
    
    panel = Panel(
        "\n".join(details),
        title=f"🖥️ Hardware Details: {hw_name}",
        border_style="blue"
    )
    console.print(panel)


def display_model_calculators():
    """Display available model calculators."""
    table = Table(title="🧮 Available Model Calculators")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Category", style="green")
    table.add_column("Description", style="magenta")
    
    # Categorize models
    classical_models = []
    deep_models = []
    
    for name in MODEL_CALCULATORS.keys():
        if any(term in name.lower() for term in ["regression", "svm", "tree", "forest", "kmeans", "pca"]):
            classical_models.append(name)
        else:
            deep_models.append(name)
    
    # Add classical ML models
    for model in sorted(classical_models):
        description = get_model_description(model)
        table.add_row(model, "Classical ML", description)
    
    # Add deep learning models
    for model in sorted(deep_models):
        description = get_model_description(model)
        table.add_row(model, "Deep Learning", description)
    
    console.print(table)


def get_model_description(model_name: str) -> str:
    """Get a brief description of the model."""
    descriptions = {
        "linear_regression_normal": "Linear regression using normal equations",
        "linear_regression_qr": "Linear regression using QR decomposition",
        "logistic_regression_newton": "Logistic regression with Newton-Raphson",
        "logistic_regression_sgd": "Logistic regression with SGD",
        "svm_linear": "Linear Support Vector Machine",
        "svm_rbf": "RBF kernel Support Vector Machine",
        "decision_tree": "CART decision tree",
        "random_forest": "Random Forest ensemble",
        "xgboost_like": "Gradient boosting (XGBoost-style)",
        "kmeans": "K-Means clustering",
        "pca_svd": "Principal Component Analysis with SVD",
        "mlp": "Multi-Layer Perceptron",
        "simple_cnn": "Convolutional Neural Network",
        "transformer_encoder": "Transformer encoder",
        "transformer_decoder": "Transformer decoder (GPT-style)",
        "resnet50": "ResNet-50 architecture"
    }
    return descriptions.get(model_name, "ML model")


def display_model_details(model_name: str):
    """Display detailed information for specific model."""
    if model_name not in MODEL_CALCULATORS:
        console.print(f"❌ Model calculator not found: {model_name}", style="red")
        return
    
    # This would require additional metadata about models
    # For now, show basic information
    description = get_model_description(model_name)
    
    panel = Panel(
        f"Description: {description}\n\n"
        f"Use 'ml-capacity examples' to see configuration examples.",
        title=f"🧮 Model Details: {model_name}",
        border_style="blue"
    )
    console.print(panel)


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
