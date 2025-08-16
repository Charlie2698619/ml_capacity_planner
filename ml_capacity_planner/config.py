"""Configuration models and validation using Pydantic."""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class PrecisionType(str, Enum):
    """Supported precision types."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


class HardwareType(str, Enum):
    """Supported hardware types."""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


class OptimizationType(str, Enum):
    """Optimization algorithms."""
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"


class ModelSpec(BaseModel):
    """Base model specification."""
    n: Optional[int] = Field(default=None, description="Number of samples")
    p: Optional[int] = Field(default=None, description="Number of features")
    batch_size: Optional[int] = Field(default=32, description="Batch size")
    epochs: Optional[int] = Field(default=10, description="Number of epochs")
    precision: PrecisionType = Field(default=PrecisionType.FP32, description="Model precision")
    
    class Config:
        extra = "allow"  # Allow additional fields for specific models


class MLPSpec(ModelSpec):
    """MLP specific configuration."""
    d_in: int = Field(description="Input dimension")
    d_out: int = Field(default=1, description="Output dimension")
    layers: List[int] = Field(default=[128, 128], description="Hidden layer sizes")
    activation: str = Field(default="relu", description="Activation function")
    dropout: float = Field(default=0.0, description="Dropout rate", ge=0.0, le=1.0)


class CNNSpec(ModelSpec):
    """CNN specific configuration."""
    image_h: int = Field(default=224, description="Image height")
    image_w: int = Field(default=224, description="Image width")
    c_in: int = Field(default=3, description="Input channels")
    channels: List[int] = Field(default=[64, 128, 256], description="Channel progression")
    kernel: int = Field(default=3, description="Kernel size")
    stride: int = Field(default=2, description="Stride")
    num_classes: int = Field(default=1000, description="Number of output classes")


class TransformerSpec(ModelSpec):
    """Transformer specific configuration."""
    d_model: int = Field(default=768, description="Model dimension")
    n_layers: int = Field(default=12, description="Number of layers")
    n_heads: int = Field(default=12, description="Number of attention heads")
    seq_len: int = Field(default=512, description="Sequence length")
    vocab_size: int = Field(default=30000, description="Vocabulary size")
    n_tokens: int = Field(default=10_000_000, description="Total tokens for training")


class ModelConfig(BaseModel):
    """Model configuration."""
    name: str = Field(description="Model type name")
    spec: Dict[str, Any] = Field(description="Model-specific parameters")
    optimization: OptimizationType = Field(default=OptimizationType.ADAM, description="Optimizer")
    learning_rate: float = Field(default=1e-3, description="Learning rate", gt=0.0)


class HardwareConfig(BaseModel):
    """Hardware configuration."""
    name: str = Field(description="Hardware profile name")
    efficiency: float = Field(default=0.25, description="Hardware efficiency factor", gt=0.0, le=1.0)
    parallel_devices: int = Field(default=1, description="Number of parallel devices", ge=1)
    memory_limit_gb: Optional[float] = Field(default=None, description="Memory limit override")


class ScalabilityConfig(BaseModel):
    """Scalability analysis configuration."""
    enabled: bool = Field(default=False, description="Enable scalability analysis")
    device_counts: List[int] = Field(default=[1, 2, 4, 8], description="Device counts to analyze")
    batch_sizes: List[int] = Field(default=[32, 64, 128, 256], description="Batch sizes to analyze")
    data_parallel: bool = Field(default=True, description="Data parallelism analysis")
    model_parallel: bool = Field(default=False, description="Model parallelism analysis")


class ReportConfig(BaseModel):
    """Report generation configuration."""
    format: str = Field(default="markdown", description="Output format (markdown, json, html)")
    include_plots: bool = Field(default=True, description="Include performance plots")
    include_recommendations: bool = Field(default=True, description="Include optimization recommendations")
    detailed_breakdown: bool = Field(default=False, description="Include detailed computation breakdown")


class Config(BaseModel):
    """Main configuration schema."""
    model: ModelConfig = Field(description="Model configuration")
    hardware: HardwareConfig = Field(description="Hardware configuration")
    scalability: ScalabilityConfig = Field(default_factory=ScalabilityConfig, description="Scalability analysis")
    report: ReportConfig = Field(default_factory=ReportConfig, description="Report configuration")
    
    @validator('model')
    def validate_model_spec(cls, v):
        """Validate model specification based on model type."""
        model_name = v.name.lower()
        spec = v.spec
        
        # Basic validation - more specific validation can be added per model type
        if 'n' in spec and spec['n'] <= 0:
            raise ValueError("Number of samples must be positive")
        if 'p' in spec and spec['p'] <= 0:
            raise ValueError("Number of features must be positive")
        
        return v


class YAMLConfig:
    """YAML configuration examples and templates."""
    
    @staticmethod
    def get_example_configs() -> Dict[str, Dict]:
        """Get example configurations for different scenarios."""
        return {
            "linear_regression": {
                "model": {
                    "name": "linear_regression_normal",
                    "spec": {
                        "n": 100000,
                        "p": 1000
                    }
                },
                "hardware": {
                    "name": "CPU_8C_3.5Ghz_baseline",
                    "efficiency": 0.3
                },
                "report": {
                    "format": "markdown",
                    "include_plots": True
                }
            },
            "deep_learning": {
                "model": {
                    "name": "mlp",
                    "spec": {
                        "n": 50000,
                        "d_in": 784,
                        "d_out": 10,
                        "layers": [512, 256, 128],
                        "batch_size": 64,
                        "epochs": 20,
                        "precision": "fp16"
                    },
                    "optimization": "adamw",
                    "learning_rate": 0.001
                },
                "hardware": {
                    "name": "RTX_3060_fp16",
                    "efficiency": 0.4,
                    "parallel_devices": 1
                },
                "scalability": {
                    "enabled": True,
                    "device_counts": [1, 2, 4],
                    "batch_sizes": [32, 64, 128, 256]
                },
                "report": {
                    "format": "html",
                    "include_plots": True,
                    "include_recommendations": True,
                    "detailed_breakdown": True
                }
            },
            "transformer": {
                "model": {
                    "name": "transformer_encoder",
                    "spec": {
                        "d_model": 768,
                        "n_layers": 12,
                        "n_heads": 12,
                        "seq_len": 512,
                        "vocab_size": 30000,
                        "n_tokens": 1000000000,
                        "batch_size": 32,
                        "epochs": 1,
                        "precision": "fp16"
                    }
                },
                "hardware": {
                    "name": "A100_80GB_fp16",
                    "efficiency": 0.5,
                    "parallel_devices": 8
                },
                "scalability": {
                    "enabled": True,
                    "device_counts": [1, 2, 4, 8],
                    "data_parallel": True,
                    "model_parallel": True
                }
            }
        }
