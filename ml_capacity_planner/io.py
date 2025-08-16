
"""Enhanced I/O module with validation and better error handling."""

import yaml
import json
from pathlib import Path
from typing import Union, Dict, Any
from .config import Config, YAMLConfig


def load_config(path: Union[str, Path]) -> Config:
    """Load and validate configuration from YAML or JSON file."""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    try:
        with open(path, "r") as f:
            if path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Validate configuration using Pydantic
        config = Config(**data)
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {path}: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON syntax in {path}: {e}")
    except Exception as e:
        raise ValueError(f"Configuration validation error: {e}")


def save_report(path: Union[str, Path], content: Union[str, Dict[str, Any]]) -> None:
    """Save report to file with proper formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, "w") as f:
            if path.suffix.lower() == ".json":
                if isinstance(content, str):
                    # Try to parse as JSON first
                    try:
                        data = json.loads(content)
                        json.dump(data, f, indent=2)
                    except json.JSONDecodeError:
                        # Fall back to raw string
                        json.dump({"content": content}, f, indent=2)
                else:
                    json.dump(content, f, indent=2)
            elif path.suffix.lower() == ".html":
                if isinstance(content, dict):
                    # Convert dict to HTML format
                    html_content = dict_to_html(content)
                    f.write(html_content)
                else:
                    f.write(str(content))
            else:
                # Default to markdown/text
                if isinstance(content, dict):
                    # Convert dict to markdown
                    md_content = dict_to_markdown(content)
                    f.write(md_content)
                else:
                    f.write(str(content))
                    
    except Exception as e:
        raise IOError(f"Failed to save report to {path}: {e}")


def dict_to_markdown(data: Dict[str, Any]) -> str:
    """Convert dictionary to markdown format."""
    lines = []
    
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"## {key.replace('_', ' ').title()}")
            lines.append("")
            for subkey, subvalue in value.items():
                lines.append(f"- **{subkey.replace('_', ' ').title()}**: {subvalue}")
            lines.append("")
        elif isinstance(value, list):
            lines.append(f"## {key.replace('_', ' ').title()}")
            lines.append("")
            for item in value:
                lines.append(f"- {item}")
            lines.append("")
        else:
            lines.append(f"**{key.replace('_', ' ').title()}**: {value}")
            lines.append("")
    
    return "\n".join(lines)


def dict_to_html(data: Dict[str, Any]) -> str:
    """Convert dictionary to HTML format."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Capacity Planning Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1, h2 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #f2f2f2; }
            .highlight { background-color: #ffffcc; }
        </style>
    </head>
    <body>
        <h1>ML Capacity Planning Report</h1>
    """
    
    for key, value in data.items():
        html += f"<h2>{key.replace('_', ' ').title()}</h2>"
        if isinstance(value, dict):
            html += "<table>"
            for subkey, subvalue in value.items():
                html += f"<tr><td><strong>{subkey.replace('_', ' ').title()}</strong></td><td>{subvalue}</td></tr>"
            html += "</table>"
        elif isinstance(value, list):
            html += "<ul>"
            for item in value:
                html += f"<li>{item}</li>"
            html += "</ul>"
        else:
            html += f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>"
    
    html += "</body></html>"
    return html


def create_example_configs(output_dir: Union[str, Path] = "examples") -> None:
    """Create example configuration files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    examples = YAMLConfig.get_example_configs()
    
    for name, config in examples.items():
        config_file = output_dir / f"{name}.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"Created example config: {config_file}")


def validate_config_file(path: Union[str, Path]) -> bool:
    """Validate a configuration file without loading it fully."""
    try:
        load_config(path)
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False
