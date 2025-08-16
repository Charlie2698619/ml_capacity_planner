# ML Capacity Planner

A small toolkit to estimate ML training/inference resource needs, runtime, and scalability from a YAML model configuration.

This repo provides:
- calculators for classical and deep models (FLOPs, params, memory)
- hardware profiles and bottleneck analysis
- a CLI to load YAML configs and generate reports (JSON/Markdown)

## Requirements
- Python 3.10+
- Recommended: create and use a virtual environment

## Quick install
```bash
# from project root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # if present, or install dependencies used by the project
```

## Basic usage
Generate a plan from an example YAML and save JSON results:
```bash
ml-capacity plan examples/linear_regression.yaml --format json --output results.json
```
Generate a human-readable report:
```bash
ml-capacity plan examples/deep_learning.yaml --format markdown --output report.md
```
List available example configs:
```bash
ml-capacity examples
```

## Run tests
If the project includes tests, run:
```bash
pytest -q
```

## Pushing to GitHub
1. Create a repository on GitHub named `ml_capacity_planner` (or your preferred name).
2. Add remote and push (HTTPS):
```bash
git remote add origin https://github.com/<your-username>/ml_capacity_planner.git
git push -u origin master
```
Or use SSH after adding your public key to GitHub:
```bash
git remote add origin git@github.com:<your-username>/ml_capacity_planner.git
git push -u origin master
```

## Contributing
- Update or add example YAMLs in `examples/`.
- Add calculators under `calculators/` and register them in the registry if needed.
- Add tests under `tests/` and run `pytest`.

## Notes
- The CLI uses YAML model specs. Check `examples/` for valid input shape.
- If you encounter division-by-zero or missing hardware fields, try adding `mem_gb` or `vram_gb` to your hardware config.

## License
Pick a license you prefer (MIT recommended). If you want, I can add a `LICENSE` file.

---

If you'd like a shorter or longer README, or want a specific license added, tell me which and I'll update the file.
