# Hallucination Detection
![workflow](https://github.com/graphml-lab-pwr/hallucinations/actions/workflows/main.yaml/badge.svg)
![python-3.12](https://img.shields.io/badge/Python-3.12-blue)

## Installation

1. Install all necessary dependencies (choose either `cpu` or `gpu`, depending on your platform):

```bash
make install_<cpu|gpu>
```

Dependencies in this project are defined in through 3 files:
- `requirements.txt` - dependencies common among all platforms (GPU and CPU)
- `requirements-cpu.txt` - dependencies specific to CPU platform
- `requirements-gpu.txt` - dependencies specific to GPU platform

## Development

1. Install pre-commit hooks (prevent `git commit` while static checks are failing):
   ```bash
   pre-commit install
   ```

2. Configure local `minio` profile
   ```bash
   dvc remote modify --local minio profile <profile from `~/.aws/credentials`>
   ```
> [!IMPORTANT]
> `minio` endpoint available only through VPN.

3. Run static code checks:
    - check only :
         ```bash
         make quality
         ```
    - check and fix:
         ```bash
         make fix
         ```
