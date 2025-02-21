all: quality
check_dirs := hallucinations scripts tests
# Check that source code meets quality standards

quality:
	pre-commit run --all-files
	mypy --install-types --non-interactive $(check_dirs)

fix:
	pre-commit run --all-files

test:
	pytest

install_cpu:
	pip install -r requirements-cpu.txt -r requirements.txt

install_gpu:
	pip install -r requirements-gpu.txt -r requirements.txt
	pip install flash-attn==2.6.3 --no-build-isolation
