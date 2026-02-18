# Contributing

Thank you for your interest in contributing to GANNs with friends!

## Ways to contribute

- Report bugs and issues
- Improve documentation
- Add new features
- Optimize performance
- Create tutorials and examples
- Help other users

## Getting started

### Fork and clone

```bash
git clone https://github.com/YOUR_USERNAME/GANNs-with-freinds.git
cd GANNs-with-freinds
```

### Set up development environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black pylint mypy
```

### Create a branch

```bash
git checkout -b feature/my-new-feature
```

## Development workflow

### 1. Make changes

Edit code following the project conventions (see below).

### 2. Test your changes

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_models.py

# Check code style
black src/
pylint src/

# Type checking
mypy src/
```

### 3. Commit

```bash
git add .
git commit -m "Add feature: brief description"
```

Use clear commit messages:
- "Fix: bug in gradient aggregation"
- "Add: CPU auto-detection for workers"
- "Docs: update installation guide"
- "Refactor: simplify database queries"

### 4. Push and create pull request

```bash
git push origin feature/my-new-feature
```

Then create a pull request on GitHub.

## Code style

### Python conventions

- Follow PEP 8
- Use type hints
- Write docstrings for functions
- Keep functions focused and small
- Use meaningful variable names

```python
def load_model_weights(
    model_path: str,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Load model weights from checkpoint file.
    
    Args:
        model_path: Path to checkpoint file
        device: Target device for weights
    
    Returns:
        Dictionary containing model state dict
    """
    checkpoint = torch.load(model_path, map_location=device)
    return checkpoint['model_state_dict']
```

### Single quotes

Use single quotes for strings:

```python
# Good
config = load_config('config.yaml')

# Avoid
config = load_config("config.yaml")
```

### Sentence case for headers

All documentation headers use sentence case:

```markdown
# Installation guide

## Setup instructions

### Download dataset
```

### No emojis in code or docs

Keep documentation professional and accessible.

## Testing

### Write tests for new features

```python
# tests/test_worker.py
import pytest
from src.worker import Worker

def test_worker_initialization():
    worker = Worker('test_config.yaml')
    assert worker.device is not None
    assert worker.batch_size > 0

def test_cpu_batch_size_adjustment():
    # Test CPU auto-detection
    worker = Worker('test_config_cpu.yaml')
    assert worker.batch_size <= 8
```

### Run full test suite

```bash
pytest tests/ -v
```

## Documentation

### Update relevant docs

If you change functionality, update:
- README.md (if user-facing)
- docs/ (detailed documentation)
- Code docstrings
- CHANGELOG.md

### Build and check docs

```bash
cd docs
pip install -r requirements.txt
make html
make serve  # View at http://localhost:8000
```

### Documentation style

- Clear and concise
- Include code examples
- Use sentence case for headers
- No emojis or symbols
- Single quotes in code examples

## Pull request guidelines

### Before submitting

- Tests pass
- Code follows style guide
- Documentation updated
- Commits are clean and logical
- PR description explains changes

### PR description template

```markdown
## Description
Brief description of changes

## Type of change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
How were changes tested?

## Checklist
- [ ] Tests pass
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## Reporting bugs

### Create an issue

Include:
- Clear title
- Steps to reproduce
- Expected behavior
- Actual behavior
- System information (OS, Python version, GPU)
- Error messages and logs

### Example bug report

```markdown
**Title**: Worker crashes when batch size > 128

**Description**:
Worker crashes with CUDA out of memory when batch_size is set above 128 in config.yaml.

**Steps to reproduce**:
1. Set batch_size: 256 in config.yaml
2. Run python src/worker.py
3. Worker crashes after claiming first work unit

**Expected**: Worker should process batch or reduce size automatically

**Actual**: RuntimeError: CUDA out of memory

**System**:
- OS: Ubuntu 22.04
- Python: 3.10.12
- GPU: NVIDIA RTX 3060 (12GB)
- PyTorch: 2.0.1+cu118

**Logs**:
[attach relevant logs]
```

## Feature requests

### Propose new features

Open an issue with:
- Clear use case
- Expected behavior
- Why it's beneficial
- Potential implementation approach

### Discuss first

For major features, discuss with maintainers before implementing.

## Code review process

### What we look for

- Correctness
- Code quality
- Test coverage
- Documentation
- Performance impact
- Backward compatibility

### Be responsive

- Respond to review comments
- Make requested changes
- Ask questions if unclear
- Be open to feedback

## Project areas

### Priority contributions

1. **Testing**
   - Increase test coverage
   - Add integration tests
   - Test edge cases

2. **Documentation**
   - Improve clarity
   - Add more examples
   - Create tutorials

3. **Performance**
   - Optimize database queries
   - Reduce network overhead
   - Improve gradient aggregation

4. **Features**
   - Multi-GPU support per worker
   - Gradient compression
   - Web-based monitoring dashboard
   - Support for other datasets/models

5. **Usability**
   - Better error messages
   - Improved logging
   - Setup automation

## Community

### Be respectful

- Assume good intentions
- Be patient with beginners
- Give constructive feedback
- Follow code of conduct

### Help others

- Answer questions
- Review pull requests
- Improve documentation
- Share knowledge

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

## Questions?

- Open an issue for clarification
- Ask in pull request comments
- Contact project maintainers

## Thank you!

Your contributions make this project better for everyone. We appreciate your time and effort.
