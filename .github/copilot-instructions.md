# Copilot instructions for GANNs with Friends

## Project overview

This is an **educational distributed deep learning system** where students participate as workers in a compute cluster to train a GAN (Generative Adversarial Network) to generate celebrity faces. The system uses PostgreSQL as a coordination hub instead of complex networking, making it accessible for classroom environments.

**Key principle:** This is an educational project. Code should be readable, well-documented, and designed for learning, not just performance.

## Development philosophy

This project uses **function-level meta-coding** (AI-assisted development):
- Human provides architectural decisions, design choices, and testing
- AI assists with implementation, consistency, and documentation
- Quality and maintainability take precedence over speed
- All architectural decisions should be intentional and documented

## Code style and conventions

### Python code

- **Type hints**: Use type hints for all function parameters and return values
- **Docstrings**: Use Google-style docstrings for all functions, classes, and modules
  - Include `Args:`, `Returns:`, and `Raises:` sections where applicable
  - Be descriptive but concise
- **Imports**: Group imports logically (standard library, third-party, local)
- **Naming**: Use descriptive names; prefer clarity over brevity
- **Context managers**: Use `@contextmanager` for database sessions and resource cleanup

**Example docstring pattern:**
```python
def process_gradients(self, iteration: int, gradients: List[torch.Tensor]) -> bool:
    """Aggregate and apply gradients for the current iteration.
    
    Args:
        iteration: Current training iteration number
        gradients: List of gradient tensors from workers
        
    Returns:
        True if aggregation successful, False otherwise
        
    Raises:
        DatabaseError: If gradient retrieval fails
    """
```

### Documentation

- **Format**: Use Markdown for all documentation
- **Sphinx**: Documentation is built with Sphinx + MyST parser
- **Structure**: Keep docs organized in logical sections (getting-started, guides, resources, etc.)
- **Tone**: Educational and approachable, not overly formal
- **Examples**: Include practical examples demonstrating concepts
- **Links**: Use relative links for internal documentation references

### Configuration

- **YAML format**: Use `config.yaml` for all configuration
- **Sensible defaults**: Default values should work for most users
- **Documentation**: Every config option should be documented in `docs/guides/configuration.md`
- **Validation**: Validate configuration at startup with clear error messages

## Architecture patterns

### Database as coordinator

The PostgreSQL database is the central coordination mechanism:
- No direct worker-to-coordinator communication
- Workers poll for work units
- All state stored in database tables
- Use atomic operations for work unit claims

### Fault tolerance

Design for resilience:
- Workers can disconnect/reconnect at any time
- Work unit timeouts handle stalled workers
- Stale work gets cancelled when iterations advance
- Heartbeat system tracks active workers

### Work unit pattern

```python
# Workers claim work atomically
work_unit = db.claim_work_unit(worker_id, iteration)

# Process and submit results
gradients = compute_gradients(work_unit.batch_indices)
db.save_gradients(work_unit.id, gradients)

# Coordinator aggregates when threshold met
if db.count_completed_work_units(iteration) >= threshold:
    aggregate_and_update_models(iteration)
    db.cancel_pending_work_units(iteration)  # Cancel stale work
```

## Key components

### Database manager (`src/database/db_manager.py`)
- All database operations go through DatabaseManager
- Use context managers for session handling
- Separate methods for each operation type
- Include proper error handling and rollback

### Models (`src/models/dcgan.py`)
- Standard PyTorch nn.Module pattern
- Include initialization functions (e.g., `weights_init`)
- Keep model architecture simple and documented

### Main coordinator (`src/main.py`)
- Clear separation of concerns
- Progress tracking and logging
- Configuration validation
- Graceful shutdown handling

### Worker (`src/worker.py`)
- Polling loop with configurable interval
- Heartbeat mechanism
- Graceful shutdown on keyboard interrupt
- Clear status messages

## Common patterns

### Error handling
```python
try:
    # Operation
    result = perform_operation()
except SpecificError as e:
    logger.error(f"Failed to perform operation: {e}")
    # Handle or re-raise
```

### Configuration access
```python
# Load from config
config = yaml.safe_load(open('config.yaml'))
batch_size = config['training']['batch_size']
```

### Logging
```python
# Use print statements for now (no logging framework)
print(f"Processing iteration {iteration}...")
print(f"Completed work unit {work_unit_id}")
```

## Testing

- **No automated tests**: This project uses manual testing only
- Test changes by running the affected scripts
- Verify with actual database operations
- Test both coordinator and worker modes
- Check edge cases (worker disconnect, database errors, etc.)

## Documentation updates

When adding features or making changes:
1. Update relevant documentation files in `docs/`
2. Update `config.yaml.template` if adding config options
3. Ensure all examples in docs reflect current code
4. Rebuild documentation with `cd docs && make html`

## What to avoid

- **Don't add automated testing infrastructure** - This project intentionally uses manual testing
- **Don't overcomplicate** - Favor simple, readable solutions over clever optimizations
- **Don't assume networking** - Everything goes through the database
- **Don't add dependencies** unless necessary - Keep requirements lean
- **Don't break backward compatibility** in config.yaml without documenting migration

## Educational focus

Always consider: "Would a student understand this code?"
- Add explanatory comments for complex logic
- Use descriptive variable names
- Break complex operations into smaller, named functions
- Document the "why" not just the "what"
- Include examples in docstrings

## Version control

- **Do not write commit messages** - Human handles all git commits
- **Do not make commits** - Human reviews changes and commits them
- **Do not push changes** - Human controls when code is pushed
- Human is responsible for version control decisions and workflow

## When helping with code

1. **Understand the context**: Ask for architectural guidance before implementing
2. **Maintain consistency**: Follow existing patterns in the codebase
3. **Document thoroughly**: Add docstrings and comments
4. **Keep it educational**: Prioritize readability and learning value
5. **Verify assumptions**: Check that suggested changes align with project goals
6. **Test manually**: Describe how to test the changes

## Project priorities (in order)

1. **Educational value** - Students should learn from reading and using this code
2. **Reliability** - Distributed training should work across diverse hardware
3. **Simplicity** - Minimize complexity and external dependencies
4. **Documentation** - Everything should be well-documented
5. **Performance** - Optimize only after above priorities are met
