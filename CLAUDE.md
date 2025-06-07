# Claude Development Guidelines

## Core Principles

### 1. Correctness First
- **Always verify code before implementation**: Read existing code thoroughly to understand context and dependencies
- **Test assumptions**: Don't assume how existing code works - examine it directly
- **Validate inputs and outputs**: Check that functions handle edge cases and invalid inputs gracefully
- **Follow existing patterns**: Maintain consistency with the codebase's established conventions

### 2. Diligent Code Review
- **Read the entire relevant codebase section** before making changes
- **Understand the data flow** and how components interact
- **Check for breaking changes** that might affect other parts of the system
- **Verify imports and dependencies** are correct and available

### 3. Comprehensive Testing Strategy

#### Before Writing Code:
1. **Examine existing test files** to understand testing patterns
2. **Identify test cases** that need to be covered
3. **Plan test data** and expected outcomes
4. **Consider edge cases** and error conditions

#### During Development:
1. **Write tests alongside code** - don't defer testing
2. **Test incrementally** - verify each component works before moving to the next
3. **Use meaningful test data** that reflects real-world scenarios
4. **Test both success and failure paths**

#### After Implementation:
1. **Run all existing tests** to ensure no regressions
2. **Verify new functionality** with comprehensive test cases
3. **Test integration points** between components
4. **Validate performance** if applicable

## Python-Specific Guidelines

### Environment Setup
- **Always use `uv` for package management** as specified in workspace rules
- Use `python3` or `uv run python` for execution
- Check `pyproject.toml` for project dependencies and configuration

### Package Management with uv

#### Installing Packages
```bash
# Install a package and add to dependencies
uv add pandas

# Install a specific version
uv add pandas==2.1.0

# Install development dependencies
uv add --dev pytest black mypy

# Install optional dependencies
uv add --optional visualization matplotlib seaborn

# Install from requirements.txt
uv pip install -r requirements.txt
```

#### Managing Dependencies
```bash
# Sync dependencies (install all from pyproject.toml)
uv sync

# Update all dependencies
uv lock --upgrade

# Update specific package
uv add pandas --upgrade

# Remove a package
uv remove pandas

# Show installed packages
uv pip list

# Show dependency tree
uv tree
```

#### Running Python with uv
```bash
# Run Python script with uv environment
uv run python script.py

# Run Python with specific arguments
uv run python -m pytest tests/

# Run interactive Python shell
uv run python

# Execute module directly
uv run -m jupyter notebook
```

#### Project Initialization
```bash
# Initialize new project with pyproject.toml
uv init

# Add Python version requirement
uv python pin 3.11

# Create virtual environment
uv venv

# Activate environment (if needed for other tools)
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

#### Troubleshooting Common Issues

**Package Not Found:**
```bash
# Update package index
uv pip install --upgrade pip

# Try installing with pip fallback
uv pip install pandas

# Check if package exists
uv search pandas
```

**Dependency Conflicts:**
```bash
# Show conflict details
uv lock --verbose

# Force resolution (use carefully)
uv add pandas --force

# Check what would be installed
uv add pandas --dry-run
```

**Environment Issues:**
```bash
# Recreate environment
rm -rf .venv
uv venv
uv sync

# Check Python version
uv run python --version

# Verify uv installation
uv --version
```

#### Best Practices
- **Always use `uv add`** instead of `pip install` to ensure dependencies are tracked
- **Commit `uv.lock`** to version control for reproducible builds
- **Use `uv sync`** when setting up project on new machine
- **Pin Python version** in pyproject.toml for consistency
- **Separate dev dependencies** using `--dev` flag
- **Use virtual environments** to isolate project dependencies

#### Example Workflow
```bash
# Setting up a new data analysis project
uv init data-analysis
cd data-analysis
uv python pin 3.11
uv add pandas numpy matplotlib jupyter
uv add --dev pytest black mypy
uv sync

# Running the project
uv run python analysis.py
uv run jupyter notebook

# Adding new dependencies as needed
uv add scikit-learn seaborn
```

### Code Quality Standards
```python
# Good: Clear, testable function with proper error handling
def calculate_reimbursement(expenses: List[Expense], policy: Policy) -> ReimbursementResult:
    """Calculate reimbursement based on expenses and policy rules.
    
    Args:
        expenses: List of expense objects to process
        policy: Reimbursement policy to apply
        
    Returns:
        ReimbursementResult with calculated amounts and details
        
    Raises:
        ValueError: If expenses list is empty or policy is invalid
    """
    if not expenses:
        raise ValueError("Expenses list cannot be empty")
    if not policy.is_valid():
        raise ValueError("Invalid policy configuration")
    
    # Implementation with clear logic
    result = ReimbursementResult()
    for expense in expenses:
        if policy.is_reimbursable(expense):
            result.add_reimbursement(expense, policy.calculate_amount(expense))
    
    return result
```
## Error Handling and Logging

### Error Handling
```python
# Good: Specific exceptions with helpful messages
try:
    result = process_expense(expense)
except InvalidExpenseError as e:
    logger.error(f"Invalid expense data: {e}")
    raise ValidationError(f"Expense validation failed: {e}")
except PolicyError as e:
    logger.error(f"Policy application failed: {e}")
    raise ProcessingError(f"Unable to process expense: {e}")
```

### Logging
- **Use appropriate log levels** (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Include context** in log messages (user ID, transaction ID, etc.)
- **Log both successes and failures** for audit trails
- **Avoid logging sensitive information** (passwords, personal data)

## Performance and Security

### Performance Considerations
- **Profile code** for bottlenecks before optimizing
- **Use appropriate data structures** for the task
- **Consider memory usage** with large datasets
- **Implement caching** where appropriate

### Security Best Practices
- **Validate all inputs** from external sources
- **Sanitize data** before processing
- **Use parameterized queries** for database operations
- **Handle sensitive data** according to privacy requirements

## Debugging and Troubleshooting

### Debugging Strategy
1. **Reproduce the issue** with minimal test case
2. **Add logging** to understand data flow
3. **Use debugger** to step through problematic code
4. **Check assumptions** about data types and values
5. **Verify dependencies** are working as expected

### Common Issues Checklist
- [ ] Are all imports available and correct?
- [ ] Are data types what the code expects?
- [ ] Are there any null/None values causing issues?
- [ ] Are file paths and permissions correct?
- [ ] Are external services/APIs responding correctly?
- [ ] Are environment variables set properly?

## Code Review Checklist

Before submitting code:
- [ ] All tests pass (unit, integration, and existing tests)
- [ ] Code follows project style guidelines
- [ ] Functions have appropriate docstrings
- [ ] Error handling is comprehensive
- [ ] No hardcoded values (use configuration)
- [ ] Performance is acceptable for expected load
- [ ] Security considerations are addressed
- [ ] Documentation is updated if needed

## Continuous Improvement

### Learning from Issues
- **Document solutions** to complex problems
- **Update guidelines** based on lessons learned
- **Share knowledge** with team members
- **Refactor code** to prevent similar issues

### Code Quality Metrics
- **Test coverage** should be > 80%
- **Cyclomatic complexity** should be reasonable
- **Code duplication** should be minimized
- **Documentation coverage** should be comprehensive

## Emergency Procedures

### When Things Go Wrong
1. **Stop and assess** the situation
2. **Revert changes** if they caused the issue
3. **Isolate the problem** to specific components
4. **Fix incrementally** with tests for each fix
5. **Document the incident** and prevention measures

### Recovery Steps
- **Backup data** before making fixes
- **Test fixes** in isolated environment first
- **Deploy fixes** gradually with monitoring
- **Verify resolution** with comprehensive testing

---

Remember: **Quality over speed**. It's better to take time to do things right than to rush and create problems that take longer to fix later.