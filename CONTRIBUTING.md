# Contributing to Telecom Customer Churn Analysis

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Code of Conduct

This project adheres to professional standards. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Issues

- Check if the issue already exists
- Provide a clear description of the problem
- Include steps to reproduce
- Specify your environment (Python version, OS, etc.)

### Submitting Changes

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/Telecom-Churn-Analysis.git
   cd Telecom-Churn-Analysis
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clean, well-documented code
   - Follow PEP 8 style guidelines
   - Add docstrings and type hints

4. **Test your changes**
   ```bash
   python telecom_churn_analysis.py
   ```

5. **Commit your changes**
   ```bash
   git commit -m "feat: description of your changes"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Ensure code follows project standards

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install development tools (optional)
pip install pylint black flake8
```

## Code Style

- Follow PEP 8
- Use meaningful variable names
- Add comments for complex logic
- Keep functions concise and focused

## Commit Message Convention

- `feat: ` for new features
- `fix: ` for bug fixes
- `docs: ` for documentation
- `style: ` for formatting
- `refactor: ` for code refactoring
- `test: ` for tests

Example: `feat: add cross-validation for model evaluation`

## Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, your changes will be merged

## Questions?

Feel free to open an issue or contact the maintainers.

Thank you for contributing!
