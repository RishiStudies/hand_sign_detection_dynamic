# Configuration Directory

This folder contains configuration files for the project.

## Files

| File | Purpose |
|------|---------|
| `pytest.ini` | Pytest configuration |
| `.env.example` | Example environment variables |

## Usage

### Environment Variables

Copy `.env.example` to `.env` in the project root and customize:

```bash
cp config/.env.example .env
# Edit .env with your settings
```

### Pytest

The pytest.ini is automatically discovered. Run tests from project root:

```bash
pytest
```

Or specify the config explicitly:

```bash
pytest -c config/pytest.ini
```
