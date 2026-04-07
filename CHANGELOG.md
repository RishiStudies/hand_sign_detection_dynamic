# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-04-05

### Added
- LSTM model training for dynamic hand sign detection
- Parallel video preprocessing with `ProcessPoolExecutor`
- WLASL dataset support (11,998 videos)
- Real-time gesture combo detection
- RESTful API with FastAPI
- Rate limiting and API key authentication
- Docker support (backend + worker containers)
- CI/CD pipeline with GitHub Actions
- Comprehensive test suite

### Security
- Replaced MD5 with Blake2b for cache key hashing
- Added timing-safe API key comparison (prevents timing attacks)
- Implemented rate limiting for failed authentication attempts
- Strengthened API key complexity requirements (32+ chars)

### Changed
- Refactored codebase to use `logging` module instead of `print()`
- Improved exception handling with specific exception types
- Modularized training script into smaller functions
- Reorganized project structure for production readiness

### Project Structure
```
hand_sign_detection_dynamic/
├── src/                    # Source code
│   └── hand_sign_detection/
├── scripts/                # Utility scripts
├── notebooks/              # Jupyter notebooks
├── data/                   # Datasets (not in git)
├── models/                 # Trained models (not in git)
├── tests/                  # Test suite
├── docs/                   # Documentation
├── config/                 # Configuration files
├── requirements/           # Dependency files
└── examples/               # Usage examples
```

## [0.9.0] - 2026-04-04

### Added
- Initial project restructuring
- WLASL video preprocessor
- LSTM trainer with early stopping
- Random Forest classifier for static gestures
- MediaPipe hand landmark extraction

### Changed
- Consolidated requirements into `requirements/` folder
- Moved documentation to `docs/` folder
- Archived legacy notebooks to `.archive/`

---

[Unreleased]: https://github.com/username/hand_sign_detection_dynamic/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/username/hand_sign_detection_dynamic/compare/v0.9.0...v1.0.0
[0.9.0]: https://github.com/username/hand_sign_detection_dynamic/releases/tag/v0.9.0
