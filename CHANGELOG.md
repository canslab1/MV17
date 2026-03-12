# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## v1.0.0 (2025-03-12)

Complete rewrite from Python 2/3 command-line scripts to Python 3 GUI application.

### Added
- PySide6 graphical user interface with 5 functional tabs
- Background worker threads for all computationally expensive operations
- `NetworkManager` as shared data layer with Qt Signal/Slot communication
- `algorithm_adapter.py` encapsulating all algorithms with NetworkX 3.x compatibility
- `run_sir_batch.py` for command-line batch SIR experiments
- 25 real-world and 4 theoretical network datasets (29 edgelist files with precomputed auxiliary data)

### Changed
- All legacy `util/` modules replaced with unified `algorithm_adapter.py`
- Progress reporting via Qt signals (supports real-time GUI feedback)
- Modern Python APIs: f-strings, type hints

## v0.1.0 (2014)

Original implementation by Yu-Hsiang Fu, Chung-Yuan Huang, and Chuen-Tsai Sun.

- Core algorithm: k-shell entropy × neighbor degree sum (MV17 measure)
- SIR epidemic simulation for node influence validation
- Edge list, pair-value, position, and MessagePack file I/O
- Matplotlib visualization
