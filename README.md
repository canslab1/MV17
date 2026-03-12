# MV17 — Network Spreader Analysis Tool

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

A GUI application for identifying influential spreaders in complex networks using the **MV17 measure**, which combines **global diversity** (k-shell entropy) and **local topology features** (neighbor degree sum).

## Overview

Identifying the most influential spreaders in complex networks is critical for accelerating information dissemination, controlling epidemics, and optimizing viral marketing. Traditional centrality measures (degree, betweenness, closeness) each capture only one aspect of node importance.

MV17 addresses this by proposing a two-step framework that considers both a node's **global position** across network core layers and its **local connectivity** to nearby high-degree neighbors. The framework is parameter-free and topology-driven, requiring no community labels or edge weights.

## The MV17 Measure

The MV17 measure identifies influential spreaders using a two-step framework:

1. **Global Diversity (E)** — Uses k-shell decomposition + Shannon entropy to measure how diversely a node's neighbors are distributed across network core layers.
2. **Local Feature (L)** — Uses log₂ of the sum of neighbors' neighbors' degrees to capture a node's local reach.

The final influence score is: **IF = E × L**

## Features

- **Two-step framework** — Combines global diversity and local topology into a single influence score.
- **SIR simulation** — Compare spreading capability of nodes ranked by different centrality measures.
- **Multiple centrality metrics** — Degree, k-core, betweenness, closeness, PageRank, clustering coefficient, k-core entropy, neighbor-core, neighbor-degree, and MV17.
- **Dual interface** — PySide6 GUI with 5 functional tabs and a CLI batch experiment runner.
- **Background processing** — All heavy computations run in worker threads with progress feedback.
- **Batch analysis** — Process multiple networks and compare results in a single run.

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/canslab1/MV17.git
cd MV17
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| NetworkX | Network analysis and graph algorithms |
| NumPy | Numerical computing |
| Matplotlib | Visualization and plotting |
| PySide6 | Qt-based GUI |

## Usage

### GUI Mode

```bash
python main.py
```

Launches a desktop application with five tabs:

- **Network I/O** — Browse and load edgelist files, extract giant connected component (GCC).
- **Visualization** — Interactive network drawing with attribute-based node coloring and k-core decomposition view.
- **Node Attributes** — Compute degree, k-core, PageRank, betweenness, closeness, MV17, and more.
- **SIR Experiment** — Run SIR propagation simulations comparing multiple centrality measures.
- **Statistics** — Basic network statistics, scatter plots, scatter matrix, propagation curves, batch analysis.

### CLI Mode

```bash
python run_sir_batch.py <network_name> <edgelist_path> <beta> <output_file>
```

Example:

```bash
python run_sir_batch.py netscience "edgelist/classical network/netscience_gcc.txt" 0.2 results.json
```

## Network Datasets

### Datasets from the Paper (Table 1)

The following 13 networks were used in the published experiments. All datasets are included in this repository. Original data sourced from [Stanford SNAP](https://snap.stanford.edu/data/).

| Type | Network | N | E | File |
|------|---------|---|---|------|
| Collaboration | ca-AstroPh | 17,903 | 196,972 | `collaboration network/ca_astroph_gcc.txt` |
| Collaboration | ca-CondMat | 21,363 | 91,286 | `collaboration network/ca_condmat_gcc.txt` |
| Collaboration | ca-GrQc | 4,158 | 13,422 | `collaboration network/ca_grqc_gcc.txt` |
| Collaboration | ca-HepPh | 11,204 | 117,619 | `collaboration network/ca_hepph_gcc.txt` |
| Collaboration | ca-HepTh | 8,638 | 24,806 | `collaboration network/ca_hepth_gcc.txt` |
| Social | Jazz-Musicians | 198 | 2,742 | `collaboration network/jazz_musicians_gcc.txt` |
| Communication | Email-Contacts | 12,625 | 20,362 | `communication network/email_contacts_gcc.txt` |
| Communication | Email-Enron | 33,696 | 180,811 | `communication network/email_enron_gcc.txt` |
| Other | C.elegansNeural | 297 | 2,148 | `classical network/celegansneural_gcc.txt` |
| Other | Dolphins | 62 | 159 | `classical network/dolphins_gcc.txt` |
| Other | LesMis | 77 | 254 | `classical network/lesmis_gcc.txt` |
| Other | NetScience | 379 | 914 | `classical network/netscience_gcc.txt` |
| Other | PolBlogs | 1,222 | 16,714 | `classical network/polblogs_gcc.txt` |

### Additional Datasets

| Category | Network | File |
|----------|---------|------|
| Classical | Karate Club | `classical network/karate_gcc.txt` |
| Classical | Football | `classical network/football_list_gcc.txt` |
| Classical | US Air 97 | `classical network/us_air97_gcc.txt` |
| Communication | Email URV | `communication network/email_urv_gcc.txt` |
| Communication | Wiki Vote | `communication network/wiki_vote_gcc.txt` |
| Other | AS-20000102 | `other network/as20000102_gcc.txt` |
| Other | Facebook Combined | `other network/facebook_combined_gcc.txt` |
| Other | Daily Network | `other network/daily_network_gcc.txt` |
| Other | PGP Network | `other network/pgp_network_gcc.txt` |
| Other | Random Geometric | `other network/random_geometric_graph_1000_gcc.txt` |
| Other | 2-Cliques | `other network/2cliques_network_gcc.txt` |
| Other | K-Core Example | `other network/k-core_network_gcc.txt` |
| Theoretical | Barabási–Albert | `theoretical network/ba_n=100_k=5.txt` |
| Theoretical | Erdős–Rényi | `theoretical network/random_n=100_k=5.txt` |
| Theoretical | Regular | `theoretical network/regular_n=100_k=5.txt` |
| Theoretical | Small-World | `theoretical network/sw_n=100_k=5_p=0.1.txt` |

All edgelist files use space-separated integer node pairs, one edge per line.

### Precomputed Auxiliary Files

For most networks, precomputed auxiliary files are included alongside the base edgelist:

| Suffix | Content |
|--------|---------|
| `_tbet.txt` | Betweenness centrality (node, value) |
| `_clos.txt` | Closeness centrality (node, value) |
| `_pos.txt` | Node layout positions (node, x, y) |

These files are generated by the GUI and cached for faster reloading. They can be safely deleted and regenerated.

## Algorithm Overview

```
Load edgelist → Build undirected graph → Extract GCC
  → Step 1: Global Diversity
      → K-shell decomposition for each node
      → Shannon entropy of neighbors' k-shell distribution → E_i
  → Step 2: Local Feature
      → Sum of neighbors' neighbors' degrees
      → log₂ transform → L_i
  → Combine: IF_i = E_i × L_i
  → Validate via SIR epidemic simulation (Monte Carlo, 5000 rounds)
```

## Project Structure

```
MV17/
├── main.py                     # Entry point (GUI)
├── main_window.py              # Main window with 5 tabs
├── run_sir_batch.py            # CLI batch SIR experiment runner
├── gui_app/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── algorithm_adapter.py    # All algorithms (centrality, SIR, I/O)
│   │   ├── network_manager.py     # Shared data manager (Qt Signals)
│   │   └── worker_threads.py      # Background workers (7 types)
│   ├── tabs/
│   │   ├── __init__.py
│   │   ├── tab_network_io.py      # Tab 1: Load/save networks
│   │   ├── tab_network_viz.py     # Tab 2: Network visualization
│   │   ├── tab_node_attributes.py # Tab 3: Compute node metrics
│   │   ├── tab_sir_experiment.py  # Tab 4: SIR propagation simulation
│   │   └── tab_statistics.py      # Tab 5: Statistical analysis
│   └── widgets/
│       ├── __init__.py
│       ├── matplotlib_canvas.py   # Matplotlib-Qt integration
│       └── progress_dialog.py     # Progress dialog for long tasks
├── edgelist/                   # Network datasets (120 files)
│   ├── classical network/      # 9 networks + auxiliary files
│   ├── collaboration network/  # 6 networks + auxiliary files
│   ├── communication network/  # 4 networks + auxiliary files
│   ├── other network/          # 7 networks + auxiliary files
│   └── theoretical network/    # 4 networks + auxiliary files
├── requirements.txt
├── pyproject.toml
├── LICENSE
├── CITATION.cff
├── CONTRIBUTING.md
├── CHANGELOG.md
└── .gitignore
```

## Authors

- **Chung-Yuan Huang** (黃崇源) — Department of Computer Science and Information Engineering, Chang Gung University, Taiwan (gscott@mail.cgu.edu.tw)
- **Yu-Hsiang Fu** (傅小羊) — Department of Computer Science, National Chiao Tung University, Taiwan
- **Chuen-Tsai Sun** (孫春在) — Department of Computer Science, National Chiao Tung University, Taiwan

## Citation

If you use this software in your research, please cite:

> Fu, Y.-H., Huang, C.-Y., & Sun, C.-T. (2015). Using global diversity and local topology features to identify influential network spreaders. *Physica A*, 433, 344–355. https://doi.org/10.1016/j.physa.2015.03.042

See `CITATION.cff` for machine-readable citation metadata.

## References

1. Fu, Y.-H., Huang, C.-Y., & Sun, C.-T. (2015). Using global diversity and local topology features to identify influential network spreaders. *Physica A: Statistical Mechanics and its Applications*, 433, 344–355. https://doi.org/10.1016/j.physa.2015.03.042

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
