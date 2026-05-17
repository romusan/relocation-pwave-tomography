# Fast-marching earthquake relocation and P-wave tomography in the Middle Magdalena Valley

This repository contains the reproducibility package associated with the manuscript:

**Fast-marching earthquake relocation and P-wave tomography constrain a seismicity-defined Nazca Benioff zone beneath the Middle Magdalena Valley, Colombia**

The package includes the manuscript source, final paper PDF, Python scripts, processed tables, diagnostic metrics, and figures used to support the results. The workflow applies an alternating Fast-Marching earthquake-relocation and P-wave tomography approach to an independent 2022-2026 Servicio Geologico Colombiano earthquake subset.

## Repository structure

- `paper/`: LaTeX manuscript source, compiled PDF, and bibliography.
- `code/`: Python scripts used to generate the processed results and figures.
- `codigos_finales/`: same final scripts retained for compatibility with the local working folder.
- `data/`: processed CSV/JSON/TEX tables derived from the earthquake catalog and inversion outputs.
- `figures/`: manuscript figures and diagnostic figures.
- `metrics/`: JSON summary metrics used in the manuscript.

Raw SGC XML event files and waveform data are not included because they are large and are publicly obtainable from the Servicio Geologico Colombiano services. The repository provides processed tables and scripts sufficient to audit the calculations and regenerate the manuscript diagnostics from the project workspace.

## Main manuscript files

- `paper/joint_vp_relocation_sgc_new_events_q1_draft.tex`
- `paper/joint_vp_relocation_sgc_new_events_q1_draft.pdf`
- `paper/references_q1.bib`

## Main scripts

- `code/run_sgc_new_events_joint_vp_relocation.py`
- `code/run_sgc_new_events_joint_vp_relocation_1d_sensitivity.py`
- `code/analyze_current_dataset_1d_sensitivity.py`
- `code/analyze_location_uncertainty_station_diagnostics.py`
- `code/analyze_rms_improvement_examples.py`
- `code/generate_nazca_benioff_vp_surface.py`

Additional scripts related to waveform download and the companion Q/attenuation workflow are retained for traceability.

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The scripts were developed with Python 3.10+ and require the scientific Python stack listed in `requirements.txt`.

## Reproducibility notes

The core processed outputs are already included in `data/`, `figures/`, and `metrics/`. To reproduce specific diagnostics, run the relevant scripts from the repository root, for example:

```bash
python code/analyze_rms_improvement_examples.py
python code/analyze_location_uncertainty_station_diagnostics.py
python code/analyze_current_dataset_1d_sensitivity.py
python code/generate_nazca_benioff_vp_surface.py
```

Some full workflow scripts expect the original local project layout and public SGC services to be available. The processed tables included here allow the reviewer to inspect the numerical basis of the manuscript without downloading the raw XML or waveform archive.

## Data sources

Earthquake metadata, waveform access, and station metadata are available from the Servicio Geologico Colombiano public services. Slab2 comparison uses the South America Slab2 model of Hayes et al. (2018), as cited in the manuscript.

## Citation

Please cite the archived release of this repository:

```text
Author names removed for review. (2026). Fast-marching earthquake relocation and P-wave tomography in the Middle Magdalena Valley, Colombia (v1.0.0). GitHub. https://github.com/romusan/relocation-pwave-tomography
```

If the GitHub repository is archived with Zenodo, cite the Zenodo DOI generated from release `v1.0.0`.
