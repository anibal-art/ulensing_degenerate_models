# LRT paper-figure formatting update

This bundle is a presentation-only refactor of the existing LRT figure scripts.

## Scientific invariants

The update does **not** change:
- datasets or input paths;
- event selections;
- calibrated LRT thresholds;
- bin edges or minimum-count rules;
- Monte-Carlo seeds/sample counts;
- fit parameters or covariance calculations;
- reported statistics.

It changes figure presentation only.

## Global paper style

New helper:

`analysis/lrt/plotting/paper_style.py`

It provides:
- consistent serif/STIX typography;
- journal-scale font sizes and line weights;
- colorblind-safe line cycle;
- constrained layout;
- 300 dpi PNG output;
- vector PDF companions;
- panel labels `(a)`, `(b)`, ... for multi-panel figures.

All existing PNG filenames are preserved for backwards compatibility.
Every updated figure script also writes a PDF with the same stem.

## Layout changes

- `tight_layout()` and figure-level `suptitle` calls were removed.
- Single-panel plot titles were removed; titles belong in the manuscript caption.
- Multi-panel headings were retained where they identify distinct panels.
- Multi-panel figures now receive compact panel labels.
- Optional free-text boxes that could cover data were removed.
- Figure dimensions were normalized for double-column journal use.
- Script 38 was re-laid out so the model/CV information is in a dedicated
  text area rather than on top of the data panel.
- Script 30 was formatted for consistency but remains scientifically
  excluded from the main narrative.

## Installation

From the repository root, first make a safety copy or commit your current work.

Then copy the bundle over the repository tree, preserving paths.

After copying:

```bash
python -m compileall -q analysis/lrt
```

## Regenerating figures

Run the figure scripts from the repository root as before. Existing PNGs are
overwritten with paper-formatted versions and vector PDFs are created next
to them.

Some scripts (notably the MC propagation / flexible-model diagnostics) do
more than simple plotting and may take longer to rerun. Their scientific
configuration was not changed.

## Updating the presentation/manuscript to PDF figures

Use:

```bash
python tools/paperize_tex_figures.py path/to/presentation.tex
```

This changes figure references ending in `.png` to `.pdf` and makes a
`.bak` copy of the TeX file first.

Then compile with:

```bash
bash tools/compile_latex_clean.sh path/to/presentation.tex
```

The compile helper uses `latexmk -halt-on-error` and prints relevant layout
warnings from the log.
