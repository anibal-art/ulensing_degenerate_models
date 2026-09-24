# LRT analysis report

This directory contains the LaTeX report for the LSSTMONTS hidden-parallax likelihood-ratio analysis.

The current `build_figure_manifest.py` contains curated, figure-specific scientific interpretations for the LRT validation figures inspected on 2026-09-22. Future/unknown figures are still included automatically with a generic fallback note.

## Build

From the repository root:

```bash
python analysis/lrt/report/build_figure_manifest.py
cd analysis/lrt/report
pdflatex lrt_analysis_report.tex
pdflatex lrt_analysis_report.tex
```

The report never reruns simulations or fits.
