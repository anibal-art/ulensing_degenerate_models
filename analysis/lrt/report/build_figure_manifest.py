#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a curated LaTeX appendix containing every figure below analysis/lrt/figures.

Current known LRT-validation figures receive figure-specific captions and scientific
interpretations based on visual inspection and the corresponding numerical result tables.
Unknown future figures fall back to a generic caption so the inventory remains complete.

Run:
    python analysis/lrt/report/build_figure_manifest.py
"""

from pathlib import Path
import re

REPORT_DIR = Path(__file__).resolve().parent
LRT_ROOT = REPORT_DIR.parent
FIG_ROOT = LRT_ROOT / "figures"
OUT_DIR = REPORT_DIR / "generated"
OUT_TEX = OUT_DIR / "figure_manifest.tex"
OUT_CSV = OUT_DIR / "figure_manifest.csv"
SUPPORTED = {".png", ".pdf", ".jpg", ".jpeg"}


def latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def humanize(stem: str) -> str:
    s = stem.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s[:1].upper() + s[1:] if s else stem


def section_title(rel_parent: Path) -> str:
    if str(rel_parent) in ("", "."):
        return "Uncategorized figures"
    return " / ".join(humanize(part) for part in rel_parent.parts)


CURATED = {
"lrt_validation/01_h0_survival_vs_chi2/h0_survival_vs_chi2.png": (
    r"Empirical $H_0$ survival function of $\Delta\chi^2$ compared with the $\chi^2_2$ reference.",
    r"The empirical survival follows the asymptotic shape closely but remains systematically below the $\chi^2_2$ curve over the positive tail. This means that the Wilks reference assigns slightly more probability to large $\Delta\chi^2$ than observed in the simulated null population, so the corresponding Wilks critical values are conservative. The step-like structure beyond roughly $\Delta\chi^2\sim 18$ is expected from finite-tail counting: only of order ten null events remain near $\alpha=10^{-4}$."
),
"lrt_validation/01_h0_survival_vs_chi2/h0_survival_ratio.png": (
    r"Ratio of empirical $H_0$ survival to the $\chi^2_2$ survival.",
    r"The ratio is below unity throughout the resolved positive tail. At $c=0$ it starts below one partly because $1.66\%$ of the frozen $H_0$ fits have negative $\Delta\chi^2$. The ratio decreases to about $0.7$--$0.9$ through the scientifically relevant range, quantitatively showing the conservative nature of the $\chi^2_2$ approximation. Large fluctuations in the far-right tail should not be over-interpreted because the empirical survival there is determined by only a handful of events."
),
"lrt_validation/02_bootstrap_thresholds/bootstrap_critical_values.png": (
    r"Bootstrap uncertainty of empirical critical values compared with Wilks thresholds.",
    r"The empirical critical values lie below the $\chi^2_2$ values at every tested false-positive level. At $\alpha=0.05$, $0.01$, and $10^{-3}$ the bootstrap intervals are narrow and do not reach the Wilks value, so the small conservative shift is statistically resolved in the Monte Carlo sample. At $\alpha=10^{-4}$ the interval widens substantially, reflecting the fact that only about ten null events constrain this tail; the Wilks value then falls inside the bootstrap interval. This supports $\alpha=10^{-3}$ as the primary calibrated operating point."
),
"lrt_validation/03_crossvalidated_fpr/crossvalidated_fpr.png": (
    r"Held-out false-positive rate after cross-validated empirical calibration.",
    r"The held-out $H_0$ false-positive rates track the requested target levels over nearly three orders of magnitude. The mean values are $0.04997$, $0.01001$, $0.00102$, and $1.06\times10^{-4}$ for targets $0.05$, $0.01$, $10^{-3}$, and $10^{-4}$, respectively. The relative uncertainty increases toward the deepest tail, but no systematic calibration bias is visible. This is the strongest direct check that the empirical threshold generalizes beyond the exact events used to estimate it."
),
"lrt_validation/04_conditional_h0_stability/n_photometry_points.png": (
    r"Conditional $H_0$ false-positive rate versus total photometric sampling.",
    r"Across five approximately equal-population bins, the false-positive rate ranges from $7.65\times10^{-4}$ to $1.25\times10^{-3}$. There is no monotonic increase or decrease with the total number of photometric points, and every Wilson interval includes the global target $10^{-3}$. The global threshold therefore appears stable against total sampling density within the selected population."
),
"lrt_validation/04_conditional_h0_stability/nearest_peak_distance_tE.png": (
    r"Conditional $H_0$ false-positive rate versus distance of the nearest observation from peak, in units of $t_E$.",
    r"The first four bins fluctuate around $8\times10^{-4}$--$1.1\times10^{-3}$. The poorest-coverage bin has the largest point estimate, $1.25\times10^{-3}$, but its Wilson interval still overlaps the target. The figure therefore suggests at most a weak increase in false positives for the largest peak gaps, not a demonstrated breakdown of the global calibration."
),
"lrt_validation/04_conditional_h0_stability/n_within_0p25_tE.png": (
    r"Conditional $H_0$ false-positive rate versus the number of observations within $0.25t_E$ of peak.",
    r"The five bins remain close to the $10^{-3}$ target, with point estimates between $8.52\times10^{-4}$ and $1.18\times10^{-3}$. No monotonic trend with central-peak sampling is visible, and all confidence intervals overlap the target level."
),
"lrt_validation/04_conditional_h0_stability/n_within_0p5_tE.png": (
    r"Conditional $H_0$ false-positive rate versus the number of observations within $0.5t_E$ of peak.",
    r"All bins are statistically compatible with the global target. The lowest point estimate, $6.66\times10^{-4}$, occurs in an intermediate sampling bin rather than at an extreme, while the other bins lie near $10^{-3}$. This argues against a simple sampling-driven false-positive mechanism within $0.5t_E$."
),
"lrt_validation/04_conditional_h0_stability/n_within_1_tE.png": (
    r"Conditional $H_0$ false-positive rate versus the number of observations within $t_E$ of peak.",
    r"This is the flattest of the conditional-sampling diagnostics: all five point estimates lie between $8.94\times10^{-4}$ and $1.15\times10^{-3}$. The global threshold is therefore very stable with respect to the amount of sampling inside the principal $\pm t_E$ event window."
),
"lrt_validation/04_conditional_h0_stability/n_within_2_tE.png": (
    r"Conditional $H_0$ false-positive rate versus the number of observations within $2t_E$ of peak.",
    r"The point estimates show some non-monotonic variation, from about $7.2\times10^{-4}$ in sparsely sampled bins to $1.41\times10^{-3}$ in the highest-sampling bin. The highest bin has a Wilson interval whose lower edge is essentially the target value, so this is not compelling evidence for a systematic dependence. The absence of monotonicity also disfavors a simple causal interpretation."
),
"lrt_validation/04_conditional_h0_stability/n_left_within_1_tE.png": (
    r"Conditional $H_0$ false-positive rate versus pre-peak sampling within $t_E$.",
    r"The false-positive fraction remains close to $10^{-3}$ across all left-side sampling bins. The largest point estimate, $1.28\times10^{-3}$, occurs in an intermediate bin and its interval overlaps the target. There is no evidence that poor pre-peak sampling alone drives the calibrated null tail."
),
"lrt_validation/04_conditional_h0_stability/n_right_within_1_tE.png": (
    r"Conditional $H_0$ false-positive rate versus post-peak sampling within $t_E$.",
    r"The measured rates range from $7.26\times10^{-4}$ to $1.32\times10^{-3}$ without a monotonic ordering. All Wilson intervals include the target $10^{-3}$. This indicates that asymmetric post-peak sampling does not produce a detectable calibration failure at the current sample size."
),
"lrt_validation/04_conditional_h0_stability/poor_peak_coverage.png": (
    r"Conditional $H_0$ false-positive rate for the binary poor-peak-coverage diagnostic.",
    r"The well-covered population gives $96/99529=9.65\times10^{-4}$, essentially the target rate. The poor-coverage population gives $7/4408=1.59\times10^{-3}$, a larger point estimate, but the interval is wide ($7.69\times10^{-4}$ to $3.28\times10^{-3}$) because only seven false positives occur there. The figure is compatible with a modest enrichment but does not establish one."
),
"lrt_validation/05_negative_lrt_diagnostics/h0_poor_peak_coverage_negative_fraction.png": (
    r"Fraction of negative LRT values in the $H_0$ population, split by poor peak coverage.",
    r"The negative fraction changes only from $1.653\%$ for well-covered events to $1.747\%$ for poor-coverage events, with strongly overlapping confidence intervals. Thus peak coverage does not explain the $H_0$ nesting violations. The separate numeric diagnostics instead show substantially worse optimizer optimality for negative $H_0$ events, pointing toward numerical/local-minimum effects."
),
"lrt_validation/05_negative_lrt_diagnostics/h1_poor_peak_coverage_negative_fraction.png": (
    r"Fraction of negative LRT values in the $H_1$ population, split by poor peak coverage.",
    r"This is a strong effect: the negative-LRT fraction rises from $0.800\%$ for well-covered events to $2.989\%$ for poor-coverage events, a factor of about $3.7$. The confidence intervals are well separated. Sampling therefore plays a major role in the $H_1$ nesting violations, although only $422$ of $2978$ negative events satisfy the hard poor-coverage flag, so the binary flag captures only part of a broader continuous degradation in sampling quality."
),
"lrt_validation/05_negative_lrt_diagnostics/h0_h0_optimizer_success_negative_fraction.png": (
    r"Negative-LRT fraction in $H_0$ versus the $H_0$ optimizer-success flag.",
    r"All production events have \texttt{h0\_optimizer\_success=True}; consequently the figure contains only one category and cannot test whether the success flag predicts nesting violations. Its main message is negative: the optimizer can report success while the nested-model ordering is still violated. This panel is useful as an audit but not as a main scientific figure."
),
"lrt_validation/05_negative_lrt_diagnostics/h0_h1_optimizer_success_negative_fraction.png": (
    r"Negative-LRT fraction in $H_0$ versus the $H_1$ optimizer-success flag.",
    r"The success flag is identically true, so this single-point figure has no discriminating power. Negative $H_0$ LRT values therefore occur despite nominally successful $H_1$ optimization. The more informative diagnostics are optimizer optimality, evaluation count, and direct inspection of the fitted solutions."
),
"lrt_validation/05_negative_lrt_diagnostics/h1_h0_optimizer_success_negative_fraction.png": (
    r"Negative-LRT fraction in $H_1$ versus the $H_0$ optimizer-success flag.",
    r"All $H_0$ fits are marked successful, leaving only one category. The figure demonstrates that the binary success flag is too coarse to diagnose the $H_1$ nesting violations and should not be used as a quality cut by itself."
),
"lrt_validation/05_negative_lrt_diagnostics/h1_h1_optimizer_success_negative_fraction.png": (
    r"Negative-LRT fraction in $H_1$ versus the $H_1$ optimizer-success flag.",
    r"All $H_1$ fits are marked successful, including the $2978$ cases with negative $\Delta\chi^2$. The panel therefore rules out only explicit optimizer failure; it does not rule out convergence to a local basin or a solution that is insufficiently optimized relative to the nested $H_0$ fit."
),
"lrt_validation/07_roc_curve/roc_full.png": (
    r"Full empirical ROC curve using $\Delta\chi^2$ to distinguish the $H_1$ and $H_0$ populations.",
    r"The curve lies far above the random-classifier diagonal and has AUC $=0.8983$, demonstrating strong population-level separation. AUC is a threshold-independent ranking metric and is not itself the criterion used for detection; the scientific operating points are set by the empirically calibrated $H_0$ false-positive levels."
),
"lrt_validation/07_roc_curve/roc_low_fpr.png": (
    r"Low-false-positive-rate region of the empirical ROC curve with calibrated operating points.",
    r"This panel is the operationally relevant ROC view. The calibrated points give true-positive rates of $0.7751$, $0.7293$, $0.6891$, and $0.6576$ at false-positive levels $0.05$, $0.01$, $10^{-3}$, and $10^{-4}$, respectively. The curve changes relatively slowly across the low-FPR regime, so tightening the false-positive requirement by orders of magnitude reduces conditional power by only about twelve percentage points from $0.05$ to $10^{-4}$. Results at or below $10^{-5}$ are limited by the finite $H_0$ sample resolution."
),
"lrt_validation/08_lrt_vs_wald_piE/lrt_vs_wald_piE.png": (
    r"Global LRT statistic versus the local covariance-based Wald significance of the fitted parallax vector.",
    r"The two statistics are strongly correlated, with Pearson correlation $r=0.788$ in log space, and the high-density ridge approaches the one-to-one relation at large significance. However, the cloud spans several orders of magnitude vertically at fixed $\Delta\chi^2$, especially at low and intermediate significance. This demonstrates that a local Gaussian covariance measure and a global nested-model improvement are not interchangeable. The discrepant tails are a natural target for a dedicated outlier analysis."
),
}


if not FIG_ROOT.exists():
    raise FileNotFoundError(f"Figure root not found: {FIG_ROOT}")

figures = sorted(
    p for p in FIG_ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() in SUPPORTED
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
lines = []
csv_lines = ["index,relative_path,group,stem,curated"]
current_group = None

for i, path in enumerate(figures, start=1):
    rel_from_fig = path.relative_to(FIG_ROOT)
    key = rel_from_fig.as_posix()
    group = rel_from_fig.parent
    rel_from_report = Path("..") / "figures" / rel_from_fig

    if group != current_group:
        current_group = group
        lines += ["", r"\subsection{" + latex_escape(section_title(group)) + "}", ""]

    if key in CURATED:
        caption, interpretation = CURATED[key]
        curated = True
    else:
        caption = latex_escape(humanize(path.stem)) + "."
        interpretation = (
            r"This figure was found automatically in the repository but does not yet have a "
            r"curated scientific interpretation in the manifest. Cross-check it against the "
            r"corresponding numerical output under \texttt{analysis/lrt/results/}."
        )
        curated = False

    lines.extend([
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.94\textwidth]{" + str(rel_from_report).replace("\\", "/") + "}",
        r"\caption{" + caption + "}",
        r"\label{fig:auto-" + str(i) + "}",
        r"\end{figure}",
        "",
        r"\noindent\textbf{Interpretation.} " + interpretation,
        "",
        r"\FloatBarrier",
        "",
    ])

    csv_lines.append(f'{i},"{key}","{group.as_posix()}","{path.stem}",{int(curated)}')

OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
OUT_CSV.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

print(f"Figures found: {len(figures)}")
print(f"Curated figures: {sum(1 for p in figures if p.relative_to(FIG_ROOT).as_posix() in CURATED)}")
print(f"Wrote: {OUT_TEX}")
print(f"Wrote: {OUT_CSV}")
