# Nanopore protein peak-card analysis

Python scripts for extracting peak positions from nanopore event histograms, comparing event-specific peak cards with reference cards, and clustering avidin–biotin and insulin modification experiments. These scripts use predefined rules and hierarchical clustering; they do not train a machine-learning classifier.

## Contents

| File | Purpose | Main input | Main output |
| --- | --- | --- | --- |
| `peak_extraction_binary_mixture.py` | Extract characteristic blockade peak positions from event histograms in a binary mixture. | Folder of multicolumn CSV histograms | `peak_cards_fwhm.csv` and `peak_cards_fwhm_summary.csv` |
| `binary_standard_card_classifier.py` | Compare each event peak card with IgG and apo-Tf reference cards, report similarity scores and event counts. | Summary CSV with `Event` and `Peak_Positions` columns | `classified_events_with_error.csv` plus diagnostic outputs |
| `single_event_comparison.py` | Compare one event card with user-supplied SA, BSA, IgG and G6PD reference cards. | Peak-position list and reference-card dictionary supplied in Python | Similarity scores and assigned class returned by `classify_single_event()` |
| `clustering_avidin.py` | Cluster peak cards jointly across avidin: biotin ratios to compare cluster composition. | Ratio-labelled CSV files containing `Peak_Positions` and optionally `Event` | `results_unified/` with cluster assignments, centres, counts, plots and event lists |
| `clustering_insulin_ptm.py` | Cluster peak cards jointly across insulin acylation conditions. | Ratio-labelled CSV files containing `Peak_Positions` or `Peaks_str`, optionally `Event`; a supported wide histogram format can also be read | `results_ptm_global/` with assignments, centres, counts, plots and event lists |

## Requirements

- Python 3.9 or later is recommended.
- `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn` and `tabulate`.

Install dependencies with:

```bash
python -m pip install numpy pandas scipy matplotlib scikit-learn tabulate
```

## Input data and workflow

1. Prepare event histogram CSV files for `peak_extraction_binary_mixture.py`. Each file is read without a header. Adjacent columns form pairs: raw current, histogram counts; a file may contain multiple pairs. Set `INPUT_FOLDER`, `I0_VALUE` and the peak filters in the `main()` configuration block before running. Current values are divided by the specified `I0_VALUE` to obtain `I/I0`.
2. Run the extractor and use its `peak_cards_fwhm_summary.csv` output for binary classification. The summary contains `Event` and a bracketed, comma-separated `Peak_Positions` field such as `[0.827, 0.858, 0.908]`. The companion `peak_cards_fwhm.csv` has a different, one-column-per-event layout and is **not** the expected classifier input.
3. In `binary_standard_card_classifier.py`, set `INPUT_FILE` to the summary CSV, choose `OUTPUT_FILE`, and check the IgG and apo-Tf `STANDARD_CARDS` and acceptance parameters. Run the classifier to obtain event-level assignments and count summaries. Cards below the similarity threshold are marked `Uncertain`.
4. For avidin–biotin or insulin experiments, edit each clustering script's `ratio_files_dict` so that its labels point to the corresponding peak-card CSV files. The avidin script includes `1:0`, `1:1`, `1:2`, `1:4` and `1:10` labels; the insulin script includes `0eq`, `1.2eq` and `6eq`. These are dataset configuration labels, not required filenames. The files must contain `Peak_Positions` (the insulin script also accepts `Peaks_str`). `Event` is optional and is generated when absent. Each experiment combines eligible cards across its conditions before agglomerative clustering with average linkage, so cluster IDs can be compared within that experiment.
5. `single_event_comparison.py` is a callable class rather than a ready-made batch CSV program. Supply four reference cards and an event peak list, for example:

```python
from single_event_comparison import QuadPeakCardClassifier

reference_cards = {
    "SA": [],   # replace with measured positions
    "BSA": [],  # replace with measured positions
    "IgG": [],  # replace with measured positions
    "G6PD": [], # replace with measured positions
}
event_peaks = []  # replace with measured event positions

classifier = QuadPeakCardClassifier(tolerance=0.004)
classifier.set_standard_cards(reference_cards)
result = classifier.classify_single_event(event_peaks)
classifier.print_report(result)
```

Replace the placeholder lists with numerical peak positions before running the example.

## Run

After editing the configuration blocks, run the relevant scripts from the repository folder:

```bash
python peak_extraction_binary_mixture.py
python binary_standard_card_classifier.py
python clustering_avidin.py
python clustering_insulin_ptm.py
```

The scripts contain study-specific **absolute Windows paths** to the authors' data. Replace those paths with paths on your own computer before running. Input datasets are not included here; without appropriate CSV inputs the batch scripts cannot reproduce the published outputs. The scripts write output files to the paths or folders set in their configuration blocks. Do not use `single_event_comparison.py` as a command-line batch program; import its class as shown above.

## Parameters and interpretation

The scripts expose peak-position tolerances, smoothing and peak-width filters, excluded baseline ranges, minimum peak counts, similarity thresholds, cluster counts and minimum cluster sizes in their source code. Review those values against the Methods and Supplementary Information before using the scripts on a different dataset. Reference cards embedded in the binary classifier are IgG `[0.827, 0.858, 0.908, 0.931, 0.961, 0.985]` and apo-Tf `[0.655, 0.704, 0.746, 0.793, 0.867, 0.895, 0.948, 0.978]` in units of `I/I0`.

The `single_event_comparison.py` script also returns a softmax transform of the similarity scores. These numbers are **not calibrated probabilities** of protein identity. Cluster labels are algorithmic groupings; interpretation of a cluster as a molecular species requires supporting evidence.

## Data availability and citation

The code alone does not contain the raw ionic-current recordings or all intermediate CSV inputs. Consult the accompanying paper and its data availability statement for access to the experimental data. If you use these scripts, cite the associated paper and identify the GitHub repository and version or commit used for the analysis.
