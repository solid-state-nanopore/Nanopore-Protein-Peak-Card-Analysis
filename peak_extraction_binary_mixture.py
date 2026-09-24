"""
Nanopore full-point distribution peak extractor (FWHM threshold version).
Processes multi-column alternating format: Bin Center(1), Histo(1), Bin Center(2), Histo(2), ...
Uses relative threshold to filter small fluctuations, strictly excludes baseline region,
and adds FWHM thresholds.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, peak_widths
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class NanoporePeakExtractor:
    """
    Nanopore data peak extractor.
    Supports multi-column alternating format and I/I0 normalization.
    Uses relative threshold and FWHM thresholds to filter small fluctuations.
    """

    def __init__(self, smoothing_method='savgol', window_length=11, polyorder=3):
        """
        Initialize peak extractor.
        """
        self.smoothing_method = smoothing_method
        self.window_length = window_length
        self.polyorder = polyorder
        self.I0_value = None

    def load_multicolumn_histogram(self, file_path, skiprows=0, delimiter=','):
        """
        Load multi-column alternating full-point distribution data.
        """
        try:
            try:
                data = pd.read_csv(file_path, delimiter=delimiter,
                                   skiprows=skiprows, header=None,
                                   encoding='utf-8', engine='python')
            except:
                data = pd.read_csv(file_path, delimiter=delimiter,
                                   skiprows=skiprows, header=None,
                                   encoding='gbk', engine='python')

            print(f"  File {Path(file_path).name} has {data.shape[1]} columns, {data.shape[0]} rows")

            if data.shape[1] % 2 != 0:
                print(f"  Warning: number of columns {data.shape[1]} is not even; format may be unexpected")

            histograms = []
            n_pairs = data.shape[1] // 2

            for i in range(n_pairs):
                current_col = i * 2
                counts_col = i * 2 + 1

                current_raw = data.iloc[:, current_col].values
                counts_raw = data.iloc[:, counts_col].values

                current = pd.to_numeric(current_raw, errors='coerce')
                counts = pd.to_numeric(counts_raw, errors='coerce')

                valid_mask = ~(np.isnan(current) | np.isnan(counts))
                current_valid = current[valid_mask]
                counts_valid = counts[valid_mask]

                positive_mask = (current_valid > 0) & (counts_valid > 0)
                current_valid = current_valid[positive_mask]
                counts_valid = counts_valid[positive_mask]

                if len(current_valid) > 0 and len(counts_valid) > 0:
                    histograms.append({
                        'pair_index': i + 1,
                        'current_raw': current_valid,
                        'counts': counts_valid,
                        'n_points': len(current_valid)
                    })
                    print(f"    Pair {i+1}: {len(current_valid)} valid data points")

            print(f"  Successfully extracted {len(histograms)} valid histogram datasets")
            return histograms

        except Exception as e:
            print(f"  Error loading file {file_path}: {e}")
            return None

    def normalize_current(self, current_raw, I0):
        """Normalize current values to I/I0."""
        return current_raw / I0

    def calculate_fwhm(self, current, counts, peak_idx):
        """
        Calculate full width at half maximum (FWHM).

        Parameters
        ----------
        current : numpy array
            Current values.
        counts : numpy array
            Count array.
        peak_idx : int
            Peak index.

        Returns
        -------
        fwhm : float
            FWHM expressed in current values.
        """
        try:
            peak_height = counts[peak_idx]
            half_height = peak_height / 2

            left_idx = peak_idx
            while left_idx > 0 and counts[left_idx] > half_height:
                left_idx -= 1

            right_idx = peak_idx
            while right_idx < len(counts) - 1 and counts[right_idx] > half_height:
                right_idx += 1

            if left_idx > 0:
                y1, y2 = counts[left_idx], counts[left_idx + 1]
                x1, x2 = current[left_idx], current[left_idx + 1]
                if y2 > y1:
                    left_current = x1 + (half_height - y1) * (x2 - x1) / (y2 - y1)
                else:
                    left_current = current[left_idx]
            else:
                left_current = current[0]

            if right_idx < len(counts) - 1:
                y1, y2 = counts[right_idx - 1], counts[right_idx]
                x1, x2 = current[right_idx - 1], current[right_idx]
                if y2 < y1:
                    right_current = x1 + (half_height - y1) * (x2 - x1) / (y2 - y1)
                else:
                    right_current = current[right_idx]
            else:
                right_current = current[-1]

            fwhm = right_current - left_current
            return abs(fwhm)

        except:
            return 0

    def process_single_histogram(self, current, counts,
                                 max_peaks=15,
                                 baseline_exclude_range=[0.98, 1.05],
                                 height_threshold_ratio=0.3,
                                 min_absolute_height=10,
                                 min_fwhm=0.005,
                                 max_fwhm=0.05):
        """
        Process a single histogram and extract peaks.
        Only finds local maxima (increase followed by decrease), then filters by
        relative threshold and FWHM.

        Parameters
        ----------
        current : numpy array
            Current values (I/I0).
        counts : numpy array
            Count array.
        max_peaks : int
            Maximum number of returned peaks.
        baseline_exclude_range : list
            Baseline exclusion range [min, max].
        height_threshold_ratio : float
            Relative height threshold (percentage of the highest peak).
        min_absolute_height : float
            Minimum absolute height threshold.
        min_fwhm : float
            Minimum FWHM threshold; peaks below this are considered noise.
        max_fwhm : float
            Maximum FWHM threshold; peaks above this may be too broad.
        """
        if len(counts) >= self.window_length:
            counts_smoothed = savgol_filter(counts, self.window_length, self.polyorder)
            counts_smoothed = np.maximum(counts_smoothed, 0)
        else:
            counts_smoothed = counts.copy()

        peaks = []
        diff = np.diff(counts_smoothed)

        for i in range(1, len(diff)):
            if diff[i-1] > 0 and diff[i] < 0:
                peaks.append(i)

        if len(peaks) == 0:
            return []

        peak_values = current[peaks]
        peak_heights = counts_smoothed[peaks]

        peak_fwhms = []
        valid_peaks_indices = []

        for idx, peak_idx in enumerate(peaks):
            fwhm = self.calculate_fwhm(current, counts_smoothed, peak_idx)
            peak_fwhms.append(fwhm)
            valid_peaks_indices.append(idx)

        peak_fwhms = np.array(peak_fwhms)

        non_baseline_mask = ~((peak_values >= baseline_exclude_range[0]) &
                              (peak_values <= baseline_exclude_range[1]))

        if not np.any(non_baseline_mask):
            return []

        fwhm_mask = (peak_fwhms >= min_fwhm) & (peak_fwhms <= max_fwhm)

        combined_mask = non_baseline_mask & fwhm_mask

        if not np.any(combined_mask):
            return []

        candidate_values = peak_values[combined_mask]
        candidate_heights = peak_heights[combined_mask]
        candidate_fwhms = peak_fwhms[combined_mask]

        if len(candidate_heights) > 0:
            max_height = np.max(candidate_heights)
            relative_threshold = max_height * height_threshold_ratio
            height_threshold = max(relative_threshold, min_absolute_height)

            height_mask = candidate_heights >= height_threshold
            final_peak_values = candidate_values[height_mask]
            final_peak_heights = candidate_heights[height_mask]
            final_peak_fwhms = candidate_fwhms[height_mask]
        else:
            return []

        if len(final_peak_values) == 0:
            return []

        if len(final_peak_values) < len(candidate_values):
            print(f"    FWHM filtering: {len(candidate_values)} -> {len(final_peak_values)} peaks")

        if len(final_peak_values) > max_peaks:
            top_indices = np.argsort(final_peak_heights)[-max_peaks:]
            final_peak_values = final_peak_values[top_indices]

        final_peak_values = np.sort(final_peak_values)

        return final_peak_values.tolist()

    def batch_process_folder(self, input_folder, output_file='peak_cards.csv',
                             file_pattern='*.csv',
                             I0_value=None,
                             skiprows=0,
                             delimiter=',',
                             select_pair=None,
                             max_peaks=15,
                             baseline_exclude_range=[0.98, 1.05],
                             height_threshold_ratio=0.3,
                             min_absolute_height=10,
                             min_fwhm=0.005,
                             max_fwhm=0.05):
        """
        Batch-process all data files in a folder.
        """
        if I0_value is None:
            print("Error: I0_value must be provided for normalization!")
            return None

        self.I0_value = I0_value
        print(f"Using I0 = {I0_value} for normalization")
        print(f"Peak detection parameters: max peaks = {max_peaks}")
        print(f"Baseline exclusion range: {baseline_exclude_range}")
        print(f"Height thresholds: relative = {height_threshold_ratio*100}% of maximum peak, absolute = {min_absolute_height}")
        print(f"FWHM thresholds: min = {min_fwhm}, max = {max_fwhm}")

        folder_path = Path(input_folder)
        file_list = list(folder_path.glob(file_pattern))

        if len(file_list) == 0:
            print(f"No files matching {file_pattern} found in {input_folder}")
            return None

        print(f"Found {len(file_list)} data files; starting processing...")

        all_cards = {}
        all_cards_detailed = []

        for i, file_path in enumerate(file_list):
            event_name = file_path.stem
            print(f"\nProcessing file {i+1}/{len(file_list)}: {event_name}")

            histograms = self.load_multicolumn_histogram(
                file_path,
                skiprows=skiprows,
                delimiter=delimiter
            )

            if histograms is None:
                continue

            pairs_to_process = range(len(histograms))
            if select_pair is not None:
                if 1 <= select_pair <= len(histograms):
                    pairs_to_process = [select_pair - 1]
                else:
                    print(f"  Warning: selected histogram pair {select_pair} does not exist; processing all pairs")

            for idx in pairs_to_process:
                hist = histograms[idx]

                current_norm = self.normalize_current(hist['current_raw'], I0_value)

                peak_card = self.process_single_histogram(
                    current_norm, hist['counts'],
                    max_peaks=max_peaks,
                    baseline_exclude_range=baseline_exclude_range,
                    height_threshold_ratio=height_threshold_ratio,
                    min_absolute_height=min_absolute_height,
                    min_fwhm=min_fwhm,
                    max_fwhm=max_fwhm
                )

                if len(histograms) > 1:
                    hist_event_name = f"{event_name}_pair{hist['pair_index']}"
                else:
                    hist_event_name = event_name

                all_cards[hist_event_name] = peak_card

                all_cards_detailed.append({
                    'Event': hist_event_name,
                    'File': file_path.name,
                    'Pair': hist['pair_index'],
                    'Num_Peaks': len(peak_card),
                    'Peak_Positions': str(peak_card),
                    'Mean_Peak': np.mean(peak_card) if peak_card else np.nan,
                    'Min_Peak': min(peak_card) if peak_card else np.nan,
                    'Max_Peak': max(peak_card) if peak_card else np.nan,
                    'I0_used': I0_value
                })

                print(f"  Pair {hist['pair_index']}: extracted {len(peak_card)} peaks: {peak_card}")

        if all_cards:
            self.save_to_csv(all_cards, output_file, I0_value, baseline_exclude_range, min_fwhm, max_fwhm)
            self.save_detailed_summary(all_cards_detailed, output_file)
            print(f"\nSuccessfully processed {len(all_cards)} events/histogram pairs")
        else:
            print("\nNo peak data were successfully extracted")

        return all_cards

    def save_to_csv(self, all_cards, output_file, I0_value, baseline_range, min_fwhm, max_fwhm):
        """Save all peak cards to a CSV file."""
        max_len = max(len(card) for card in all_cards.values())

        data = {}
        for event_name, card in all_cards.items():
            padded_card = card + [np.nan] * (max_len - len(card))
            data[event_name] = padded_card

        df = pd.DataFrame(data)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# I/I0 normalization using I0 = {I0_value}\n")
            f.write(f"# Baseline exclusion range: {baseline_range}; values in this range were completely removed.\n")
            f.write(f"# FWHM thresholds: min={min_fwhm}, max={max_fwhm}\n")
            f.write(f"# Each column is one event/histogram pair; each row is a peak position.\n")

        df.to_csv(output_file, mode='a', index=False, encoding='utf-8')
        print(f"\nResults saved to: {output_file}")

    def save_detailed_summary(self, detailed_data, output_file):
        """Save detailed summary."""
        if not detailed_data:
            return

        df = pd.DataFrame(detailed_data)
        summary_file = output_file.replace('.csv', '_summary.csv')
        df.to_csv(summary_file, index=False, encoding='utf-8')
        print(f"Detailed summary saved to: {summary_file}")


def main():
    """
    Main function: run peak extraction workflow.
    """
    print("=" * 60)
    print("Nanopore single-molecule event peak extraction system (FWHM threshold version)")
    print("=" * 60)

    # ======================================================
    # User configuration
    # ======================================================

    INPUT_FOLDER = r"G:\数据\2026年\Supplementary experiment\Supplementary experiment nanopore\0330-15nm-SiO2-Cu-Fe-IgG apo-Tf-good-yes\2 50-50 Tf-IgG-good\0401\26401017events-2"

    OUTPUT_FILE = "peak_cards_fwhm.csv"

    FILE_PATTERN = "*.csv"

    I0_VALUE = 7300.0

    SKIPROWS = 0
    DELIMITER = ','

    SELECT_PAIR = None

    MAX_PEAKS = 15

    BASELINE_EXCLUDE_RANGE = [0.98, 1.05]

    HEIGHT_THRESHOLD_RATIO = 0.1
    MIN_ABSOLUTE_HEIGHT = 10

    MIN_FWHM = 0.003
    MAX_FWHM = 0.03

    # ======================================================
    # The code below normally does not need modification
    # ======================================================

    import os
    if not os.path.exists(INPUT_FOLDER):
        print(f"\nError: folder {INPUT_FOLDER} does not exist!")
        return

    extractor = NanoporePeakExtractor(
        smoothing_method='savgol',
        window_length=11,
        polyorder=3
    )

    print(f"\nInput folder: {INPUT_FOLDER}")
    print(f"File pattern: {FILE_PATTERN}")
    print(f"I0 value: {I0_VALUE}")
    print(f"Delimiter: '{DELIMITER}'")
    print(f"Histogram pairs to process: {'all' if SELECT_PAIR is None else f'pair {SELECT_PAIR}'}")

    all_cards = extractor.batch_process_folder(
        input_folder=INPUT_FOLDER,
        output_file=OUTPUT_FILE,
        file_pattern=FILE_PATTERN,
        I0_value=I0_VALUE,
        skiprows=SKIPROWS,
        delimiter=DELIMITER,
        select_pair=SELECT_PAIR,
        max_peaks=MAX_PEAKS,
        baseline_exclude_range=BASELINE_EXCLUDE_RANGE,
        height_threshold_ratio=HEIGHT_THRESHOLD_RATIO,
        min_absolute_height=MIN_ABSOLUTE_HEIGHT,
        min_fwhm=MIN_FWHM,
        max_fwhm=MAX_FWHM
    )

    if all_cards:
        print(f"\nProcessing complete! Results saved.")
        print(f"  1. {OUTPUT_FILE}")
        print(f"  2. {OUTPUT_FILE.replace('.csv', '_summary.csv')}")
        print(f"\nParameter summary:")
        print(f"  - Baseline exclusion: {BASELINE_EXCLUDE_RANGE}")
        print(f"  - Height threshold: {HEIGHT_THRESHOLD_RATIO*100}% of maximum peak or {MIN_ABSOLUTE_HEIGHT} counts")
        print(f"  - FWHM thresholds: {MIN_FWHM} ~ {MAX_FWHM}")
    else:
        print("\nProcessing failed. Please check input files and parameter settings.")


if __name__ == "__main__":
    main()
