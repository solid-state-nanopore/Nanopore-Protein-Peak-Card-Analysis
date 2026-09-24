"""
Avidin-biotin titration experiment: unified cross-ratio clustering analysis
and export of cluster event lists for each ratio.

- Global clustering: all ratios are pooled, so cluster IDs are comparable across ratios.
- Export event lists for each cluster in each ratio to CSV.
- Retain cluster distribution plots and cluster rug plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

CLUSTER_COLORS = {
    0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728',
    4: '#9467bd', 5: '#8c564b', 6: '#e377c2', 7: '#7f7f7f',
    8: '#bcbd22', 9: '#17becf'
}


class UnifiedClusterAnalysis:
    def __init__(self, tolerance=0.0015, n_clusters=6, min_peaks=1,
                 peak_min=0.75, peak_max=1.05, exclude_baseline=True,
                 peak_count_penalty_weight=0.6,
                 min_cluster_size=5, min_per_ratio_ratio=0.05,
                 merge_peaks_tolerance=None):
        self.tolerance = tolerance
        self.n_clusters = n_clusters
        self.min_peaks = min_peaks
        self.peak_min = peak_min
        self.peak_max = peak_max
        self.exclude_baseline = exclude_baseline
        self.peak_count_penalty_weight = peak_count_penalty_weight
        self.min_cluster_size = min_cluster_size
        self.min_per_ratio_ratio = min_per_ratio_ratio
        self.merge_peaks_tolerance = merge_peaks_tolerance if merge_peaks_tolerance is not None else tolerance

        self.all_peaks = []
        self.all_ratios = []
        self.event_ids = []
        self.raw_cluster_labels = None
        self.cluster_labels = None
        self.cluster_centers = []

    def load_all_ratios_from_files(self, ratio_files_dict):
        """Load data files for all ratios."""
        for ratio, file_list in ratio_files_dict.items():
            for file_path in file_list:
                file_path = Path(file_path)
                if not file_path.exists():
                    print(f"Warning: file does not exist: {file_path}")
                    continue
                try:
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                except:
                    df = pd.read_csv(file_path, encoding='utf-8')
                if 'Peak_Positions' not in df.columns:
                    print(f"Warning: {file_path} is missing Peak_Positions column; skipping")
                    continue
                if 'Event' not in df.columns:
                    df['Event'] = [f"{ratio}_{i}" for i in range(len(df))]
                for _, row in df.iterrows():
                    event_id = str(row['Event'])
                    peak_str = str(row['Peak_Positions']).strip('[]')
                    if not peak_str or peak_str == 'nan':
                        continue
                    try:
                        peaks = np.array([float(x.strip()) for x in peak_str.split(',')])
                    except:
                        continue
                    peaks = peaks[peaks >= self.peak_min]
                    peaks = peaks[peaks <= self.peak_max]
                    if self.exclude_baseline:
                        peaks = peaks[~((peaks >= 0.98) & (peaks <= 1.02))]
                    if len(peaks) < self.min_peaks:
                        continue
                    self.all_peaks.append(np.sort(peaks))
                    self.all_ratios.append(ratio)
                    self.event_ids.append(event_id)
        print(f"Loaded {len(self.all_peaks)} events in total")
        print(f"Event counts by ratio: {Counter(self.all_ratios)}")
        return len(self.all_peaks)

    def _peaks_distance(self, p1, p2):
        p1, p2 = np.sort(p1), np.sort(p2)
        max_len = max(len(p1), len(p2))
        if max_len == 0:
            return 1.0
        penalty = self.peak_count_penalty_weight * (abs(len(p1)-len(p2)) / max_len)
        matches = []
        used2 = set()
        for v1 in p1:
            for idx2, v2 in enumerate(p2):
                if idx2 in used2:
                    continue
                if abs(v1 - v2) <= self.tolerance:
                    matches.append((v1, v2))
                    used2.add(idx2)
                    break
        if len(matches) == 0:
            return 1.0
        match_ratio = len(matches) / min(len(p1), len(p2))
        pos_diff = np.mean([abs(m[0]-m[1]) for m in matches]) / (self.tolerance + 1e-6)
        pos_sim = 1.0 / (1.0 + pos_diff)
        similarity = 0.7 * match_ratio + 0.3 * pos_sim
        distance = 1.0 - similarity + penalty
        return min(max(distance, 0.0), 1.0)

    def compute_distance_matrix(self):
        n = len(self.all_peaks)
        dist = np.zeros((n, n))
        print(f"Computing distance matrix ({n}x{n})...")
        for i in range(n):
            if i % 100 == 0:
                print(f"  Progress: {i}/{n}")
            for j in range(i+1, n):
                d = self._peaks_distance(self.all_peaks[i], self.all_peaks[j])
                dist[i, j] = d
                dist[j, i] = d
        return dist

    def perform_clustering(self):
        dist_matrix = self.compute_distance_matrix()
        print(f"Using hierarchical clustering, target number of clusters = {self.n_clusters}")
        clustering = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric='precomputed',
            linkage='average'
        )
        self.raw_cluster_labels = clustering.fit_predict(dist_matrix)
        self._filter_small_clusters()
        self._filter_low_ratio_per_ratio()
        self._compute_cluster_centers()
        self._merge_peaks_in_centers()
        return self.cluster_labels

    def _filter_small_clusters(self):
        if self.raw_cluster_labels is None:
            raise ValueError("Please run perform_clustering() first")
        labels = self.raw_cluster_labels.copy()
        cluster_counts = Counter(labels)
        small_clusters = {cl for cl, cnt in cluster_counts.items() if cnt < self.min_cluster_size}
        filtered_labels = np.where(np.isin(labels, list(small_clusters)), -1, labels)
        self.cluster_labels = filtered_labels
        unique_valid = sorted([cl for cl in np.unique(filtered_labels) if cl != -1])
        label_mapping = {old: new for new, old in enumerate(unique_valid)}
        self.cluster_labels = np.array([label_mapping.get(lbl, -1) for lbl in filtered_labels])
        removed = np.sum(self.cluster_labels == -1)
        print(f"\n[Global filtering] Removing clusters with total events < {self.min_cluster_size}:")
        for cl in small_clusters:
            print(f"  Original cluster {cl} (total {cluster_counts[cl]} events) removed")
        print(f"  Removed {removed} events; remaining valid clusters: {len(unique_valid)}")

    def _filter_low_ratio_per_ratio(self):
        if self.cluster_labels is None:
            return
        current_labels = self.cluster_labels.copy()
        unique_clusters = np.unique(current_labels)
        unique_clusters = unique_clusters[unique_clusters >= 0]
        if len(unique_clusters) == 0:
            print("No valid clusters; skipping ratio-level filtering")
            return
        ratio_total_counts = Counter(self.all_ratios)
        n_events = len(self.all_peaks)
        mask_to_remove = np.zeros(n_events, dtype=bool)
        for cl in unique_clusters:
            idxs_cl = np.where(current_labels == cl)[0]
            if len(idxs_cl) == 0:
                continue
            ratios_in_cl = [self.all_ratios[i] for i in idxs_cl]
            ratio_counts = Counter(ratios_in_cl)
            for r, cnt in ratio_counts.items():
                total_in_ratio = ratio_total_counts[r]
                ratio_val = cnt / total_in_ratio if total_in_ratio > 0 else 0
                if ratio_val < self.min_per_ratio_ratio:
                    idxs_remove = [i for i in idxs_cl if self.all_ratios[i] == r]
                    mask_to_remove[idxs_remove] = True
                    print(f"  Cluster {cl} accounts for {ratio_val*100:.1f}% in ratio {r} (<{self.min_per_ratio_ratio*100}%); these events will be removed")
        new_labels = current_labels.copy()
        new_labels[mask_to_remove] = -1
        unique_new_valid = sorted([cl for cl in np.unique(new_labels) if cl != -1])
        label_mapping = {old: new for new, old in enumerate(unique_new_valid)}
        self.cluster_labels = np.array([label_mapping.get(lbl, -1) for lbl in new_labels])
        removed_second = np.sum(mask_to_remove)
        print(f"\n[Ratio-level filtering] Removed {removed_second} events")
        print(f"  Remaining valid clusters: {len(unique_new_valid)}")

    def _compute_cluster_centers(self):
        unique = np.unique(self.cluster_labels)
        unique = unique[unique >= 0]
        centers = []
        for lab in unique:
            idxs = np.where(self.cluster_labels == lab)[0]
            peaks_list = [self.all_peaks[i] for i in idxs]
            if not peaks_list:
                centers.append(np.array([]))
                continue
            max_len = max(len(p) for p in peaks_list)
            mat = np.full((len(peaks_list), max_len), np.nan)
            for row, p in enumerate(peaks_list):
                mat[row, :len(p)] = p
            center = np.nanmedian(mat, axis=0)
            centers.append(center)
        self.cluster_centers = centers

    def _merge_peaks_in_centers(self):
        if not self.cluster_centers:
            return
        tol = self.merge_peaks_tolerance
        merged_centers = []
        for center in self.cluster_centers:
            if len(center) == 0:
                merged_centers.append(center)
                continue
            sorted_peaks = np.sort(center)
            merged = []
            current_group = [sorted_peaks[0]]
            for p in sorted_peaks[1:]:
                if p - current_group[-1] <= tol:
                    current_group.append(p)
                else:
                    merged.append(np.mean(current_group))
                    current_group = [p]
            merged.append(np.mean(current_group))
            merged_centers.append(np.array(merged))
        self.cluster_centers = merged_centers
        print("\n[Cluster center peak merging] Completed, merge tolerance =", tol)
        unique = np.unique(self.cluster_labels)
        unique = unique[unique >= 0]
        for i, lab in enumerate(unique):
            print(f"  Cluster {lab}: number of peaks after merging = {len(self.cluster_centers[i])}")

    def export_cluster_events_by_ratio(self, output_dir='results_unified'):
        """
        Export event lists for each cluster in each ratio to CSV.
        """
        Path(output_dir).mkdir(exist_ok=True, parents=True)

        ratio_cluster_events = defaultdict(lambda: defaultdict(list))
        for idx, (label, ratio, event_id) in enumerate(zip(
            self.cluster_labels, self.all_ratios, self.event_ids
        )):
            if label >= 0:
                ratio_cluster_events[ratio][int(label)].append(event_id)

        rows = []
        for ratio in sorted(ratio_cluster_events.keys()):
            for cl in sorted(ratio_cluster_events[ratio].keys()):
                events = ratio_cluster_events[ratio][cl]
                rows.append({
                    'Ratio': ratio,
                    'Cluster': cl,
                    'Event_Count': len(events),
                    'Event_IDs': '; '.join(events)
                })

        df = pd.DataFrame(rows)
        output_file = Path(output_dir) / 'cluster_events_by_ratio.csv'
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Saved cluster event lists by ratio: {output_file}")

        for ratio, clusters in ratio_cluster_events.items():
            rows_ratio = []
            for cl, events in clusters.items():
                for ev in events:
                    rows_ratio.append({
                        'Cluster': cl,
                        'Event_ID': ev
                    })
            df_ratio = pd.DataFrame(rows_ratio)
            output_file_ratio = Path(output_dir) / f'cluster_events_{ratio.replace(":", "_")}.csv'
            df_ratio.to_csv(output_file_ratio, index=False, encoding='utf-8-sig')

        print(f"Saved independent event lists for each ratio: {output_dir}/cluster_events_*.csv")

        return df, dict(ratio_cluster_events)

    def get_cluster_events_by_ratio(self):
        """Get event lists for each cluster in each ratio."""
        result = defaultdict(lambda: defaultdict(list))
        for idx, (label, ratio, event_id) in enumerate(zip(
            self.cluster_labels, self.all_ratios, self.event_ids
        )):
            if label >= 0:
                result[ratio][int(label)].append(event_id)
        return {r: dict(clusters) for r, clusters in result.items()}

    def get_cluster_size_by_ratio(self):
        df = pd.DataFrame({'ratio': self.all_ratios, 'cluster': self.cluster_labels})
        df = df[df['cluster'] >= 0]
        pivot = pd.crosstab(df['ratio'], df['cluster'])
        desired_order = ['1:0', '1:1', '1:2', '1:4', '1:10']
        pivot = pivot.reindex([r for r in desired_order if r in pivot.index], fill_value=0)
        return pivot

    def get_clusters_by_ratio(self):
        ratio_clusters = {}
        ratios = list(dict.fromkeys(self.all_ratios))
        for ratio in ratios:
            ratio_clusters[ratio] = {}

        unique_clusters = np.unique(self.cluster_labels)
        unique_clusters = unique_clusters[unique_clusters >= 0]

        for i, cl in enumerate(unique_clusters):
            idxs = np.where(self.cluster_labels == cl)[0]
            center_peaks = self.cluster_centers[i] if i < len(self.cluster_centers) else np.array([])

            for ratio in ratios:
                ratio_idxs = [idx for idx in idxs if self.all_ratios[idx] == ratio]
                if ratio_idxs:
                    ratio_clusters[ratio][cl] = {
                        'center_peaks': center_peaks,
                        'count': len(ratio_idxs)
                    }
        return ratio_clusters

    def plot_cluster_distribution(self, output_dir='results_unified', fmt='png'):
        pivot = self.get_cluster_size_by_ratio()
        if pivot.empty:
            print("Warning: no valid clusters available for plotting")
            return None
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        Path(output_dir).mkdir(exist_ok=True)
        ax = pivot_pct.plot(kind='bar', stacked=True, figsize=(10,6), colormap='tab10', edgecolor='black')
        ax.set_xlabel('Biotin : Avidin molar ratio')
        ax.set_ylabel('Percentage of events (%)')
        ax.set_title('Cluster distribution (global clustering)')
        ax.legend(title='Cluster ID', bbox_to_anchor=(1.05,1), loc='upper left')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f'unified_cluster_distribution.{fmt}', dpi=300)
        plt.show()
        return pivot_pct

    def plot_cluster_rugplots(self, output_dir='results_unified',
                              clusters_to_plot=None, fmt='png'):
        Path(output_dir).mkdir(exist_ok=True)
        unique_clusters = np.unique(self.cluster_labels)
        unique_clusters = unique_clusters[unique_clusters >= 0]
        if clusters_to_plot is None:
            clusters_to_plot = unique_clusters
        else:
            clusters_to_plot = [c for c in clusters_to_plot if c in unique_clusters]

        for cl in clusters_to_plot:
            idxs = np.where(self.cluster_labels == cl)[0]
            if len(idxs) == 0:
                print(f"Cluster {cl} has no events; skipping plot")
                continue
            events_peaks = [self.all_peaks[i] for i in idxs]
            center_idx = list(unique_clusters).index(cl)
            center_peaks = self.cluster_centers[center_idx] if center_idx < len(self.cluster_centers) else np.array([])

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6),
                                           gridspec_kw={'height_ratios': [3, 1]})
            for i, peaks in enumerate(events_peaks):
                ax1.scatter(peaks, [i] * len(peaks), s=20, c='blue', alpha=0.5, linewidth=0)
            ax1.set_ylabel('Event index (sorted)', fontsize=12)
            ax1.set_xlim(self.peak_min, self.peak_max)
            ax1.set_title(f'Cluster {cl} (n={len(idxs)} events) - all events', fontsize=14)
            ax1.grid(True, linestyle=':', alpha=0.6)
            ax1.tick_params(axis='both', labelsize=10)

            ax2.stem(center_peaks, [0.3]*len(center_peaks), linefmt='r-', markerfmt='ro', basefmt=' ')
            ax2.set_ylim(0, 0.6)
            ax2.set_xlim(self.peak_min, self.peak_max)
            ax2.set_yticks([])
            ax2.set_xlabel('I/I0', fontsize=12)
            ax2.set_title('Center peaks (merged)', fontsize=12)
            ax2.tick_params(axis='x', labelsize=10)

            plt.tight_layout()
            plt.savefig(Path(output_dir) / f'cluster_{cl}_rugplot.{fmt}', dpi=300)
            plt.close()
            print(f"Saved full-point distribution plot for cluster {cl}: cluster_{cl}_rugplot.{fmt}")

    def export_assignments(self, output_file='unified_cluster_assignments.csv'):
        df_out = pd.DataFrame({
            'Event_ID': self.event_ids,
            'Ratio': self.all_ratios,
            'Final_Cluster': self.cluster_labels,
            'Original_Cluster': self.raw_cluster_labels,
            'Peaks': [list(p) for p in self.all_peaks]
        })
        df_out.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Cluster assignments saved to {output_file}")
        return df_out

    def export_results_to_csv(self, output_dir='results_unified'):
        Path(output_dir).mkdir(exist_ok=True)
        unique = np.unique(self.cluster_labels)
        unique = unique[unique >= 0]
        centers_df = pd.DataFrame({
            'Cluster': unique,
            'Center_Peaks': [str(np.round(c, 4).tolist()) for c in self.cluster_centers],
            'Event_Count': [np.sum(self.cluster_labels == lab) for lab in unique]
        })
        centers_df.to_csv(Path(output_dir) / 'cluster_centers.csv', index=False, encoding='utf-8-sig')
        counts_pivot = self.get_cluster_size_by_ratio()
        if not counts_pivot.empty:
            counts_pivot.to_csv(Path(output_dir) / 'cluster_counts.csv', encoding='utf-8-sig')
            pct_pivot = counts_pivot.div(counts_pivot.sum(axis=1), axis=0) * 100
            pct_pivot.to_csv(Path(output_dir) / 'cluster_percentages.csv', encoding='utf-8-sig')
        removal_stats = {
            'Total_Events': len(self.all_peaks),
            'Removed_Events': int(np.sum(self.cluster_labels == -1)),
            'Remaining_Events': int(np.sum(self.cluster_labels >= 0)),
            'Min_Cluster_Size_Threshold': self.min_cluster_size,
            'Min_Per_Ratio_Ratio_Threshold': self.min_per_ratio_ratio,
            'Merge_Peaks_Tolerance': self.merge_peaks_tolerance
        }
        pd.Series(removal_stats).to_csv(Path(output_dir) / 'removal_stats.csv', header=True)
        print(f"Results saved as multiple CSV files in {output_dir}")

    def print_cluster_centers(self):
        print("\nRepresentative peak positions for each valid cluster (after merging):")
        unique = np.unique(self.cluster_labels)
        unique = unique[unique >= 0]
        for i, lab in enumerate(unique):
            if i < len(self.cluster_centers):
                center = self.cluster_centers[i]
            else:
                center = np.array([])
            n_events = np.sum(self.cluster_labels == lab)
            print(f"Cluster {lab} (n={n_events}): peak positions = {np.round(center, 4)}")


if __name__ == "__main__":
    base_dir = r"H:\数据\2026年\Supplementary experiment\Supplementary experiment nanopore\0416-12nm-Cu-Fe-avidin-biotin-good-数据分析\重新改分析方法\提取峰值后"

    ratio_files_dict = {
        "1:0": [
            f"{base_dir}\\1-0-100-1Events-71-all_titration_peaks.csv",
            f"{base_dir}\\1-0-26416003Events-8.2min-106-event-38-85-all_titration_peaks.csv",
            f"{base_dir}\\1-0-26416003Events-9.2min-101-all_titration_peaks.csv"
        ],
        "1:1": [
            f"{base_dir}\\1-1-26416004Events0-4min-48-all_titration_peaks.csv",
            f"{base_dir}\\1-1-26416004Events-10min-44-all_titration_peaks.csv"
        ],
        "1:2": [
            f"{base_dir}\\1-2-26416006Events-87s-50-all_titration_peaks.csv"
        ],
        "1:4": [
            f"{base_dir}\\1-4-26416009Events-5-7.5min-96-events-50-96-all_titration_peaks.csv",
            f"{base_dir}\\1-4-26416009Events-11.5-221-event-45-144-all_titration_peaks.csv"
        ],
        "1:10": [
            f"{base_dir}\\1-10-26416010Events-6.5-9.5min-event1-72-all_titration_peaks.csv"
        ],
    }

    TOLERANCE = 0.003
    N_CLUSTERS = 8
    MIN_PEAKS = 3
    PEAK_MIN = 0.75
    PEAK_MAX = 1.05
    EXCLUDE_BASELINE = True
    PEAK_COUNT_PENALTY_WEIGHT = 0.8
    MIN_CLUSTER_SIZE = 10
    MIN_PER_RATIO_RATIO = 0.05
    MERGE_PEAKS_TOLERANCE = 0.005

    OUTPUT_DIR = 'results_unified'
    OUTPUT_FORMAT = 'tiff'

    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

    analyzer = UnifiedClusterAnalysis(
        tolerance=TOLERANCE,
        n_clusters=N_CLUSTERS,
        min_peaks=MIN_PEAKS,
        peak_min=PEAK_MIN,
        peak_max=PEAK_MAX,
        exclude_baseline=EXCLUDE_BASELINE,
        peak_count_penalty_weight=PEAK_COUNT_PENALTY_WEIGHT,
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_per_ratio_ratio=MIN_PER_RATIO_RATIO,
        merge_peaks_tolerance=MERGE_PEAKS_TOLERANCE
    )

    n_events = analyzer.load_all_ratios_from_files(ratio_files_dict)
    if n_events < 2:
        print("Not enough valid events for clustering")
        exit()

    print("\n" + "="*60)
    print("Running global clustering analysis...")
    print("="*60)
    analyzer.perform_clustering()

    analyzer.print_cluster_centers()

    analyzer.plot_cluster_distribution(output_dir=OUTPUT_DIR, fmt=OUTPUT_FORMAT)

    analyzer.export_assignments(output_file=Path(OUTPUT_DIR) / 'unified_cluster_assignments.csv')
    analyzer.export_results_to_csv(output_dir=OUTPUT_DIR)

    print("\n" + "="*60)
    print("Exporting cluster event lists by ratio...")
    print("="*60)
    df_events, cluster_events_dict = analyzer.export_cluster_events_by_ratio(output_dir=OUTPUT_DIR)

    print("\nCluster distribution summary by ratio:")
    print("-" * 60)
    for ratio, clusters in cluster_events_dict.items():
        print(f"\n{ratio}:")
        for cl, events in sorted(clusters.items()):
            print(f"  Cluster {cl}: {len(events)} events")
            event_preview = events[:5]
            print(f"    Example event IDs: {event_preview}" + (" ..." if len(events) > 5 else ""))

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    print(f"Results saved in: {Path(OUTPUT_DIR).absolute()}")
    print("\nKey output files (for downstream full-point distribution plotting):")
    print(f"  - cluster_events_by_ratio.csv : event lists for each ratio and cluster (summary)")
    print(f"  - cluster_events_*.csv : independent event lists for each ratio")
    print(f"  - unified_cluster_assignments.csv : detailed cluster assignments for all events")
    print("\nUsage:")
    print("  1. Open cluster_events_by_ratio.csv")
    print("  2. Locate the target ratio and target cluster")
    print("  3. Copy the event IDs in the Event_IDs column")
    print("  4. Extract these events from the original full-point distribution data table for plotting")
