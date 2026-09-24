"""
Insulin post-translational modification (PTM) nanopore data analysis:
cross-ratio global unified clustering analysis.

- Global clustering: all ratios are pooled, so cluster IDs are comparable across ratios.
- Match-score filtering: events farther than a threshold from the cluster center can be removed.
- Export event lists for each cluster in each ratio to CSV.
- Identify newly appearing / significantly increased clusters compared with the previous ratio.
- Output event lists required for full-point distribution plots.
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
    def __init__(self, tolerance=0.008, n_clusters=8, min_peaks=2,
                 peak_min=0.55, peak_max=1.00, exclude_baseline=None,
                 peak_count_penalty_weight=0.8,
                 min_cluster_size=5, min_per_ratio_ratio=0.05,
                 merge_peaks_tolerance=None,
                 match_threshold=None,
                 hist_threshold=1):
        """
        Global clustering analyzer.

        Parameters
        ----------
        tolerance : float
            Peak matching tolerance.
        n_clusters : int
            Target number of clusters.
        min_peaks : int
            Minimum number of peaks.
        peak_min : float
            Minimum peak position.
        peak_max : float
            Maximum peak position.
        exclude_baseline : tuple
            Baseline region to exclude, e.g. (low, high).
        peak_count_penalty_weight : float
            Penalty weight for peak-count differences.
        min_cluster_size : int
            Minimum cluster size.
        min_per_ratio_ratio : float
            Minimum fraction within a ratio.
        merge_peaks_tolerance : float
            Peak merging tolerance.
        match_threshold : float
            Match-score threshold.
        hist_threshold : int
            Histogram count threshold; only bins with Histo >= this value are considered peaks.
        """
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
        self.match_threshold = match_threshold
        self.hist_threshold = hist_threshold

        self.all_peaks = []
        self.all_ratios = []
        self.event_ids = []
        self.file_names = []
        self.raw_cluster_labels = None
        self.cluster_labels = None
        self.cluster_centers = []
        self.distance_matrix = None

    def load_all_ratios_from_files(self, ratio_files_dict):
        """
        Load data files for all ratios.
        Supports:
        1. Standard format with a Peak_Positions column.
        2. Wide format with Bin Center and Histo(tN) columns.
        """
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

                cols = df.columns.tolist()

                if 'Bin Center' in cols and any('Histo(t' in col for col in cols):
                    print(f"  File {file_path.name}: detected wide format (histogram data)")
                    self._load_wide_format(df, ratio, file_path.name)
                elif 'Peak_Positions' in cols:
                    print(f"  File {file_path.name}: detected standard format (Peak_Positions)")
                    self._load_standard_format(df, ratio, file_path.name)
                elif 'Peaks_str' in cols:
                    print(f"  File {file_path.name}: detected standard format (Peaks_str)")
                    self._load_standard_format(df, ratio, file_path.name, peaks_col='Peaks_str')
                else:
                    print(f"Warning: {file_path.name} format not recognized; available columns: {cols[:10]}...")
                    continue

        print(f"\nLoaded {len(self.all_peaks)} events in total")
        print(f"Event counts by ratio: {Counter(self.all_ratios)}")
        return len(self.all_peaks)

    def _load_wide_format(self, df, ratio, file_name):
        """
        Load wide-format data (Bin Center + Histo(tN)).
        Each Histo(tN) column is one event histogram.
        Bin Center gives the corresponding bin center values.
        """
        histo_cols = [col for col in df.columns if 'Histo(t' in col]

        if not histo_cols:
            print(f"  Warning: {file_name} has no Histo columns")
            return

        if 'Bin Center' not in df.columns:
            print(f"  Warning: {file_name} has no 'Bin Center' column")
            return

        bin_centers = df['Bin Center'].values

        for idx, histo_col in enumerate(histo_cols):
            histo_values = df[histo_col].values

            peaks = bin_centers[histo_values >= self.hist_threshold]

            peaks = peaks[peaks >= self.peak_min]
            peaks = peaks[peaks <= self.peak_max]

            if self.exclude_baseline is not None:
                low, high = self.exclude_baseline
                peaks = peaks[~((peaks >= low) & (peaks <= high))]

            if len(peaks) < self.min_peaks:
                continue

            event_id = f"{ratio}_{file_name}_{idx}"

            self.all_peaks.append(np.sort(peaks))
            self.all_ratios.append(ratio)
            self.event_ids.append(event_id)
            self.file_names.append(file_name)

    def _load_standard_format(self, df, ratio, file_name, peaks_col='Peak_Positions'):
        """Load standard-format data containing peak positions."""
        if 'Event' not in df.columns:
            df['Event'] = [f"{ratio}_{i}" for i in range(len(df))]

        for _, row in df.iterrows():
            event_id = str(row['Event'])
            peak_str = str(row[peaks_col]).strip('[]')
            if not peak_str or peak_str == 'nan':
                continue
            try:
                peaks = np.array([float(x.strip()) for x in peak_str.split(',')])
            except:
                continue

            peaks = peaks[peaks >= self.peak_min]
            peaks = peaks[peaks <= self.peak_max]

            if self.exclude_baseline is not None:
                low, high = self.exclude_baseline
                peaks = peaks[~((peaks >= low) & (peaks <= high))]

            if len(peaks) < self.min_peaks:
                continue

            self.all_peaks.append(np.sort(peaks))
            self.all_ratios.append(ratio)
            self.event_ids.append(event_id)
            self.file_names.append(file_name)

    def _peaks_distance(self, p1, p2):
        """Calculate distance between two peak sets."""
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
        """Compute distance matrix."""
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
        self.distance_matrix = dist
        return dist

    def _compute_event_to_cluster_distance(self, event_idx, cluster_idxs):
        """Calculate average distance from one event to all events in a cluster."""
        if len(cluster_idxs) == 0:
            return 1.0
        distances = [self.distance_matrix[event_idx, j] for j in cluster_idxs if j != event_idx]
        if len(distances) == 0:
            return 0.0
        return np.mean(distances)

    def _filter_low_match_events(self):
        """Remove events with low match scores to their assigned clusters."""
        if self.cluster_labels is None or self.distance_matrix is None:
            print("Warning: no clustering result or distance matrix; skipping match filtering")
            return

        if self.match_threshold is None:
            print("Match threshold not set; skipping match filtering")
            return

        current_labels = self.cluster_labels.copy()
        unique_clusters = np.unique(current_labels)
        unique_clusters = unique_clusters[unique_clusters >= 0]

        if len(unique_clusters) == 0:
            print("No valid clusters; skipping match filtering")
            return

        cluster_idxs_dict = {}
        for cl in unique_clusters:
            cluster_idxs_dict[cl] = np.where(current_labels == cl)[0].tolist()

        mask_to_remove = np.zeros(len(self.all_peaks), dtype=bool)
        removed_events = []

        for cl in unique_clusters:
            idxs = cluster_idxs_dict[cl]
            if len(idxs) < 3:
                mask_to_remove[idxs] = True
                for idx in idxs:
                    removed_events.append((idx, cl, "cluster too small"))
                continue

            for idx in idxs:
                avg_dist = self._compute_event_to_cluster_distance(idx, idxs)
                if avg_dist > self.match_threshold:
                    mask_to_remove[idx] = True
                    removed_events.append((idx, cl, avg_dist))

        new_labels = current_labels.copy()
        new_labels[mask_to_remove] = -1

        unique_new_valid = sorted([cl for cl in np.unique(new_labels) if cl != -1])
        label_mapping = {old: new for new, old in enumerate(unique_new_valid)}
        self.cluster_labels = np.array([label_mapping.get(lbl, -1) for lbl in new_labels])

        removed_count = np.sum(mask_to_remove)
        print(f"\n[Match filtering] Removed {removed_count} low-match events")
        print(f"  Match threshold: {self.match_threshold}")
        if removed_events:
            print(f"  Removal details (first 10):")
            for idx, cl, dist in removed_events[:10]:
                if isinstance(dist, (int, float)):
                    print(f"    Event {self.event_ids[idx]} originally in cluster {cl}, average distance {dist:.4f}")
                else:
                    print(f"    Event {self.event_ids[idx]} originally in cluster {cl}, reason: {dist}")
            if len(removed_events) > 10:
                print(f"    ... {len(removed_events) - 10} more events removed")
        print(f"  Remaining valid clusters: {len(unique_new_valid)}")
        print(f"  Remaining events: {np.sum(self.cluster_labels >= 0)}")

    def perform_clustering(self):
        """Run global clustering."""
        dist_matrix = self.compute_distance_matrix()
        print(f"Using hierarchical clustering, target number of clusters = {self.n_clusters}")
        clustering = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric='precomputed',
            linkage='average'
        )
        self.raw_cluster_labels = clustering.fit_predict(dist_matrix)
        self.cluster_labels = self.raw_cluster_labels.copy()

        self._filter_low_match_events()
        self._filter_small_clusters()
        self._filter_low_ratio_per_ratio()
        self._compute_cluster_centers()
        self._merge_peaks_in_centers()

        return self.cluster_labels

    def _filter_small_clusters(self):
        """Remove clusters with too few total events."""
        if self.raw_cluster_labels is None:
            raise ValueError("Please run perform_clustering() first")
        labels = self.cluster_labels.copy()
        cluster_counts = Counter(labels)
        small_clusters = {cl for cl, cnt in cluster_counts.items() if cnt < self.min_cluster_size and cl != -1}

        if small_clusters:
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
        else:
            print(f"\n[Global filtering] All clusters have size >= {self.min_cluster_size}; no removal needed")

    def _filter_low_ratio_per_ratio(self):
        """Remove clusters with too-low fractions in individual ratios."""
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
        if removed_second > 0:
            print(f"\n[Ratio-level filtering] Removed {removed_second} events")
            print(f"  Remaining valid clusters: {len(unique_new_valid)}")

    def _compute_cluster_centers(self):
        """Compute cluster centers."""
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
        """Merge nearby peaks in cluster centers."""
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
            output_file_ratio = Path(output_dir) / f'cluster_events_{ratio}.csv'
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

    def identify_new_clusters(self, ratio_order=None):
        """
        Identify newly appearing or significantly increased clusters compared with the previous ratio.

        Parameters
        ----------
        ratio_order : list
            Ratio order, e.g. ['0eq', '1eq', '6eq'].

        Returns
        -------
        dict
            {ratio: {'new': [cluster_ids], 'increased': [cluster_ids], ...}}
        """
        if ratio_order is None:
            ratio_order = sorted(self.all_ratios)

        cluster_events_dict = self.get_cluster_events_by_ratio()

        result = {}
        prev_clusters = set()

        for i, ratio in enumerate(ratio_order):
            if ratio not in cluster_events_dict:
                continue

            current_clusters = set(cluster_events_dict[ratio].keys())
            current_counts = {cl: len(events) for cl, events in cluster_events_dict[ratio].items()}
            current_total = sum(current_counts.values())

            new_clusters = []
            increased_clusters = []

            if i == 0:
                new_clusters = list(current_clusters)
            else:
                new_clusters = list(current_clusters - prev_clusters)

                if ratio_order[i-1] in cluster_events_dict:
                    prev_counts = {cl: len(events) for cl, events in cluster_events_dict[ratio_order[i-1]].items()}
                    prev_total = sum(prev_counts.values())

                    for cl in current_clusters & prev_clusters:
                        current_pct = current_counts[cl] / current_total * 100
                        prev_pct = prev_counts[cl] / prev_total * 100 if prev_total > 0 else 0
                        if current_pct > prev_pct * 1.5 and (current_pct - prev_pct) > 5:
                            increased_clusters.append(cl)

            result[ratio] = {
                'new': new_clusters,
                'increased': increased_clusters,
                'all': list(current_clusters),
                'counts': current_counts,
                'total': current_total
            }

            prev_clusters = current_clusters

            print(f"\n{ratio}:")
            print(f"  Total events: {current_total}")
            print(f"  All clusters: {sorted(current_clusters)}")
            if new_clusters:
                print(f"  Newly appearing clusters: {sorted(new_clusters)}")
            if increased_clusters:
                print(f"  Clusters with increased proportion: {sorted(increased_clusters)}")

        return result

    def export_full_point_plot_events(self, output_dir='results_unified', ratio_order=None):
        """
        Export event lists required for full-point distribution plots.
        For each ratio, find events in newly appearing / significantly increased clusters.
        """
        Path(output_dir).mkdir(exist_ok=True, parents=True)

        if ratio_order is None:
            ratio_order = sorted(self.all_ratios)

        new_cluster_info = self.identify_new_clusters(ratio_order)
        cluster_events_dict = self.get_cluster_events_by_ratio()

        result = {}

        for ratio in ratio_order:
            if ratio not in new_cluster_info or ratio not in cluster_events_dict:
                continue

            result[ratio] = {
                'new': [],
                'increased': [],
                'all': []
            }

            for cl in new_cluster_info[ratio]['new']:
                if cl in cluster_events_dict[ratio]:
                    result[ratio]['new'].extend(cluster_events_dict[ratio][cl])

            for cl in new_cluster_info[ratio]['increased']:
                if cl in cluster_events_dict[ratio]:
                    result[ratio]['increased'].extend(cluster_events_dict[ratio][cl])

            for cl, events in cluster_events_dict[ratio].items():
                result[ratio]['all'].extend(events)

            rows = []
            for ev in result[ratio]['new']:
                rows.append({'Ratio': ratio, 'Event_ID': ev, 'Category': 'new'})
            for ev in result[ratio]['increased']:
                rows.append({'Ratio': ratio, 'Event_ID': ev, 'Category': 'increased'})

            new_or_increased = set(result[ratio]['new']) | set(result[ratio]['increased'])
            for ev in result[ratio]['all']:
                if ev not in new_or_increased:
                    rows.append({'Ratio': ratio, 'Event_ID': ev, 'Category': 'other'})

            df = pd.DataFrame(rows)
            output_file = Path(output_dir) / f'full_point_plot_events_{ratio}.csv'
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\nSaved full-point plot event list for {ratio}: {output_file}")

            print(f"  New-cluster events: {len(result[ratio]['new'])}")
            print(f"  Increased-proportion cluster events: {len(result[ratio]['increased'])}")
            print(f"  Total events: {len(result[ratio]['all'])}")

        return result

    def get_cluster_size_by_ratio(self):
        """Get counts for each cluster in each ratio."""
        df = pd.DataFrame({'ratio': self.all_ratios, 'cluster': self.cluster_labels})
        df = df[df['cluster'] >= 0]
        pivot = pd.crosstab(df['ratio'], df['cluster'])
        return pivot

    def plot_cluster_distribution(self, output_dir='results_unified', fmt='png'):
        """Plot stacked bar chart of cluster distributions."""
        pivot = self.get_cluster_size_by_ratio()
        if pivot.empty:
            print("Warning: no valid clusters available for plotting")
            return None

        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        Path(output_dir).mkdir(exist_ok=True)

        n_clusters = len(pivot_pct.columns)
        colors = [CLUSTER_COLORS.get(i, '#888888') for i in range(n_clusters)]

        ax = pivot_pct.plot(kind='bar', stacked=True, figsize=(10,6),
                            color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('PTM equivalent (fatty acid eq)', fontsize=12)
        ax.set_ylabel('Percentage of events (%)', fontsize=12)
        ax.set_title('Cluster distribution across PTM levels (global clustering)', fontsize=14)
        ax.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 105)

        plt.tight_layout()
        plt.savefig(Path(output_dir) / f'cluster_distribution.{fmt}', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved cluster distribution plot: {output_dir}/cluster_distribution.{fmt}")
        return pivot_pct

    def plot_cluster_rugplots(self, output_dir='results_unified',
                              clusters_to_plot=None, fmt='png'):
        """Plot rug plots (full-point distribution plots) for selected clusters."""
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

            if len(center_peaks) > 0:
                ax2.stem(center_peaks, [0.3]*len(center_peaks), linefmt='k-', markerfmt='ro', basefmt=' ')
                ax2.set_ylim(0, 0.6)
            else:
                ax2.set_ylim(0, 0.6)
            ax2.set_xlim(self.peak_min, self.peak_max)
            ax2.set_yticks([])
            ax2.set_xlabel('I/I0', fontsize=12)
            ax2.set_title('Center peaks (merged)', fontsize=12)
            ax2.tick_params(axis='x', labelsize=10)

            plt.tight_layout()
            plt.savefig(Path(output_dir) / f'cluster_{cl}_rugplot.{fmt}', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved rug plot for cluster {cl}: cluster_{cl}_rugplot.{fmt}")

    def export_assignments(self, output_file='cluster_assignments.csv'):
        """Export cluster assignments for all events."""
        df_out = pd.DataFrame({
            'Event_ID': self.event_ids,
            'Ratio': self.all_ratios,
            'Final_Cluster': self.cluster_labels,
            'Original_Cluster': self.raw_cluster_labels,
            'File_Name': self.file_names,
            'Peaks': [list(p) for p in self.all_peaks]
        })
        df_out.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Cluster assignments saved to {output_file}")
        return df_out

    def export_results_to_csv(self, output_dir='results_unified'):
        """Export all results to CSV."""
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
            'Merge_Peaks_Tolerance': self.merge_peaks_tolerance,
            'Match_Threshold': self.match_threshold if self.match_threshold is not None else 'Not used',
            'Hist_Threshold': self.hist_threshold
        }
        pd.Series(removal_stats).to_csv(Path(output_dir) / 'removal_stats.csv', header=True)

        print(f"Results saved as multiple CSV files in {output_dir}")

    def print_cluster_centers(self):
        """Print cluster centers."""
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
    DATA_DIR = r"H:\数据\2026年\Supplementary experiment\Supplementary experiment nanopore\0516-12nm-Cu-Fe-insulin-PTM-good\重新数据处理"

    ratio_files_dict = {
        "0eq": [
            f"{DATA_DIR}\\peak_cards_fwhm_summary-1-0-1.csv",
        ],
        "1eq": [
            f"{DATA_DIR}\\peak_cards_fwhm_summary-1-1-1.csv",
        ],
        "6eq": [
            f"{DATA_DIR}\\peak_cards_fwhm_summary-1-6-2.csv",
        ],
    }

    TOLERANCE = 0.008
    N_CLUSTERS = 6
    MIN_PEAKS = 2
    PEAK_MIN = 0.55
    PEAK_MAX = 1.00
    EXCLUDE_BASELINE = (0.95, 1.02)
    PEAK_COUNT_PENALTY_WEIGHT = 0.8
    MIN_CLUSTER_SIZE = 5
    MIN_PER_RATIO_RATIO = 0.05
    MERGE_PEAKS_TOLERANCE = 0.008
    MATCH_THRESHOLD = None
    HIST_THRESHOLD = 1

    RATIO_ORDER = ['0eq', '1eq', '6eq']

    OUTPUT_DIR = 'results_ptm_global_0_1_6eq'
    OUTPUT_FORMAT = 'png'

    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
    print(f"\nResults will be saved to: {Path(OUTPUT_DIR).resolve()}")

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
        merge_peaks_tolerance=MERGE_PEAKS_TOLERANCE,
        match_threshold=MATCH_THRESHOLD,
        hist_threshold=HIST_THRESHOLD
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

    analyzer.export_assignments(output_file=Path(OUTPUT_DIR) / 'cluster_assignments.csv')
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
    print("Identifying newly appearing / significantly increased clusters...")
    print("="*60)
    new_cluster_info = analyzer.identify_new_clusters(ratio_order=RATIO_ORDER)

    print("\n" + "="*60)
    print("Exporting full-point plot event lists...")
    print("="*60)
    full_point_events = analyzer.export_full_point_plot_events(
        output_dir=OUTPUT_DIR,
        ratio_order=RATIO_ORDER
    )

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    print(f"Results saved in: {Path(OUTPUT_DIR).absolute()}")

    print("\nKey output files:")
    print(f"  - cluster_events_by_ratio.csv : event lists for each ratio and cluster (summary)")
    print(f"  - cluster_events_*.csv : independent event lists for each ratio")
    print(f"  - full_point_plot_events_*.csv : full-point plot event lists by category")
    print(f"  - cluster_assignments.csv : detailed cluster assignments for all events")
    print(f"  - cluster_distribution.{OUTPUT_FORMAT} : cluster distribution plot")

    print("\nUsage:")
    print("  1. Open full_point_plot_events_*.csv")
    print("  2. Select events with Category 'new' or 'increased'")
    print("  3. Extract these events from the original full-point distribution data table for plotting")
    print("="*60)
