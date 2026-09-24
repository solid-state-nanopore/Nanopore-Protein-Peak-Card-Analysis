"""
Binary classifier based on standard peak cards with error estimation.
Uses known IgG and apo-Tf standard peak cards to classify unknown events.
Includes similarity threshold, confidence evaluation, and independent error estimation.
"""

import numpy as np
import pandas as pd
from tabulate import tabulate
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class PeakCardClassifier:
    """
    Peak-card-based classifier with error estimation.
    Includes similarity threshold, unclassified handling, and independent error estimation.
    """

    def __init__(self,
                 tolerance: float = 0.003,
                 min_peaks_per_card: int = 2,
                 min_similarity_threshold: float = 0.3,
                 confidence_level: float = 0.95):
        """
        Initialize the classifier.

        Parameters
        ----------
        tolerance : float
            Peak matching tolerance.
        min_peaks_per_card : int
            Minimum number of peaks per card.
        min_similarity_threshold : float
            Minimum similarity threshold. Below this value, the result is unclassified.
        confidence_level : float
            Confidence level for error estimation (default: 0.95).
        """
        self.tolerance = tolerance
        self.min_peaks_per_card = min_peaks_per_card
        self.min_similarity_threshold = min_similarity_threshold
        self.confidence_level = confidence_level
        self.cards = {}
        self.card_ids = []
        self.standard_cards = {}
        self.classification_results = []

    def load_cards_from_csv(self, csv_file: str):
        """
        Load cards to be classified from a CSV file.
        """
        loaded = 0

        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_file, encoding='gbk')

        required_cols = ['Event', 'Peak_Positions']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Input CSV is missing required columns: {missing_cols}. Current columns: {list(df.columns)}"
            )

        df.columns = [str(col).replace('\ufeff', '').strip() for col in df.columns]

        required_columns = ['Event', 'Peak_Positions']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Input CSV is missing required columns: {missing_columns}. Current columns: {list(df.columns)}")

        for _, row in df.iterrows():
            card_id = row['Event']
            peak_str = row['Peak_Positions'].strip('[]')
            if peak_str:
                peaks = np.array([float(x.strip()) for x in peak_str.split(',')])

                peaks = peaks[~((peaks >= 0.98) & (peaks <= 1.02))]

                if len(peaks) >= self.min_peaks_per_card:
                    self.cards[card_id] = np.sort(peaks)
                    loaded += 1

        self.card_ids = list(self.cards.keys())
        print(f"Loaded {loaded} cards to classify")
        return loaded

    def set_standard_cards(self, standard_cards: Dict[str, np.ndarray]):
        """
        Set standard cards.
        """
        self.standard_cards = {}
        for name, peaks in standard_cards.items():
            self.standard_cards[name] = np.sort(np.array(peaks))
            print(f"Standard card set: {name} -> {self.standard_cards[name]}")

    def calculate_similarity(self, peaks1: np.ndarray, peaks2: np.ndarray) -> float:
        """
        Calculate similarity between two cards (0-1).
        """
        if len(peaks1) == 0 or len(peaks2) == 0:
            return 0.0

        p1 = np.sort(peaks1)
        p2 = np.sort(peaks2)

        matches = []
        used_in_p2 = set()

        for val1 in p1:
            for i, val2 in enumerate(p2):
                if i in used_in_p2:
                    continue
                if abs(val1 - val2) <= self.tolerance:
                    matches.append((val1, val2))
                    used_in_p2.add(i)
                    break

        match_ratio = len(matches) / min(len(p1), len(p2)) if min(len(p1), len(p2)) > 0 else 0

        if len(matches) == 0:
            return 0.0

        if len(matches) >= 2:
            idx1 = [np.where(np.abs(p1 - m[0]) <= self.tolerance)[0][0] for m in matches]
            idx2 = [np.where(np.abs(p2 - m[1]) <= self.tolerance)[0][0] for m in matches]
            idx1 = np.sort(idx1)
            idx2 = np.sort(idx2)

            if len(idx1) >= 2:
                gaps1 = np.diff(p1[idx1])
                gaps2 = np.diff(p2[idx2])
                gap_diff = np.mean(np.abs(gaps1 - gaps2)) / (self.tolerance * 10 + 0.001)
                position_score = 1.0 / (1.0 + gap_diff)
            else:
                position_score = 1.0
        else:
            position_score = 1.0

        final_score = 0.7 * match_ratio + 0.3 * position_score
        return final_score

    def classify_card(self, card_id: str) -> Dict:
        """
        Classify a single card with uncertainty estimation.
        """
        if card_id not in self.cards:
            return None

        peaks = self.cards[card_id]
        similarities = {}

        for std_name, std_peaks in self.standard_cards.items():
            sim = self.calculate_similarity(peaks, std_peaks)
            similarities[std_name] = sim

        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        best_class = sorted_sims[0][0]
        best_score = sorted_sims[0][1]

        if len(sorted_sims) > 1:
            second_score = sorted_sims[1][1]
            confidence = best_score - second_score
        else:
            confidence = best_score

        scores = np.array([s for _, s in similarities.items()])
        exp_scores = np.exp(scores * 5)
        prob_igg = exp_scores[0] / exp_scores.sum() if 'IgG' in similarities else 0
        prob_aptf = exp_scores[1] / exp_scores.sum() if 'apo-Tf' in similarities else 0

        if best_score < self.min_similarity_threshold:
            predicted_class = "Uncertain"
            certainty_prob = 0
        else:
            predicted_class = best_class
            certainty_prob = max(prob_igg, prob_aptf)

        return {
            'card_id': card_id,
            'peaks': peaks,
            'similarities': similarities,
            'predicted_class': predicted_class,
            'best_score': best_score,
            'confidence': confidence,
            'certainty_prob': certainty_prob,
            'prob_igg': prob_igg,
            'prob_aptf': prob_aptf
        }

    def classify_all(self) -> pd.DataFrame:
        """
        Classify all cards.
        """
        print("\nStarting classification...")
        results = []

        for i, card_id in enumerate(self.card_ids):
            result = self.classify_card(card_id)
            if result:
                results.append(result)

                if (i + 1) % 10 == 0 or i + 1 == len(self.card_ids):
                    print(f"  Processed {i+1}/{len(self.card_ids)} cards")

        self.classification_results = results

        df = pd.DataFrame([{
            'Event': r['card_id'],
            'Num_Peaks': len(r['peaks']),
            'Peak_Positions': str(r['peaks'].tolist()),
            'IgG_Similarity': round(r['similarities'].get('IgG', 0), 3),
            'apo-Tf_Similarity': round(r['similarities'].get('apo-Tf', 0), 3),
            'Predicted_Class': r['predicted_class'],
            'Best_Score': round(r['best_score'], 3),
            'Confidence': round(r['confidence'], 3),
            'Certainty_Prob': round(r['certainty_prob'], 3),
            'Prob_IgG': round(r['prob_igg'], 3),
            'Prob_apo_Tf': round(r['prob_aptf'], 3)
        } for r in results])

        return df

    def get_class_counts(self) -> Dict[str, int]:
        """
        Get event counts for each class.
        """
        counts = {name: 0 for name in self.standard_cards.keys()}
        counts['Uncertain'] = 0

        for r in self.classification_results:
            counts[r['predicted_class']] += 1

        return counts

    def get_class_counts_with_error(self) -> Dict[str, Dict]:
        """
        Get class counts and independent error ranges.
        Standard error based on Poisson distribution: SE = sqrt(n).
        """
        counts = self.get_class_counts()
        total_classified = counts.get('IgG', 0) + counts.get('apo-Tf', 0)

        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)

        igg_count = counts.get('IgG', 0)
        igg_poisson_se = np.sqrt(igg_count) if igg_count > 0 else 0
        igg_poisson_ci = (max(0, igg_count - z_score * igg_poisson_se),
                          igg_count + z_score * igg_poisson_se)
        igg_relative_error = (igg_poisson_se / igg_count * 100) if igg_count > 0 else 0
        igg_margin = z_score * igg_poisson_se

        aptf_count = counts.get('apo-Tf', 0)
        aptf_poisson_se = np.sqrt(aptf_count) if aptf_count > 0 else 0
        aptf_poisson_ci = (max(0, aptf_count - z_score * aptf_poisson_se),
                           aptf_count + z_score * aptf_poisson_se)
        aptf_relative_error = (aptf_poisson_se / aptf_count * 100) if aptf_count > 0 else 0
        aptf_margin = z_score * aptf_poisson_se

        return {
            'IgG': {
                'count': igg_count,
                'poisson_ci': igg_poisson_ci,
                'poisson_margin': igg_margin,
                'standard_error': igg_poisson_se,
                'relative_error': igg_relative_error
            },
            'apo-Tf': {
                'count': aptf_count,
                'poisson_ci': aptf_poisson_ci,
                'poisson_margin': aptf_margin,
                'standard_error': aptf_poisson_se,
                'relative_error': aptf_relative_error
            },
            'Uncertain': {
                'count': counts.get('Uncertain', 0)
            },
            'total_classified': total_classified,
            'total_all': sum(counts.values())
        }

    def print_summary(self):
        """
        Print classification summary with error estimation.
        """
        print("\n" + "="*80)
        print("Classification summary with independent error estimation")
        print("="*80)

        counts = self.get_class_counts()
        total = sum(counts.values())

        print(f"\nTotal events: {total}")
        print(f"\nEvent counts by class:")
        for class_name, count in counts.items():
            percentage = count / total * 100 if total > 0 else 0
            print(f"  {class_name}: {count} events ({percentage:.1f}%)")

        classified_total = total - counts.get('Uncertain', 0)
        if classified_total > 0:
            print(f"\nClassified events: {classified_total}/{total} ({classified_total/total*100:.1f}%)")

        confidences = [r['confidence'] for r in self.classification_results]
        print(f"\nConfidence statistics:")
        print(f"  Mean confidence: {np.mean(confidences):.3f}")
        print(f"  Min confidence: {np.min(confidences):.3f}")
        print(f"  Max confidence: {np.max(confidences):.3f}")

        low_confidence = [r for r in self.classification_results if r['confidence'] < 0.1 and r['predicted_class'] != 'Uncertain']
        if low_confidence:
            print(f"\nWARNING: Low-confidence events (confidence < 0.1): {len(low_confidence)}")
            for r in low_confidence[:5]:
                print(f"    {r['card_id']}: IgG={r['similarities']['IgG']:.3f}, "
                      f"apo-Tf={r['similarities']['apo-Tf']:.3f} -> {r['predicted_class']}")
            if len(low_confidence) > 5:
                print(f"    ... {len(low_confidence)-5} more")

        stats_with_error = self.get_class_counts_with_error()
        print(f"\n" + "-"*40)
        print("Independent error estimation (95% confidence interval)")
        print("-"*40)

        print(f"\n  IgG:")
        print(f"    Count: {stats_with_error['IgG']['count']}")
        print(f"    Standard error: ±{stats_with_error['IgG']['standard_error']:.1f}")
        print(f"    Relative error: ±{stats_with_error['IgG']['relative_error']:.1f}%")
        print(f"    95% CI: [{stats_with_error['IgG']['poisson_ci'][0]:.1f}, {stats_with_error['IgG']['poisson_ci'][1]:.1f}]")

        print(f"\n  apo-Tf:")
        print(f"    Count: {stats_with_error['apo-Tf']['count']}")
        print(f"    Standard error: ±{stats_with_error['apo-Tf']['standard_error']:.1f}")
        print(f"    Relative error: ±{stats_with_error['apo-Tf']['relative_error']:.1f}%")
        print(f"    95% CI: [{stats_with_error['apo-Tf']['poisson_ci'][0]:.1f}, {stats_with_error['apo-Tf']['poisson_ci'][1]:.1f}]")

    def plot_similarity_distribution(self, output_file: str = 'similarity_distribution.png'):
        """
        Plot similarity distribution scatter plot.
        """
        if not self.classification_results:
            print("No classification results available for plotting")
            return

        plt.figure(figsize=(12, 8))

        igg_sims = [r['similarities']['IgG'] for r in self.classification_results]
        aptf_sims = [r['similarities']['apo-Tf'] for r in self.classification_results]

        colors = []
        markers = []
        for r in self.classification_results:
            if r['predicted_class'] == 'IgG':
                colors.append('red')
                markers.append('o')
            elif r['predicted_class'] == 'apo-Tf':
                colors.append('blue')
                markers.append('s')
            else:
                colors.append('gray')
                markers.append('x')

        for i in range(len(self.classification_results)):
            plt.scatter(igg_sims[i], aptf_sims[i],
                        c=colors[i], marker=markers[i],
                        alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Equal similarity line')

        plt.axvline(x=self.min_similarity_threshold, color='gray', linestyle=':',
                    alpha=0.5, label=f'Threshold: {self.min_similarity_threshold}')
        plt.axhline(y=self.min_similarity_threshold, color='gray', linestyle=':', alpha=0.5)

        plt.xlabel('Similarity to IgG standard card')
        plt.ylabel('Similarity to apo-Tf standard card')
        plt.title('Event classification results')
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.6, label='IgG class'),
                           Patch(facecolor='blue', alpha=0.6, label='apo-Tf class'),
                           Patch(facecolor='gray', alpha=0.6, label='Unclassified')]
        plt.legend(handles=legend_elements)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.show()
        print(f"Similarity distribution plot saved to: {output_file}")

    def plot_results_with_error_bars(self, output_file: str = 'classification_with_error.png'):
        """
        Plot classification results with error bars.
        """
        if not self.classification_results:
            print("No classification results available for plotting")
            return

        stats_with_error = self.get_class_counts_with_error()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax1 = axes[0]
        categories = ['IgG', 'apo-Tf']
        counts = [stats_with_error['IgG']['count'], stats_with_error['apo-Tf']['count']]
        errors = [stats_with_error['IgG']['poisson_margin'],
                  stats_with_error['apo-Tf']['poisson_margin']]

        bars = ax1.bar(categories, counts, yerr=errors, capsize=10,
                       color=['#FF6B6B', '#4ECDC4'], alpha=0.7,
                       edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Event count', fontsize=12)
        ax1.set_title('Classification counts (95% CI, Poisson error)', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')

        for bar, count, err in zip(bars, counts, errors):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.5,
                     f'{count} ± {err:.1f}', ha='center', va='bottom', fontsize=10)

        ax2 = axes[1]
        rel_errors = [stats_with_error['IgG']['relative_error'],
                      stats_with_error['apo-Tf']['relative_error']]
        bars2 = ax2.bar(categories, rel_errors, color=['#FF6B6B', '#4ECDC4'],
                        alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Relative error (%)', fontsize=12)
        ax2.set_title('Relative error comparison (1σ)', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, err in zip(bars2, rel_errors):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{err:.1f}%', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\nResult plot saved to: {output_file}")

    def export_to_csv(self, output_file: str):
        """
        Export classification results to CSV.
        """
        df = pd.DataFrame([{
            'Event': r['card_id'],
            'Num_Peaks': len(r['peaks']),
            'Peak_Positions': str(r['peaks'].tolist()),
            'IgG_Similarity': round(r['similarities'].get('IgG', 0), 3),
            'apo-Tf_Similarity': round(r['similarities'].get('apo-Tf', 0), 3),
            'Predicted_Class': r['predicted_class'],
            'Best_Score': round(r['best_score'], 3),
            'Confidence': round(r['confidence'], 3),
            'Certainty_Prob': round(r['certainty_prob'], 3)
        } for r in self.classification_results])

        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nClassification results saved to: {output_file}")
        return df

    def export_error_analysis(self, output_file: str = 'error_analysis.csv'):
        """
        Export error analysis results.
        """
        stats_with_error = self.get_class_counts_with_error()

        error_data = {
            'Class': ['IgG', 'apo-Tf', 'Uncertain'],
            'Count': [stats_with_error['IgG']['count'],
                      stats_with_error['apo-Tf']['count'],
                      stats_with_error['Uncertain']['count']],
            'Standard error (±)': [stats_with_error['IgG']['standard_error'],
                                   stats_with_error['apo-Tf']['standard_error'],
                                   0],
            'Relative error (%)': [f"{stats_with_error['IgG']['relative_error']:.1f}%",
                                   f"{stats_with_error['apo-Tf']['relative_error']:.1f}%",
                                   "N/A"],
            '95% CI lower': [stats_with_error['IgG']['poisson_ci'][0],
                             stats_with_error['apo-Tf']['poisson_ci'][0],
                             stats_with_error['Uncertain']['count']],
            '95% CI upper': [stats_with_error['IgG']['poisson_ci'][1],
                             stats_with_error['apo-Tf']['poisson_ci'][1],
                             stats_with_error['Uncertain']['count']],
            'Error margin (±)': [stats_with_error['IgG']['poisson_margin'],
                                 stats_with_error['apo-Tf']['poisson_margin'],
                                 0]
        }

        df_error = pd.DataFrame(error_data)
        df_error.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Error analysis saved to: {output_file}")

        if stats_with_error['IgG']['count'] > 0:
            ratio = stats_with_error['apo-Tf']['count'] / stats_with_error['IgG']['count']
            ratio_se = ratio * np.sqrt(
                (stats_with_error['apo-Tf']['standard_error'] / stats_with_error['apo-Tf']['count'])**2 +
                (stats_with_error['IgG']['standard_error'] / stats_with_error['IgG']['count'])**2
            ) if stats_with_error['apo-Tf']['count'] > 0 and stats_with_error['IgG']['count'] > 0 else 0

            z_score = stats.norm.ppf((1 + 0.95) / 2)
            ratio_ci_low = max(0, ratio - z_score * ratio_se)
            ratio_ci_high = ratio + z_score * ratio_se

            ratio_data = pd.DataFrame({
                'Metric': ['Tf:IgG ratio', 'Ratio standard error', 'Ratio 95% CI lower', 'Ratio 95% CI upper'],
                'Value': [ratio, ratio_se, ratio_ci_low, ratio_ci_high]
            })
            ratio_output = output_file.replace('.csv', '_ratio.csv')
            ratio_data.to_csv(ratio_output, index=False)
            print(f"Ratio analysis saved to: {ratio_output}")

        return df_error


def main():
    """
    Main function: run standard-card classification with error estimation.
    """
    print("="*80)
    print("Binary classifier based on standard peak cards (independent error estimation)")
    print("Classifies unknown events using IgG and apo-Tf standard peak cards")
    print("="*80)

    # ======================================================
    # User configuration
    # ======================================================

    # 1. Input file: cards to classify
    INPUT_FILE = r"H:\数据\2026年\Supplementary experiment\Supplementary experiment nanopore\0330-15nm-SiO2-Cu-Fe-IgG apo-Tf-good-yes\2 50-50 Tf-IgG-good\0401\26401017events-2\peak_cards_fwhm_summary-1.csv"

    # 2. Output file
    OUTPUT_FILE = r"H:\数据\2026年\Supplementary experiment\Supplementary experiment nanopore\0330-15nm-SiO2-Cu-Fe-IgG apo-Tf-good-yes\2 50-50 Tf-IgG-good\0401\26401017events-2\classified_events_with_error.csv"

    # 3. Standard cards
    STANDARD_CARDS = {
        'IgG': [0.827, 0.858, 0.908, 0.931, 0.961, 0.985],
        'apo-Tf': [0.655, 0.704, 0.746, 0.793, 0.867, 0.895, 0.948, 0.978]
    }

    # 4. Classification parameters
    TOLERANCE = 0.004
    MIN_PEAKS = 2
    MIN_SIMILARITY_THRESHOLD = 0.35
    CONFIDENCE_LEVEL = 0.95

    # ======================================================
    # The code below normally does not need modification
    # ======================================================

    classifier = PeakCardClassifier(
        tolerance=TOLERANCE,
        min_peaks_per_card=MIN_PEAKS,
        min_similarity_threshold=MIN_SIMILARITY_THRESHOLD,
        confidence_level=CONFIDENCE_LEVEL
    )

    classifier.set_standard_cards(STANDARD_CARDS)

    loaded_count = classifier.load_cards_from_csv(INPUT_FILE)

    if loaded_count < 1:
        print("No valid cards found. Exiting.")
        return

    print(f"\nParameters:")
    print(f"  Matching tolerance: ±{TOLERANCE}")
    print(f"  Minimum peaks: {MIN_PEAKS}")
    print(f"  Minimum similarity threshold: {MIN_SIMILARITY_THRESHOLD}")
    print(f"  Confidence level: {CONFIDENCE_LEVEL*100}%")
    print(f"  Standard cards: {list(STANDARD_CARDS.keys())}")

    df_results = classifier.classify_all()

    classifier.print_summary()

    classifier.export_to_csv(OUTPUT_FILE)

    classifier.export_error_analysis()

    classifier.plot_similarity_distribution()

    classifier.plot_results_with_error_bars()

    print("\n" + "="*80)
    print("Classification result preview (first 10 events):")
    print("="*80)
    if not df_results.empty:
        print(tabulate(df_results.head(10), headers='keys',
                       tablefmt='grid', floatfmt='.3f', showindex=False))

    print(f"\nFull results saved to: {OUTPUT_FILE}")

    print("\n" + "="*80)
    print("Concentration quantification results (with independent error ranges)")
    print("="*80)

    stats_with_error = classifier.get_class_counts_with_error()
    counts = classifier.get_class_counts()
    total = sum(counts.values())
    classified_total = total - counts.get('Uncertain', 0)

    if classified_total > 0:
        tf_count = stats_with_error['apo-Tf']['count']
        igg_count = stats_with_error['IgG']['count']
        tf_margin = stats_with_error['apo-Tf']['poisson_margin']
        igg_margin = stats_with_error['IgG']['poisson_margin']

        print(f"Total events: {total}")
        print(f"Valid classified events: {classified_total} ({classified_total/total*100:.1f}%)")
        print(f"Unclassified events: {counts.get('Uncertain', 0)} ({counts.get('Uncertain', 0)/total*100:.1f}%)")

        print(f"\nClassified event counts (with 95% confidence interval):")
        print(f"  apo-Tf (Tf): {tf_count} ± {tf_margin:.1f} events")
        print(f"               95% CI: [{stats_with_error['apo-Tf']['poisson_ci'][0]:.1f}, {stats_with_error['apo-Tf']['poisson_ci'][1]:.1f}]")
        print(f"               Relative error: ±{stats_with_error['apo-Tf']['relative_error']:.1f}%")
        print(f"  IgG: {igg_count} ± {igg_margin:.1f} events")
        print(f"        95% CI: [{stats_with_error['IgG']['poisson_ci'][0]:.1f}, {stats_with_error['IgG']['poisson_ci'][1]:.1f}]")
        print(f"        Relative error: ±{stats_with_error['IgG']['relative_error']:.1f}%")

        if igg_count > 0:
            ratio = tf_count / igg_count
            ratio_se = ratio * np.sqrt(
                (stats_with_error['apo-Tf']['standard_error'] / tf_count)**2 +
                (stats_with_error['IgG']['standard_error'] / igg_count)**2
            ) if tf_count > 0 and igg_count > 0 else 0

            z_score = stats.norm.ppf((1 + CONFIDENCE_LEVEL) / 2)
            ratio_ci_low = max(0, ratio - z_score * ratio_se)
            ratio_ci_high = ratio + z_score * ratio_se

            print(f"\n  Tf : IgG = {ratio:.2f} : 1")
            print(f"  Ratio 95% CI: [{ratio_ci_low:.2f}, {ratio_ci_high:.2f}] : 1")
            print(f"  Ratio standard error: ±{ratio_se:.2f}")
        else:
            print(f"\n  Tf : IgG = only Tf detected")

        print(f"\nExpected ratio: Tf:IgG = 3:1")

        print("\n" + "-"*40)
        print("Error bar data (ready for plotting)")
        print("-"*40)
        print(f"\napo-Tf: {tf_count} ± {tf_margin:.1f}")
        print(f"IgG: {igg_count} ± {igg_margin:.1f}")
        print(f"\n# Note: errors are based on the Poisson distribution; standard error = sqrt(n)")
        print(f"# apo-Tf standard error = sqrt({tf_count}) = {stats_with_error['apo-Tf']['standard_error']:.1f}")
        print(f"# IgG standard error = sqrt({igg_count}) = {stats_with_error['IgG']['standard_error']:.1f}")

    else:
        print("No events were successfully classified. Adjust parameters:")
        print("  1. Increase TOLERANCE (e.g., 0.005 -> 0.008)")
        print("  2. Decrease MIN_SIMILARITY_THRESHOLD (e.g., 0.35 -> 0.25)")


if __name__ == "__main__":
    try:
        import pandas as pd
        import numpy as np
        from tabulate import tabulate
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError as e:
        print(f"Missing required library: {e}")
        print("Please run: pip install pandas numpy scipy tabulate matplotlib")
        exit()

    main()
