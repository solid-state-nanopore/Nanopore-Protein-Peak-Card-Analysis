"""
Four-protein mixture peak-card classifier
Compare the peak card of an individual event with SA, BSA, IgG, and G6PD reference cards.
Return four similarity scores and the predicted class.
"""

import numpy as np
from typing import List, Dict, Tuple


class QuadPeakCardClassifier:
    """
    Four-protein reference-card classifier (single-event version).
    The similarity algorithm is the same as in the binary classifier:
      final_score = 0.7 * match_ratio + 0.3 * position_score
    """

    def __init__(self,
                 tolerance: float = 0.004,
                 min_similarity_threshold: float = 0.35):
        """
        Parameters:
        tolerance: Tolerance for matching peak positions.
        min_similarity_threshold: Minimum similarity threshold; scores below this value are marked Uncertain.
        """
        self.tolerance = tolerance
        self.min_similarity_threshold = min_similarity_threshold
        self.standard_cards: Dict[str, np.ndarray] = {}

    def set_standard_cards(self, standard_cards: Dict[str, List[float]]):
        """Set the four reference peak cards."""
        self.standard_cards = {}
        for name, peaks in standard_cards.items():
            self.standard_cards[name] = np.sort(np.array(peaks, dtype=float))
            print(f"Reference card {name:5s} -> {self.standard_cards[name]}")

    def calculate_similarity(self, peaks1: np.ndarray, peaks2: np.ndarray) -> float:
        """
        Calculate the similarity between two peak cards (0–1) using the original algorithm.
        """
        if len(peaks1) == 0 or len(peaks2) == 0:
            return 0.0

        p1 = np.sort(peaks1)
        p2 = np.sort(peaks2)

        # Match peaks greedily, using each reference peak at most once.
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

        # Position similarity based on differences between consecutive matched peaks.
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

    def classify_single_event(self, peaks: List[float]) -> Dict:
        """
        Classify the peak card of an individual event.
        Return the similarities, predicted class, and score margin in a dictionary.
        """
        peaks = np.sort(np.array(peaks, dtype=float))

        # Exclude peaks near the baseline (0.995–1.02), as implemented below.
        peaks = peaks[~((peaks >= 0.995) & (peaks <= 1.02))]

        similarities = {}
        for std_name, std_peaks in self.standard_cards.items():
            sim = self.calculate_similarity(peaks, std_peaks)
            similarities[std_name] = sim

        # Sort by similarity score.
        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        best_class, best_score = sorted_sims[0]
        second_score = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0
        confidence = best_score - second_score

        # Softmax-transformed relative scores; these are not calibrated probabilities.
        scores = np.array([s for _, s in similarities.items()])
        exp_scores = np.exp(scores * 5)
        probs = exp_scores / exp_scores.sum()
        prob_dict = {name: float(p) for name, p in zip(similarities.keys(), probs)}
        certainty_prob = float(np.max(probs))

        # Apply the acceptance threshold.
        if best_score < self.min_similarity_threshold:
            predicted_class = "Uncertain"
            certainty_prob = 0.0
        else:
            predicted_class = best_class

        return {
            'input_peaks': peaks.tolist(),
            'num_peaks': len(peaks),
            'similarities': similarities,
            'predicted_class': predicted_class,
            'best_score': best_score,
            'confidence': confidence,
            'certainty_prob': certainty_prob,
            'probs': prob_dict
        }

    def print_report(self, result: Dict):
        """Print the classification report."""
        print("\n" + "=" * 60)
        print("Single-event peak-card classification report")
        print("=" * 60)
        print(f"Number of input peaks: {result['num_peaks']}")
        print(f"Input peak positions: {[round(p, 3) for p in result['input_peaks']]}")
        print("-" * 60)
        print(f"{'Reference card':<16} {'Similarity':>12} {'Softmax score':>14}")
        print("-" * 60)
        # Print reference cards in descending order of similarity.
        sorted_items = sorted(result['similarities'].items(),
                              key=lambda x: x[1], reverse=True)
        for name, sim in sorted_items:
            prob = result['probs'][name]
            print(f"{name:<16} {sim:>12.4f} {prob:>14.4f}")
        print("-" * 60)
        print(f"Predicted class:     {result['predicted_class']}")
        print(f"Best similarity:   {result['best_score']:.4f}")
        print(f"Score margin: {result['confidence']:.4f}")
        print(f"Highest softmax score:   {result['certainty_prob']:.4f}")
        print("=" * 60)


def main():
    # ======================================================
    # User configuration
    # ======================================================

    # 1. Four reference peak cards
    STANDARD_CARDS = {
        'SA':   [0.974, 0.982, 0.987, 0.992],
        'BSA':  [0.927, 0.934, 0.943, 0.956, 0.968, 0.978, 0.987],
        'IgG':  [0.904, 0.931, 0.953, 0.979],
        'G6PD': [0.885, 0.908, 0.938, 0.957, 0.968],
    }

    # 2. Input: peak positions for one unknown event (edit here).
    UNKNOWN_PEAKS = [0.947, 0.958, 0.970, 0.979, 0.982]

    # 3. Parameters
    TOLERANCE = 0.006
    MIN_SIMILARITY_THRESHOLD = 0.6

    # ======================================================
    # The code below normally does not need modification.
    # ======================================================

    classifier = QuadPeakCardClassifier(
        tolerance=TOLERANCE,
        min_similarity_threshold=MIN_SIMILARITY_THRESHOLD
    )
    classifier.set_standard_cards(STANDARD_CARDS)

    result = classifier.classify_single_event(UNKNOWN_PEAKS)
    classifier.print_report(result)

    return result


if __name__ == "__main__":
    main()
