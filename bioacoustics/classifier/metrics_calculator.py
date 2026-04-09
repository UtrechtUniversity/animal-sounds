from sklearn import metrics
import numpy as np

SUPPORTED_METRICS = {
    "accuracy": metrics.accuracy_score,
    "precision": lambda y, p: metrics.precision_score(y, p, average="macro", zero_division=0),
    "recall": lambda y, p: metrics.recall_score(y, p, average="macro",zero_division=0),
    "f1": lambda y, p: metrics.f1_score(y, p, average="macro", zero_division=0),
}


def find_best_threshold(y_true, y_probs, metric_fn):
    best_t, best_score = 0.5, -1
    for t in np.linspace(0, 1, 101):
        preds = (y_probs >= t).astype(int)
        score = metric_fn(y_true, preds)
        if score > best_score:
            best_t, best_score = t, score
    return best_t, best_score


class MetricsCalculator:
    def compute_metrics(
        self,
        y_true,
        y_probs,
        threshold=None,
        tune_threshold=False,
        primary_metric="f1",
    ):
        """
        y_true: (N,) class indices
        y_probs: (N, C) probabilities
        """

        pos_probs = y_probs[:, 1]  # binary case

        metric_fn = SUPPORTED_METRICS[primary_metric]

        if tune_threshold:
            threshold, best_score = find_best_threshold(
                y_true, pos_probs, metric_fn
            )
        else:
            threshold = threshold or 0.5

        threshold = threshold or 0.5

        y_pred = (pos_probs >= threshold).astype(int)

        results = {}
        eval_metrics = {}
        for name, fn in SUPPORTED_METRICS.items():
            eval_metrics[name] = fn(y_true, y_pred)

        eval_metrics["auc"] = metrics.roc_auc_score(y_true, pos_probs)

        results['eval_metrics'] = eval_metrics
        results['preds'] = y_pred
        if tune_threshold:
            results["threshold"] = threshold
            results["threshold_score"] = best_score

        return results
