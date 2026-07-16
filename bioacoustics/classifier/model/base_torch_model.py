import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
import json
import pandas as pd
import random
from bioacoustics.classifier.model.acoustic_model import AcousticModel
from bioacoustics.classifier.metrics_calculator import MetricsCalculator
import logging

logger = logging.getLogger(__name__)


class BaseTorchModel(AcousticModel):
    """
    Generic base class for PyTorch deep-learning models.
    SVM and other classical models will still inherit from AcousticModel,
    while CNN-based models inherit from this class.
    """

    def __init__(self, device=None, output_dir=None, model_dir: str | None = None,
                 num_channels=1, init_mode="he_normal", dropout_rate=0.2, weight_constraint=None):
        super().__init__(output_dir=output_dir, model_dir=model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.init_mode = "he_normal"
        self.weight_constraint = None
        self.l2_reg = 0.0
        self.dropout_rate = 0.2
        self.num_channels = num_channels
        self._make_model(init_mode=init_mode, dropout_rate=dropout_rate, weight_constraint=weight_constraint)

    @abstractmethod
    def _make_cnn_model(self):
        """make cnn model"""
    def _init_weights(self, layer):
        """Apply proper weight initialization."""
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if self.init_mode == "glorot_uniform":
                nn.init.xavier_uniform_(layer.weight)
            elif self.init_mode == "he_normal":
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _apply_max_norm(self, max_norm):
        """Applies MaxNorm constraint after each optimizer step."""
        if max_norm is None:
            return
        with torch.no_grad():
            for param in self.acoustic_model.parameters():
                if param.dim() > 1:  # weight tensors only
                    norm = param.norm(2)
                    if norm > max_norm:
                        param.mul_(max_norm / norm)

    def _make_model(
        self,
        init_mode="he_normal",
        dropout_rate=0.2,
        weight_constraint=None,
    ):
        """Creates the CNN model, sets hyperparams, initializes weights."""
        self.init_mode = init_mode
        self.weight_constraint = weight_constraint
        self.dropout_rate = dropout_rate

        # subclass implements _make_cnn_model
        self._make_cnn_model()

        # apply init
        self.acoustic_model.apply(self._init_weights)
        self.acoustic_model.to(self.device)

    def fit(self,
               x_train,
               y_train,
               x_val=None,
               y_val=None,
               *args,
               **kwargs):
        """Default PyTorch training loop."""

        batch_size = kwargs.get("batch_size", 32)
        learning_rate = kwargs.get("learning_rate", 0.001)
        weight_decay = kwargs.get("weight_decay", 0.01)
        epochs = kwargs.get("epoch", 5)
        frames_per_chunk = kwargs.get("frames_per_chunk", 64)
        samples_per_epoch = kwargs.get("samples_per_epoch", len(x_train))
        augmentation = kwargs.get("augmentation", True)
        set_threshold = kwargs.get("set_threshold", False)

        train_dataset = RandomCropDataset(
            x_train, y_train, frames_per_chunk,
            samples_per_epoch=samples_per_epoch,
        )
        val_dataset = SequentialChunkDataset(
            x_val, y_val, frames_per_chunk,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size, shuffle=False
        )

        optimizer = optim.AdamW(
            self.acoustic_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay  # self.l2_reg  # L2 regularization here
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",  # because F1 is monitored
            factor=0.5,  # cut LR in half
            patience=2,  # wait 2 epochs
            min_lr=1e-6,
        )

        spec_aug = SpecAugment(
            freq_mask_param=8,
            time_mask_param=10,
            num_freq_masks=2,
            num_time_masks=2,
            p=0.7
        )

        # num_pos = (y_train == 1).sum().item()
        # num_neg = (y_train == 0).sum().item()

        # pos_weight = num_neg / num_pos
        # weights = torch.tensor([1.0, pos_weight]).to(self.device)

        # criterion = nn.CrossEntropyLoss(weight=weights)
        criterion = nn.CrossEntropyLoss()
        history = {"train_loss": [], "val_loss": []}
        best_val_f1 = 0
        best_state = None
        metrics_calculator = MetricsCalculator()

        for epoch in range(epochs):
            self.acoustic_model.train()
            total_loss = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                if augmentation:
                    xb = spec_aug(xb)
                optimizer.zero_grad()
                outputs = self.acoustic_model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()

                # apply maxnorm constraint
                self._apply_max_norm(self.weight_constraint)

                total_loss += loss.item()

            self.acoustic_model.eval()
            val_logits = []
            val_targets = []
            val_loss = 0

            with torch.no_grad():
                for batch in val_loader:
                    xb, yb = batch[0].to(self.device), batch[1].to(self.device)
                    logits = self.acoustic_model(xb)
                    loss = criterion(logits, yb)
                    val_loss += loss.item()
                    val_logits.append(logits.cpu())
                    val_targets.append(yb.cpu())

            history["train_loss"].append(total_loss / len(train_loader))
            history["val_loss"].append(val_loss / len(val_loader))

            val_logits = torch.cat(val_logits)
            val_targets = torch.cat(val_targets)
            val_probs = torch.softmax(val_logits, dim=1).numpy()
            val_targets_np = val_targets.numpy()

            val_metrics = metrics_calculator.compute_metrics(
                y_true=val_targets_np,
                y_probs=val_probs,
            )
            val_f1 = val_metrics['eval_metrics']['f1']
            logging.info(f"Epoch {epoch+1}/{epochs} "
                         f"train loss={history['train_loss'][-1]:.4f} "
                         f"val loss ={history['val_loss'][-1]:.4f},"
                         f"val_f1={val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.acoustic_model.state_dict().items()
                }
            scheduler.step(val_f1)

        # plot history
        self._plot_measures(history, title="_TORCH_CNN")

        # save model
        self.acoustic_model.load_state_dict(best_state)
        self._save_model()

        if set_threshold:
            # calculate the best threshold
            chunk_results, file_results = self.evaluate_model(x_val, y_val, batch_size=batch_size,
                                                              tune_threshold=True,
                                                              frames_per_chunk=frames_per_chunk)
            if "threshold" in chunk_results:
                self.threshold = chunk_results["threshold"]
                self._save_threshold(self.threshold)

    def evaluate_model(
            self, x, y, **kwargs):
        """
        Evaluate model on labeled data.
        If tune_threshold=True, threshold is optimized (validation only).
        """

        batch_size = kwargs.get("batch_size", 32)
        frames_per_chunk = kwargs.get("frames_per_chunk", 64)
        tune_threshold = kwargs.get("tune_threshold", False)

        self.acoustic_model.eval()

        dataset = SequentialChunkDataset(x, y, frames_per_chunk)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        all_probs = []
        all_targets = []
        all_file_indices = []

        with torch.no_grad():
            for batch in loader:
                xb, yb = batch[0].to(self.device), batch[1].to(self.device)
                file_idx = batch[2]
                logits = self.acoustic_model(xb)
                loss = criterion(logits, yb)
                total_loss += loss.item()

                probs = torch.softmax(logits, dim=1)

                all_probs.append(probs.cpu())
                all_targets.append(yb.cpu())
                all_file_indices.extend(file_idx.tolist())

        avg_loss = total_loss / len(loader)

        probs = torch.cat(all_probs).numpy()
        y_true = torch.cat(all_targets).numpy()

        # --- Metrics ---
        self._load_threshold()
        metrics_calculator = MetricsCalculator()
        eval_metrics = metrics_calculator.compute_metrics(
            y_true=y_true,
            y_probs=probs,
            threshold=self.threshold,
            tune_threshold=tune_threshold,
        )

        file_agg_results = aggregate_predictions_per_file(
            probs=probs,
            file_indices=all_file_indices,
            dataset=dataset,
            method="mean",  # "topk_mean",
            k=3
        )

        # file-level metrics
        file_probs = np.array([r["prob"] for r in file_agg_results])
        file_labels = np.array(y)  # original per-file labels
        file_metrics = metrics_calculator.compute_metrics(
            y_true=file_labels,
            y_probs=file_probs,
            threshold=self.threshold,
            tune_threshold=tune_threshold,  # false
        )
        logging.info("chunk_metrics----------", eval_metrics['eval_metrics'])
        logging.info("file_metrics-----------", file_metrics['eval_metrics'])

        chunk_results = {
            "loss": avg_loss,
            "metrics": eval_metrics['eval_metrics'],
            "probs": probs,
            "y_true": y_true,
            "preds": eval_metrics["preds"],
        }
        if "threshold" in eval_metrics:
            chunk_results["threshold"] = eval_metrics["threshold"]
            chunk_results["threshold_score"] = eval_metrics["threshold_score"]

        file_results = {
            "metrics": file_metrics['eval_metrics'],
            "probs": file_probs,
            "y_true": file_labels,
            "preds": file_metrics["preds"],
        }

        return chunk_results, file_results

    def _predict_proba(self, x, **kwargs):
        """
        Predict class probabilities for unlabeled data.
        """
        self.acoustic_model.eval()
        batch_size = kwargs.get("batch_size", 32)
        frames_per_chunk = kwargs.get("frames_per_chunk", 64)

        dataset = SequentialChunkDataset(x, None, frames_per_chunk)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_probs = []
        all_file_indices = []

        with torch.no_grad():
            for batch in loader:
                xb, file_idx = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.acoustic_model(xb)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu())
                all_file_indices.extend(file_idx.tolist())

        chunk_probs = torch.cat(all_probs).numpy()
        file_agg_results = aggregate_predictions_per_file(
            probs=chunk_probs,
            file_indices=all_file_indices,
            dataset=dataset,
            method="mean", #"topk_mean",  # "mean",
            k=3
        )
        return chunk_probs, file_agg_results

    def _predict(self, x, **kwargs):
        """
        Predict binary labels using stored or default threshold.
        """
        batch_size = kwargs.get("batch_size", 32)
        frames_per_chunk = kwargs.get("frames_per_chunk", 64)
        self._load_threshold()

        chunk_probs, file_agg_results = self._predict_proba(x, batch_size=batch_size,
                                                            frames_per_chunk=frames_per_chunk)
        file_probs = np.array([r["prob"] for r in file_agg_results])
        chunk_predicts = (chunk_probs[:, 1] >= self.threshold).astype(int)
        file_predicts = (file_probs[:, 1] >= self.threshold).astype(int)
        chunk_results = {
            "probs": chunk_probs,
            "preds": chunk_predicts,
        }
        file_results = {
            "probs": file_probs,
            "preds": file_predicts,
        }

        return {"chunk_results": chunk_results, "file_results": file_results}

    def _save_model(self):
        model_path = os.path.join(self.output_dir, "model.pth")
        torch.save(self.acoustic_model.state_dict(), model_path)

    def _load_model(self):
        model_path = os.path.join(self.model_dir, "model.pth")
        self.acoustic_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.acoustic_model.to(self.device)
        self.acoustic_model.eval()

    def save_evaluation(
            self,
            probs,
            metrics=None,
            predicts=None,
            y_true=None,
            split="",
    ):
        """Save both predictions and metrics for a dataset split."""
        if probs is None:
            raise RuntimeError("No probabilities found. Run evaluate_model or predict_proba first.")

        df = pd.DataFrame({
            "prob_neg": probs[:, 0],
            "prob_pos": probs[:, 1],
        })

        if y_true is not None:
            df["y_true"] = y_true
        if predicts is not None:
            df["predicts"] = predicts

        probs_path = f"{self.output_dir}{split}_probs.csv"
        df.to_csv(probs_path, index=False)

        metrics_path = f"{self.output_dir}{split}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        logging.info(f"Saved evaluation results for split='{split}'")


class SpecAugment:
    def __init__(
        self,
        freq_mask_param=8,
        time_mask_param=12,
        num_freq_masks=2,
        num_time_masks=2,
        p=0.8
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.p = p

    def _freq_mask(self, spec):
        _, _, F, _ = spec.shape

        f = random.randint(0, self.freq_mask_param)
        f0 = random.randint(0, F - f)

        spec[:, :, f0:f0+f, :] = 0

    def _time_mask(self, spec):
        _, _, _, T = spec.shape

        t = random.randint(0, self.time_mask_param)
        t0 = random.randint(0, T - t)

        spec[:, :, :, t0:t0+t] = 0

    def __call__(self, x):
        if random.random() > self.p:
            return x

        x = x.clone()

        for _ in range(self.num_freq_masks):
            self._freq_mask(x)

        for _ in range(self.num_time_masks):
            self._time_mask(x)

        return x


class RandomCropDataset(Dataset):
    """
    Training dataset: randomly crops a fixed-length chunk from each
    full-file spectrogram. Every epoch sees different slices — free
    augmentation that helps generalization.

    Each __getitem__ call picks one random window from one file.
    """

    def __init__(
            self,
            specs,
            labels,
            frames_per_chunk,
            samples_per_epoch=None,
            transform=None,
    ):
        """
        Parameters
        ----------
        specs : list of np.ndarray
            Each element shape (C, n_mels, n_frames_variable).
        labels : list/array of int
            Encoded label per file.
        frames_per_chunk : int
            Width of the crop in spectrogram frames.
        samples_per_epoch : int, optional
            How many samples per epoch. If None, uses len(specs).
            Set higher to oversample short files or balance classes.
        transform : callable, optional
            Augmentation (e.g., SpecAugment) applied to each chunk.
        """
        self.frames_per_chunk = frames_per_chunk
        self.transform = transform

        # filter out files shorter than one chunk
        self.specs = []
        self.labels = []

        for spec, label in zip(specs, labels):
            n_frames = spec.shape[-1]
            if n_frames < frames_per_chunk:
                # pad with zeros on the right
                pad_width = frames_per_chunk - n_frames
                spec = np.pad(spec, ((0, 0), (0, 0), (0, pad_width)), mode="constant")
            self.specs.append(spec)
            self.labels.append(label)

        if len(self.specs) == 0:
            raise ValueError(
                f"No spectrograms have >= {frames_per_chunk} frames. "
                f"Check your data or reduce chunk duration."
            )

        self.samples_per_epoch = samples_per_epoch or len(self.specs)


        label_arr = np.array(self.labels)
        n_frames = np.array([s.shape[-1] for s in self.specs])

        # weight by class balance AND file length
        class_counts = np.bincount(label_arr)
        class_weights = 1.0 / class_counts
        per_file_weight = class_weights[label_arr] * n_frames

        self.sample_weights = per_file_weight / per_file_weight.sum()

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # weighted random file selection for class balance
        file_idx = np.random.choice(len(self.specs), p=self.sample_weights)

        spec = self.specs[file_idx]
        n_frames = spec.shape[-1]

        # random start position
        max_start = n_frames - self.frames_per_chunk
        start = np.random.randint(0, max_start + 1)

        chunk = spec[:, :, start: start + self.frames_per_chunk].copy()

        if self.transform:
            chunk = self.transform(chunk)

        label = self.labels[file_idx]
        return (
            torch.tensor(chunk, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


class SequentialChunkDataset(Dataset):
    """
    Inference/evaluation dataset: yields all non-overlapping sequential
    chunks from each file. Tracks which file each chunk came from so
    predictions can be aggregated per file.
    """

    def __init__(self, specs, labels, frames_per_chunk, file_paths=None):
        """
        Parameters
        ----------
        specs : list of np.ndarray
            Each element shape (C, n_mels, n_frames_variable).
        labels : list/array of int or None
            Encoded label per file. None for predict mode.
        frames_per_chunk : int
            Width of each chunk in frames.
        file_paths : list of str, optional
            Original file paths for result mapping.
        """
        self.frames_per_chunk = frames_per_chunk
        self.specs = specs
        self.labels = labels
        self.file_paths = file_paths

        # build chunk index: (file_idx, start_frame)
        self.chunk_index = []
        self.file_indices = []  # which file each chunk belongs to

        for i, spec in enumerate(specs):
            n_frames = spec.shape[-1]
            n_full_chunks = n_frames // frames_per_chunk
            remainder = n_frames % frames_per_chunk

            for c in range(n_full_chunks):
                start = c * frames_per_chunk
                self.chunk_index.append((i, start))
                self.file_indices.append(i)

            # pad the leftover into one more chunk
            if remainder > 0:
                self.chunk_index.append((i, n_full_chunks * frames_per_chunk))
                self.file_indices.append(i)

    def __len__(self):
        return len(self.chunk_index)

    def __getitem__(self, idx):
        file_idx, start = self.chunk_index[idx]
        end = start + self.frames_per_chunk
        spec = self.specs[file_idx]

        chunk = spec[:, :, start:end].copy()

        # pad if last chunk is short
        if chunk.shape[-1] < self.frames_per_chunk:
            pad_width = self.frames_per_chunk - chunk.shape[-1]
            chunk = np.pad(chunk, ((0, 0), (0, 0), (0, pad_width)), mode="constant")

        chunk_tensor = torch.tensor(chunk, dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor(self.labels[file_idx], dtype=torch.long)
            return chunk_tensor, label, file_idx
        else:
            return chunk_tensor, file_idx

    def get_file_path(self, file_idx):
        """Get original file path for a file index."""
        if self.file_paths:
            return self.file_paths[file_idx]
        return f"file_{file_idx}"

    def num_files(self):
        return len(self.specs)


def aggregate_predictions_per_file(probs, file_indices, dataset, method="mean", k=3):
    """
    Aggregate chunk-level predictions to file-level.

    Parameters
    ----------
    probs : np.ndarray, shape (n_chunks, n_classes)
        Chunk-level probabilities.
    file_indices : list of int
        Which file each chunk belongs to.
    dataset : SequentialChunkDataset
        The dataset, to get file paths.
    method : str
        'mean' — average probabilities across chunks.
        'max' — take max probability across chunks.
        'topk_mean' — average top k chunks by positive class probability.
    k : int
        Number of top chunks to average (used with 'topk_mean').


    Returns
    -------
    list of dict with file_path, prob, predicted_label per file
    """
    file_indices = np.array(file_indices)
    results = []

    for file_idx in range(dataset.num_files()):
        mask = file_indices == file_idx
        if not mask.any():
            continue

        file_probs = probs[mask]

        if method == "mean":
            agg_prob = file_probs.mean(axis=0)
        elif method == "max":
            agg_prob = file_probs.max(axis=0)
        elif method == "topk_mean":
            pos_scores = file_probs[:, 1]
            top_k = min(k, len(pos_scores))
            top_indices = np.argsort(pos_scores)[-top_k:]
            agg_prob = file_probs[top_indices].mean(axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

        results.append({
            "file_path": dataset.get_file_path(file_idx),
            "file_idx": file_idx,
            "prob": agg_prob,
            "predicted_label": int(np.argmax(agg_prob)),
            "confidence": float(agg_prob.max()),
            "n_chunks": int(mask.sum()),
        })

    return results