import json
import numpy as np
import torch

class MetricManager(object):
    def __init__(self,
                 restype_to_int,
                 weight_metrics,
                 sum_metrics,
                 count_metrics,
                 extra_metrics,
                 dataset_names, 
                 polymer_mask_names, 
                 interface_mask_names):
        self.restype_to_int = restype_to_int
        self.weight_metrics = weight_metrics
        self.sum_metrics = sum_metrics
        self.count_metrics = count_metrics
        self.extra_metrics = extra_metrics
        self.dataset_names = dataset_names
        self.polymer_mask_names = polymer_mask_names
        self.interface_mask_names = interface_mask_names
        
        self.all_mask_names = self.get_all_masks()
        self.mask_to_row = dict(zip(self.all_mask_names, range(len(self.all_mask_names))))
        self.row_to_mask = dict(zip(range(len(self.all_mask_names)), self.all_mask_names))

        self.metric_names = self.weight_metrics + list(self.sum_metrics) + list(map(lambda x: "pred" + x, list(self.count_metrics))) + list(map(lambda x: "true" + x, list(self.count_metrics))) + extra_metrics

        self.metric_to_col = dict(zip(self.metric_names, range(len(self.metric_names))))

        self.metrics = np.zeros((len(self.mask_to_row), 
                                 len(self.metric_to_col)), dtype = np.float64)
    
    def get_all_masks(self):
        all_mask_names = []

        for dataset_name in self.dataset_names:
            for polymer_mask_name in [""] + self.polymer_mask_names:
                for interface_mask_name in [""] + self.interface_mask_names:
                    mask_name = dataset_name
                    if polymer_mask_name != "":
                        mask_name += ("_" + polymer_mask_name)
                    if interface_mask_name != "":
                        mask_name += ("_" + interface_mask_name)
                    all_mask_names.append(mask_name)

        return all_mask_names
    
    def zero_metrics(self):
        self.metrics = np.zeros((len(self.mask_to_row), 
                                 len(self.metric_to_col)), dtype = np.float64)

    def accumulate_metrics_for_mask(self, 
                                    loss, 
                                    accuracy, 
                                    canonical_base_pair_accuracy, 
                                    canonical_base_pair_mask, 
                                    S_true, 
                                    S_pred, 
                                    mask_name, 
                                    mask):
        mask_row = self.mask_to_row[mask_name]
        
        if "weights" in self.weight_metrics:
            weights_col = self.metric_to_col["weights"]
            self.metrics[mask_row, weights_col] += \
                torch.sum(mask).cpu().data.numpy()
        
        if "canonicalBasePairWeights" in self.weight_metrics:
            canonical_base_pair_weights_col = self.metric_to_col["canonicalBasePairWeights"]
            self.metrics[mask_row, canonical_base_pair_weights_col] += \
                torch.sum(mask * canonical_base_pair_mask).cpu().data.numpy()
            
        if "loss" in self.sum_metrics:
            loss_col = self.metric_to_col["loss"]
            self.metrics[mask_row, loss_col] += \
                torch.sum(loss * mask).cpu().data.numpy()
        
        if "accuracy" in self.sum_metrics:
            accuracy_col = self.metric_to_col["accuracy"]
            self.metrics[mask_row, accuracy_col] += \
                torch.sum(accuracy * mask).cpu().data.numpy()
        
        if "canonicalBasePairAccuracy" in self.sum_metrics:
            canonical_base_pair_accuracy_col = self.metric_to_col["canonicalBasePairAccuracy"]
            self.metrics[mask_row, canonical_base_pair_accuracy_col] += \
                torch.sum(canonical_base_pair_accuracy * mask * canonical_base_pair_mask).cpu().data.numpy()
        
        for residue_name in self.count_metrics:
            true_count_col = self.metric_to_col["true" + residue_name]
            self.metrics[mask_row, true_count_col] += \
                torch.sum((S_true == self.restype_to_int[residue_name]).long() * mask)
            
            pred_count_col = self.metric_to_col["pred" + residue_name]
            self.metrics[mask_row, pred_count_col] += \
                torch.sum((S_pred == self.restype_to_int[residue_name]).long() * mask)

    def accumulate(self, 
                   loss, 
                   accuracy, 
                   canonical_base_pair_accuracy, 
                   canonical_base_pair_mask, 
                   S_true, 
                   S_pred, 
                   train_or_valid, 
                   mask_for_loss, 
                   polymer_masks, 
                   interface_masks):
        for polymer_mask_name in [""] + list(polymer_masks.keys()):
            for interface_mask_name in [""] + list(interface_masks.keys()):
                mask_name = train_or_valid
                mask = mask_for_loss

                if polymer_mask_name != "":
                    mask_name += ("_" + polymer_mask_name)
                    mask = mask * polymer_masks[polymer_mask_name]
                if interface_mask_name != "":
                    mask_name += ("_" + interface_mask_name)
                    mask = mask * interface_masks[interface_mask_name]
                
                self.accumulate_metrics_for_mask(loss, 
                                                 accuracy, 
                                                 canonical_base_pair_accuracy, 
                                                 canonical_base_pair_mask, 
                                                 S_true, 
                                                 S_pred, 
                                                 mask_name, 
                                                 mask)
    
    def compute_metrics(self):
        for metric in self.sum_metrics:
            weight_metric = self.sum_metrics[metric]
            weights_col = self.metric_to_col[weight_metric]
            weights = self.metrics[:, weights_col]
            weights_zero_mask = (weights == 0)

            metric_col = self.metric_to_col[metric]

            self.metrics[weights_zero_mask, metric_col] = np.nan
            self.metrics[~weights_zero_mask, metric_col] = self.metrics[~weights_zero_mask, metric_col] / weights[~weights_zero_mask]
        
        for metric in self.count_metrics:
            weight_metric = self.count_metrics[metric]
            weights_col = self.metric_to_col[weight_metric]
            weights = self.metrics[:, weights_col]
            weights_zero_mask = (weights == 0)

            true_metric = "true" + metric
            true_metric_col = self.metric_to_col[true_metric]
            self.metrics[weights_zero_mask, true_metric_col] = np.nan
            self.metrics[~weights_zero_mask, true_metric_col] = self.metrics[~weights_zero_mask, true_metric_col] / weights[~weights_zero_mask]

            pred_metric = "pred" + metric
            pred_metric_col = self.metric_to_col[pred_metric]
            self.metrics[weights_zero_mask, pred_metric_col] = np.nan
            self.metrics[~weights_zero_mask, pred_metric_col] = self.metrics[~weights_zero_mask, pred_metric_col] / weights[~weights_zero_mask]

        # Compute the perplexity. At this point, the loss column of metrics
        # will have already been normalized by the weights.
        if "perplexity" in self.extra_metrics:   
            loss_col = self.metric_to_col["loss"]
            loss = self.metrics[:, loss_col]
            perplexity_col = self.metric_to_col["perplexity"]

            self.metrics[:, perplexity_col] = np.exp(loss)
        
    def create_print_string(self, e, step, train_time, valid_time):
        output_string = f"epoch: {e+1}, step: {step}, train_time: {train_time}, valid_time: {valid_time}"

        for mask_row in range(len(self.row_to_mask)):
            mask_name = self.row_to_mask[mask_row]

            for metric in self.metric_names:
                metric_col = self.metric_to_col[metric]
                data = np.format_float_positional(np.float32(self.metrics[mask_row, metric_col]), unique=False, precision=3)

                output_string += (f", {mask_name}_{metric}: {data}")

        return output_string

def generate_metric_manager(restype_to_int, metrics_to_compute="basic"):
    if metrics_to_compute == "basic":
        dataset_names = ["train", "valid"]
        polymer_mask_names = ["protein", "dna", "rna"]
        weight_metrics = [
            "weights",
            "canonicalBasePairWeights" 
        ]
        sum_metrics = {
            "loss": "weights", 
            "accuracy": "weights",
            "canonicalBasePairAccuracy": "canonicalBasePairWeights"
        }
        count_metrics = {}
        extra_metrics = [
            "perplexity"
        ]
        interface_mask_names = []
    elif metrics_to_compute == "all":
        dataset_names = ["train", "valid"]
        polymer_mask_names = ["protein", "dna", "rna"]
        weight_metrics = [
            "weights", 
            "canonicalBasePairWeights"
        ]
        sum_metrics = {
            "loss": "weights", 
            "accuracy": "weights",
            "canonicalBasePairAccuracy": "canonicalBasePairWeights"
        }
        count_metrics = {
            "DA": "weights",
            "DC": "weights",
            "DG": "weights",
            "DT": "weights",
            "A": "weights",
            "C": "weights",
            "G": "weights",
            "U": "weights"
        }
        extra_metrics = [
            "perplexity"
        ]
        interface_mask_names = ["interface", "nonInterface"]
    elif metrics_to_compute == "na_only_inference":
        dataset_names = ["valid"]
        polymer_mask_names = ["dna", "rna"]
        weight_metrics = [
            "weights", 
            "canonicalBasePairWeights"
        ]
        sum_metrics = {
            "loss": "weights", 
            "accuracy": "weights",
            "canonicalBasePairAccuracy": "canonicalBasePairWeights"
        }
        count_metrics = {
            "DA": "weights",
            "DC": "weights",
            "DG": "weights",
            "DT": "weights",
            "A": "weights",
            "C": "weights",
            "G": "weights",
            "U": "weights"
        }
        extra_metrics = [
            "perplexity"
        ]
        interface_mask_names = []

    metric_manager = MetricManager(restype_to_int,
                                   weight_metrics,
                                   sum_metrics,
                                   count_metrics,
                                   extra_metrics,
                                   dataset_names,
                                   polymer_mask_names,
                                   interface_mask_names)
    return metric_manager


class DFMMetricManager:
    """
    Lightweight accumulator for DFM training diagnostics.
    Tracks masked-position loss and accuracy per (split × polymer × t-bin).
    Entropy vs t is a post-training diagnostic — compute it from the final checkpoint,
    not during training.

    flush_every controls logging granularity:
      None  — manual flush only (call flush() explicitly, e.g. at epoch end for validation)
      int N — use should_flush() to check; flush() writes a rolling-window line every N steps
    """

    def __init__(self, n_bins=10, polymer_names=None, splits=None, flush_every=None):
        self.n_bins        = n_bins
        self.bin_edges     = np.linspace(0.0, 1.0, n_bins + 1)
        self.polymer_names = polymer_names or ["protein", "dna", "rna"]
        self.splits        = splits        or ["train", "valid"]
        self.all_polymers  = ["all"] + self.polymer_names

        # Row index: split × polymer × t-bin
        self.row_labels = [
            f"{split}_{polymer}_t{self.bin_edges[b]:.1f}-{self.bin_edges[b+1]:.1f}"
            for split   in self.splits
            for polymer in self.all_polymers
            for b       in range(n_bins)
        ]
        self.row_to_idx = {r: i for i, r in enumerate(self.row_labels)}

        # Columns: MW — masked-position count, LM — loss sum, AM — accuracy sum, EH — entropy sum
        self._MW = 0
        self._LM = 1
        self._AM = 2
        self._EH = 3
        self.metrics = np.zeros((len(self.row_labels), 4), dtype=np.float64)

        # t histogram per split (sanity check on uniform sampling)
        self.t_hist    = {s: np.zeros(self.n_bins, dtype=np.float64) for s in self.splits}
        self.flush_every = flush_every
        self._steps    = 0

    def zero_metrics(self):
        self.metrics[:] = 0.0
        for s in self.splits:
            self.t_hist[s][:] = 0.0
        self._steps = 0

    def accumulate(self, loss, accuracy, entropy, t, X_t,
                   polymer_masks, split, mas_token_id):
        """
        loss         : [B, L]  per-position cross-entropy
        accuracy     : [B, L]  per-position 0/1 accuracy
        entropy      : [B, L]  per-position Shannon entropy of predicted distribution
        t            : [B, 1]  sampled timestep
        X_t          : [B, L]  noised input sequence
        polymer_masks: dict    {name: [B, L]}
        split        : str     "train" or "valid"
        mas_token_id : int
        """
        loss     = loss.detach().float()
        accuracy = accuracy.detach().float()
        entropy  = entropy.detach().float()
        mas_mask = (X_t == mas_token_id).float()
        t_vals   = t.squeeze(-1)  # [B]

        for b_idx in range(t_vals.shape[0]):
            t_val   = float(t_vals[b_idx].item())
            self.t_hist[split][min(int(t_val * self.n_bins), self.n_bins - 1)] += 1

            bin_idx = min(int(t_val * self.n_bins), self.n_bins - 1)
            bin_str = f"t{self.bin_edges[bin_idx]:.1f}-{self.bin_edges[bin_idx+1]:.1f}"

            for polymer in self.all_polymers:
                row_label = f"{split}_{polymer}_{bin_str}"
                r = self.row_to_idx[row_label]
                pm = mas_mask[b_idx] if polymer == "all" \
                     else mas_mask[b_idx] * polymer_masks[polymer][b_idx].float()

                self.metrics[r, self._MW] += pm.sum().item()
                self.metrics[r, self._LM] += (loss[b_idx] * pm).sum().item()
                self.metrics[r, self._AM] += (accuracy[b_idx] * pm).sum().item()
                self.metrics[r, self._EH] += (entropy[b_idx] * pm).sum().item()

        self._steps += 1

    def should_flush(self):
        return self.flush_every is not None and self._steps >= self.flush_every

    def flush(self, step, logfile, extra_fields=None):
        """Normalise, write a summary line + JSONL record, then reset. No-op if empty."""
        if self._steps == 0:
            return
        self.compute_metrics()
        line = f"step: {step}, window: {self._steps}, " + self.create_print_string()
        if extra_fields:
            for k, v in extra_fields.items():
                line += f", {k}: {v}"
        with open(logfile, 'a') as f:
            f.write(line + "\n")
        record = self.to_dict(step, self._steps)
        if extra_fields:
            record.update(extra_fields)
        jsonl_path = logfile.rsplit('.', 1)[0] + '.jsonl'
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps(record) + "\n")
        self.zero_metrics()
        return record

    def compute_metrics(self):
        mw = self.metrics[:, self._MW]
        nz = mw > 0
        for col in [self._LM, self._AM, self._EH]:
            self.metrics[~nz, col] = np.nan
            self.metrics[nz,  col] /= mw[nz]

    def create_print_string(self):
        fmt = lambda v: np.format_float_positional(np.float32(v), unique=False, precision=3)
        parts = []
        for split in self.splits:
            for b in range(self.n_bins):
                row_label = f"{split}_all_t{self.bin_edges[b]:.1f}-{self.bin_edges[b+1]:.1f}"
                r  = self.row_to_idx[row_label]
                lm = self.metrics[r, self._LM]
                if self.metrics[r, self._MW] == 0 or np.isnan(lm):
                    continue
                parts.append(f"{row_label}_loss_mas: {fmt(lm)}")
        return ", ".join(parts)

    def to_dict(self, step, window_size):
        def _f(v):
            return None if np.isnan(v) else float(v)
        d = {"step": step, "window": window_size}
        for row_label, r in self.row_to_idx.items():
            if self.metrics[r, self._MW] == 0:
                continue
            d[f"{row_label}_loss_mas"]    = _f(self.metrics[r, self._LM])
            d[f"{row_label}_acc_mas"]     = _f(self.metrics[r, self._AM])
            d[f"{row_label}_entropy_mas"] = _f(self.metrics[r, self._EH])
        for split in self.splits:
            hist  = self.t_hist[split]
            total = hist.sum()
            d[f"{split}_t_hist"] = (hist / total).tolist() if total > 0 else None
        return d
