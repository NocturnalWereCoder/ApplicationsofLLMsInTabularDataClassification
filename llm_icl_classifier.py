# llm_icl_classifier.py
import json, numpy as np, logging
from sklearn.cluster import KMeans
from langchain_ollama.llms import OllamaLLM
from dataclasses import dataclass
from typing import List, Dict

logger = logging.getLogger(__name__)

@dataclass
class LLMICLParams:
    prototypes_per_class: int = 5
    max_examples: int = 20
    backend: str = "sampling"       # or "logprobs" (not implemented in this stub)
    samples: int = 21               # for sampling backend
    temperature: float = 0.4
    top_p: float = 0.95
    use_schema: bool = True
    verbalizers: Dict[int, List[str]] = None
    # NEW: control how often to log progress (in items). 1 = every item.
    progress_every: int = 1

class LLMICLClassifier:
    def __init__(self, params: LLMICLParams, model_name: str):
        self.p = params
        # pass top_p so it actually takes effect
        self.model = OllamaLLM(
            model=model_name,
            temperature=self.p.temperature,
            top_p=self.p.top_p,
            in_process=True
        )
        self.prototypes = []   # list[(row_dict, label_string)]
        self.classes_ = None
        self.class_keys = None # stable string keys aligned with classes_
        self.T = 1.0           # temperature for probability calibration

    def _serialize(self, row_dict):
        return json.dumps(row_dict, ensure_ascii=False)

    def _prompt(self, examples, test_row, label_map):
        head = (
            "You are a classifier. "
            f"Choose one label from {json.dumps(label_map)}.\n"
            'Return JSON: {"label": <key from the map>}\n\n'
            "Examples:\n"
        )
        shots = []
        for x, y in examples:
            shots.append(f"Input: {self._serialize(x)}\nLabel: {y}")
        test = f"\nInput: {self._serialize(test_row)}\nLabel:"
        return head + "\n".join(shots) + test

    def fit(self, X, y):
        # Stable class ordering and stable string keys for labeling
        self.classes_ = list(sorted(np.unique(y)))
        self.class_keys = [str(c) for c in self.classes_]

        # Choose prototypes per class with diversity (kmeans on X)
        X_arr = X.values.astype(float)
        y_arr = np.array(y)

        for c in self.classes_:
            idx = np.where(y_arr == c)[0]
            if len(idx) == 0:
                continue
            Xc = X_arr[idx]
            k = min(self.p.prototypes_per_class, len(idx))
            if k <= 1:
                chosen = idx[:k]
            else:
                km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(Xc)
                # pick nearest to each centroid
                d2 = ((Xc[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(-1)
                chosen_local = np.argmin(d2, axis=0)
                chosen = idx[chosen_local]

            for j in chosen:
                self.prototypes.append((dict(zip(X.columns, X.iloc[j])), str(c)))

        return self

    def _decode_label(self, text):
        # parse {"label": "..."} robustly
        try:
            start = text.find("{"); end = text.rfind("}") + 1
            obj = json.loads(text[start:end])
            return str(obj["label"]).strip()
        except Exception:
            return None

    def _predict_one_sampling(self, test_row, label_map):
        prompt = self._prompt(self.prototypes[:self.p.max_examples], test_row, label_map)
        counts = {k: 0 for k in self.class_keys}
        for _ in range(self.p.samples):
            out = self.model.invoke(prompt)
            lab = self._decode_label(out)
            if lab in counts:
                counts[lab] += 1

        # convert to probs + temperature calibration (keeps class order stable)
        probs = np.array([counts[k] for k in self.class_keys], dtype=float)
        probs = (probs + 1e-6) / (probs.sum() + 1e-6)  # smoothing
        probs = np.exp(np.log(probs) / self.T)
        probs /= probs.sum()
        return probs

    def predict(self, X):
        n_samples = len(X)
        label_map = {k: k for k in self.class_keys}

        logger.info(
            f"LLM_ICL: predicting {n_samples} samples "
            f"(backend={self.p.backend}, samples={self.p.samples})"
        )

        preds = []
        step = max(1, int(self.p.progress_every))

        for i in range(n_samples):
            row = dict(zip(X.columns, X.iloc[i]))

            if self.p.backend == "sampling":
                probs = self._predict_one_sampling(row, label_map)
            else:
                raise NotImplementedError("backend='logprobs' not implemented in this stub.")

            pred = self.classes_[int(np.argmax(probs))]
            preds.append(pred)

            # --- Progress logging: percent complete and items remaining ---
            done = i + 1
            remaining = n_samples - done
            if (done % step == 0) or (done == n_samples):
                pct = (done / n_samples * 100.0) if n_samples else 100.0
                logger.info(
                    f"LLM_ICL prediction progress: {pct:.1f}% "
                    f"({done}/{n_samples}) | remaining={remaining}"
                )

        return np.array(preds)
