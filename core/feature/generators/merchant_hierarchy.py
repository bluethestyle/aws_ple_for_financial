"""
Merchant Hierarchy Feature Generator — 27D Poincaré embeddings.

Output per row (27D): L1 Poincaré coords (4D), L2 Poincaré coords (4D),
brand SVD (8D), aggregate stats (4D), hierarchy depth (3D), spread (4D).

Ref: Ganea et al. "Hyperbolic Neural Networks" (NeurIPS 2018).
Hardware: CPU-only.  numpy + sklearn (lazy).  No torch, no GPU.
"""
from __future__ import annotations

import ast
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from core.data.dataframe import df_backend
from ..generator import AbstractFeatureGenerator, FeatureGeneratorRegistry
from .gpu_utils import _to_pandas_safe, _columns_list

logger = logging.getLogger(__name__)

_EPS = 1e-8
_DISK_CLIP = 1.0 - 1e-5   # open-disk boundary guard


# ---------------------------------------------------------------------------
# Poincaré disk math
# ---------------------------------------------------------------------------

def _lorentz_factor(x: np.ndarray) -> np.ndarray:
    """gamma = 1 / sqrt(1 - ||x||²), shape (..., 1)."""
    sq = np.clip(np.sum(x ** 2, axis=-1, keepdims=True), 0.0, _DISK_CLIP ** 2)
    return 1.0 / np.sqrt(1.0 - sq + _EPS)


def _poincare_midpoint(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Einstein midpoint on the 2D Poincaré disk."""
    if len(points) == 0:
        return np.zeros(2, dtype=np.float32)
    w = weights.astype(np.float64)
    w_sum = w.sum()
    if w_sum < _EPS:
        w = np.ones(len(points), dtype=np.float64) / len(points)
        w_sum = 1.0
    w = w / w_sum
    pts = points.astype(np.float64)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = np.where(norms >= 1.0, pts / (norms + _EPS) * _DISK_CLIP, pts)
    gamma = _lorentz_factor(pts).ravel()
    denom = np.dot(gamma, w)
    if denom < _EPS:
        return np.zeros(2, dtype=np.float32)
    mid = np.einsum("i,i,ij->j", gamma, w, pts) / denom
    mid_n = np.linalg.norm(mid)
    if mid_n >= 1.0:
        mid = mid / (mid_n + _EPS) * _DISK_CLIP
    return mid.astype(np.float32)


def _poincare_to_4d(point: np.ndarray) -> np.ndarray:
    """[x, y, radius, angle/pi] from a 2D disk point."""
    x, y = float(point[0]), float(point[1])
    return np.array([x, y, math.sqrt(x*x + y*y), math.atan2(y, x) / math.pi],
                    dtype=np.float32)


# ---------------------------------------------------------------------------
# Hierarchy loading + Poincaré positions
# ---------------------------------------------------------------------------

def _load_mcc_hierarchy(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build_lookups(
    hierarchy: Dict[str, Any],
) -> Tuple[Dict[int, str], Dict[int, str], List[str], List[str]]:
    """Build MCC→L1 and MCC→L2 lookup tables from hierarchy dict."""
    mcc_to_l1: Dict[int, str] = {}
    mcc_to_l2: Dict[int, str] = {}
    l1_names: List[str] = []
    l2_names: List[str] = []
    hier = hierarchy.get("hierarchy", hierarchy)
    for l1_name, l2_dict in hier.items():
        if not isinstance(l2_dict, dict):
            continue
        if l1_name not in l1_names:
            l1_names.append(l1_name)
        for l2_name, l2_info in l2_dict.items():
            if not isinstance(l2_info, dict):
                continue
            dotted = f"{l1_name}.{l2_name}"
            if dotted not in l2_names:
                l2_names.append(dotted)
            for code in l2_info.get("codes", []):
                c = int(code)
                mcc_to_l1[c] = l1_name
                mcc_to_l2[c] = dotted
    return mcc_to_l1, mcc_to_l2, l1_names, l2_names


def _l1_positions(l1_names: List[str], radius: float = 0.8) -> np.ndarray:
    """Evenly-spaced points on disk boundary for each L1 category. (n_l1, 2)"""
    n = max(len(l1_names), 1)
    angles = [2.0 * math.pi * i / n for i in range(n)]
    return np.array([[radius * math.cos(a), radius * math.sin(a)]
                     for a in angles], dtype=np.float32)


def _l2_positions(
    l2_names: List[str],
    l1_names: List[str],
    l1_pos: np.ndarray,
    radius: float = 0.5,
) -> np.ndarray:
    """L2 positions between parent L1 and origin. (n_l2, 2)"""
    l1_idx = {name: i for i, name in enumerate(l1_names)}
    pts = np.zeros((len(l2_names), 2), dtype=np.float32)
    scale = radius / 0.8
    for j, dotted in enumerate(l2_names):
        pts[j] = l1_pos[l1_idx.get(dotted.split(".")[0], 0)] * scale
    return pts


# ---------------------------------------------------------------------------
# Sequence parsing
# ---------------------------------------------------------------------------

def _parse_seq(raw: Any) -> List[int]:
    """Parse MCC sequence into list[int]; handles list, str, scalar, None."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [int(v) for v in raw if v is not None]
    if isinstance(raw, str):
        s = raw.strip()
        if not s or s in ("nan", "None", "[]"):
            return []
        try:
            p = ast.literal_eval(s)
            return [int(v) for v in (p if isinstance(p, (list, tuple)) else [p]) if v is not None]
        except Exception:
            return []
    try:
        v = float(raw)
        return [] if math.isnan(v) else [int(v)]
    except Exception:
        return []


def _parse_float_seq(raw: Any) -> List[float]:
    """Parse amount/day sequence into list[float]."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [float(v) for v in raw if v is not None]
    if isinstance(raw, str):
        s = raw.strip()
        if not s or s in ("nan", "None", "[]"):
            return []
        try:
            p = ast.literal_eval(s)
            return [float(v) for v in (p if isinstance(p, (list, tuple)) else [p]) if v is not None]
        except Exception:
            return []
    try:
        v = float(raw)
        return [] if math.isnan(v) else [v]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Per-customer 27D feature computation
# ---------------------------------------------------------------------------

def _gini(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    arr = np.sort(values.astype(np.float64))
    total = arr.sum()
    if total < _EPS:
        return 0.0
    n = len(arr)
    return float((2.0 * np.dot(np.arange(1, n + 1), arr)) / (n * total) - (n + 1) / n)


def _compute_customer_features(
    mcc_seq: List[int],
    amount_seq: List[float],
    day_seq: List[float],
    mcc_to_l1: Dict[int, str],
    mcc_to_l2: Dict[int, str],
    l1_pos: np.ndarray,
    l2_pos: np.ndarray,
    n_l1: int,
    n_l2: int,
    brand_embed_dim: int,
    svd_model: Any,
    l1_index: Dict[str, int],
    l2_index: Dict[str, int],
) -> np.ndarray:
    """Compute all 27 features for one customer row."""
    out = np.zeros(27, dtype=np.float32)
    n_txn = len(mcc_seq)
    if n_txn == 0:
        return out

    amounts  = np.array(amount_seq[:n_txn], dtype=np.float64) if amount_seq else np.ones(n_txn)
    days     = np.array(day_seq[:n_txn],    dtype=np.float64) if day_seq    else np.zeros(n_txn)
    max_day  = days.max() if n_txn > 0 else 0.0
    spend_w  = np.maximum(amounts, 0.0)
    recency_w = np.exp(-(max_day - days) / 180.0)
    w = spend_w * recency_w
    if w.sum() < _EPS:
        w = np.ones(n_txn, dtype=np.float64)

    l1_idx_list = [l1_index.get(mcc_to_l1.get(m, ""), -1) for m in mcc_seq]
    l2_idx_list = [l2_index.get(mcc_to_l2.get(m, ""), -1) for m in mcc_seq]

    # 1. L1 Poincaré (4D: x, y, radius, angle)
    valid_l1 = [(i, l1_idx_list[i]) for i in range(n_txn) if l1_idx_list[i] >= 0]
    if valid_l1:
        idxs = [t[1] for t in valid_l1]
        out[0:4] = _poincare_to_4d(_poincare_midpoint(l1_pos[idxs], w[[t[0] for t in valid_l1]]))

    # 2. L2 Poincaré (4D: x, y, radius, angle)
    valid_l2 = [(i, l2_idx_list[i]) for i in range(n_txn) if l2_idx_list[i] >= 0]
    if valid_l2:
        idxs = [t[1] for t in valid_l2]
        out[4:8] = _poincare_to_4d(_poincare_midpoint(l2_pos[idxs], w[[t[0] for t in valid_l2]]))

    # 3. Brand SVD on L2 histogram (8D)
    l2_hist = np.zeros(n_l2, dtype=np.float32)
    for i, l2i in enumerate(l2_idx_list):
        if l2i >= 0:
            l2_hist[l2i] += float(w[i])
    hist_sum = l2_hist.sum()
    if hist_sum > _EPS:
        l2_hist /= hist_sum
    if svd_model is not None:
        try:
            svd_out = svd_model.transform(l2_hist.reshape(1, -1))[0]
            fill = min(brand_embed_dim, len(svd_out))
            svd_n = np.linalg.norm(svd_out[:fill])
            out[8:8 + fill] = (svd_out[:fill] / (svd_n + _EPS)).astype(np.float32)
        except Exception as exc:
            logger.debug("SVD transform failed: %s", exc)
    else:
        fill = min(brand_embed_dim, n_l2)
        out[8:8 + fill] = l2_hist[:fill]

    # 4. Aggregate stats (4D: count, loyalty, recency, gini)
    unique_mccs = len(set(mcc_seq))
    out[16] = float(np.clip(math.log1p(unique_mccs) / math.log1p(500), 0, 1))
    if n_txn > 1:
        seen: set = set()
        revisit = 0
        for m in mcc_seq:
            if m in seen:
                revisit += 1
            seen.add(m)
        out[17] = float(np.clip(revisit / (n_txn - 1), 0, 1))
    if max_day > 0:
        out[18] = float(np.clip(1.0 - days[-1] / (max_day + _EPS), 0, 1))
    out[19] = float(np.clip(_gini(spend_w), 0, 1))

    # 5. Hierarchy depth (3D: l1_radius, l2_radius, coherence)
    l1_r = float(np.linalg.norm(out[0:2]))
    l2_r = float(np.linalg.norm(out[4:6]))
    out[20] = l1_r
    out[21] = l2_r
    coherence = float(np.dot(out[0:2] / (l1_r + _EPS), out[4:6] / (l2_r + _EPS)))
    out[22] = float(np.clip((coherence + 1.0) / 2.0, 0, 1))

    # 6. Hierarchy spread (4D: per-quarter spend-weighted radius)
    n_l1_safe = max(n_l1, 1)
    qsize = max(n_l1_safe // 4, 1)
    for q in range(4):
        s = q * qsize
        e = s + qsize if q < 3 else n_l1_safe
        q_mask = [i for i, li in enumerate(l1_idx_list) if li >= 0 and s <= li < e]
        if q_mask:
            q_pts = l1_pos[[l1_idx_list[i] for i in q_mask]]
            q_w   = w[q_mask]
            q_mid = _poincare_midpoint(q_pts, q_w)
            out[23 + q] = float(np.linalg.norm(q_mid) * q_w.sum() / (w.sum() + _EPS))

    return out


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

@FeatureGeneratorRegistry.register(
    "merchant_hierarchy",
    description="MCC hierarchy Poincaré embeddings + merchant statistics (27D).",
    tags=["merchant", "hierarchy", "mcc", "poincare", "hgcn"],
)
class MerchantHierarchyGenerator(AbstractFeatureGenerator):
    """MCC hierarchy Poincaré embeddings for the HGCN expert (27D output).

    Config keys (all config-driven, no hardcoding):
      mcc_hierarchy_path : str   path to mcc_hierarchy.yaml
      mcc_seq_col        : str   default "txn_mcc_seq"
      amount_seq_col     : str   default "txn_amount_seq"
      day_seq_col        : str   default "txn_day_offset_seq"
      brand_embed_dim    : int   default 8
      l1_radius          : float default 0.8
      l2_radius          : float default 0.5
    """

    supports_gpu: bool = False
    required_libraries: List[str] = ["sklearn"]
    _OUTPUT_DIM: int = 27

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        prefix: str = "mh",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._cfg: Dict[str, Any] = config or {}
        self.prefix = prefix
        self._mcc_col:    str = self._cfg.get("mcc_seq_col",    self._cfg.get("mcc_seq_column", "txn_mcc_seq"))
        self._amount_col: str = self._cfg.get("amount_seq_col", self._cfg.get("amount_seq_column", "txn_amount_seq"))
        self._day_col:    str = self._cfg.get("day_seq_col",    self._cfg.get("day_offset_seq_column", "txn_day_offset_seq"))
        self._brand_dim:  int = int(self._cfg.get("brand_embed_dim", 8))
        self._truncate_last: int = int(self._cfg.get("truncate_seq_last", 0))
        # Fitted state
        self._mcc_to_l1:    Dict[int, str] = {}
        self._mcc_to_l2:    Dict[int, str] = {}
        self._l1_names:     List[str] = []
        self._l2_names:     List[str] = []
        self._l1_pos:       Optional[np.ndarray] = None
        self._l2_pos:       Optional[np.ndarray] = None
        self._l1_index:     Dict[str, int] = {}
        self._l2_index:     Dict[str, int] = {}
        self._svd_model:    Any = None

    @property
    def output_dim(self) -> int:
        return self._OUTPUT_DIM

    @property
    def output_columns(self) -> List[str]:
        p = self.prefix
        cols = [
            f"{p}_l1_poincare_x", f"{p}_l1_poincare_y",
            f"{p}_l1_poincare_r", f"{p}_l1_poincare_angle",
            f"{p}_l2_poincare_x", f"{p}_l2_poincare_y",
            f"{p}_l2_poincare_r", f"{p}_l2_poincare_angle",
        ]
        cols += [f"{p}_brand_svd_{i}" for i in range(self._brand_dim)]
        cols += [
            f"{p}_merchant_count", f"{p}_merchant_loyalty",
            f"{p}_merchant_recency", f"{p}_merchant_spend_gini",
            f"{p}_l1_depth", f"{p}_l2_depth", f"{p}_l1l2_coherence",
        ]
        cols += [f"{p}_spread_q{q}" for q in range(4)]
        return cols   # 4+4+8+4+3+4 = 27

    def fit(self, df: Any, **context: Any) -> "MerchantHierarchyGenerator":
        """Load MCC hierarchy YAML, compute Poincaré positions, fit SVD."""
        yaml_path: str = self._cfg.get("mcc_hierarchy_path", "configs/mcc_hierarchy.yaml")
        try:
            raw_hier = _load_mcc_hierarchy(yaml_path)
        except Exception as exc:
            logger.warning("Cannot load MCC hierarchy '%s': %s — using empty.", yaml_path, exc)
            raw_hier = {}

        (self._mcc_to_l1, self._mcc_to_l2,
         self._l1_names,  self._l2_names) = _build_lookups(raw_hier)

        n_l2 = max(len(self._l2_names), 1)
        l1_r = float(self._cfg.get("l1_radius", 0.8))
        l2_r = float(self._cfg.get("l2_radius", 0.5))
        self._l1_pos   = _l1_positions(self._l1_names, l1_r)
        self._l2_pos   = _l2_positions(self._l2_names, self._l1_names, self._l1_pos, l2_r)
        self._l1_index = {n: i for i, n in enumerate(self._l1_names)}
        self._l2_index = {n: i for i, n in enumerate(self._l2_names)}

        logger.info(
            "MerchantHierarchyGenerator fitted: L1=%d, L2=%d, MCC codes=%d",
            len(self._l1_names), len(self._l2_names), len(self._mcc_to_l1),
        )

        # Fit SVD on training L2 histograms
        pdf = _to_pandas_safe(df)
        n_rows = len(pdf)
        self._svd_model = None
        if self._mcc_col in pdf.columns and n_l2 > 1:
            try:
                hists = np.zeros((n_rows, n_l2), dtype=np.float32)
                for ri, raw in enumerate(pdf[self._mcc_col]):
                    for mcc in _parse_seq(raw):
                        l2i = self._l2_index.get(self._mcc_to_l2.get(mcc, ""), -1)
                        if l2i >= 0:
                            hists[ri, l2i] += 1.0
                    rs = hists[ri].sum()
                    if rs > _EPS:
                        hists[ri] /= rs
                from sklearn.decomposition import TruncatedSVD
                nc = min(self._brand_dim, n_l2 - 1, n_rows - 1)
                if nc >= 1:
                    self._svd_model = TruncatedSVD(n_components=nc, random_state=42)
                    self._svd_model.fit(hists)
                    logger.info("SVD fitted: n_components=%d", nc)
            except Exception as exc:
                logger.warning("SVD fitting failed (histogram fallback): %s", exc, exc_info=True)
        else:
            logger.debug("MCC column '%s' absent or n_l2==1; SVD skipped.", self._mcc_col)

        self._fitted = True
        return self

    def generate(self, df: Any, **context: Any) -> Any:
        """Generate 27D Poincaré merchant hierarchy features."""
        if not self._fitted:
            raise RuntimeError("MerchantHierarchyGenerator must be fitted before generate().")

        pdf = _to_pandas_safe(df)
        n_rows = len(pdf)
        n_l1 = max(len(self._l1_names), 1)
        n_l2 = max(len(self._l2_names), 1)
        assert self._l1_pos is not None and self._l2_pos is not None

        present = set(_columns_list(pdf))
        mcc_col    = self._mcc_col    if self._mcc_col    in present else None
        amount_col = self._amount_col if self._amount_col in present else None
        day_col    = self._day_col    if self._day_col    in present else None

        trunc = self._truncate_last  # drop last N elements to prevent label leakage

        out_matrix = np.zeros((n_rows, self._OUTPUT_DIM), dtype=np.float32)
        for ri in range(n_rows):
            mcc_seq    = _parse_seq(pdf[mcc_col].iat[ri])          if mcc_col    else []
            amount_seq = _parse_float_seq(pdf[amount_col].iat[ri]) if amount_col else []
            day_seq    = _parse_float_seq(pdf[day_col].iat[ri])    if day_col    else []
            # Truncate sequences to prevent next_mcc / top_mcc_shift leakage
            if trunc > 0:
                mcc_seq    = mcc_seq[:-trunc]    if len(mcc_seq)    > trunc else []
                amount_seq = amount_seq[:-trunc] if len(amount_seq) > trunc else []
                day_seq    = day_seq[:-trunc]    if len(day_seq)    > trunc else []
            out_matrix[ri] = _compute_customer_features(
                mcc_seq=mcc_seq, amount_seq=amount_seq, day_seq=day_seq,
                mcc_to_l1=self._mcc_to_l1, mcc_to_l2=self._mcc_to_l2,
                l1_pos=self._l1_pos, l2_pos=self._l2_pos,
                n_l1=n_l1, n_l2=n_l2,
                brand_embed_dim=self._brand_dim,
                svd_model=self._svd_model,
                l1_index=self._l1_index, l2_index=self._l2_index,
            )

        col_names = self.output_columns
        result: Dict[str, np.ndarray] = {col_names[j]: out_matrix[:, j]
                                         for j in range(self._OUTPUT_DIM)}
        return df_backend.from_dict(result, index=pdf.index)
