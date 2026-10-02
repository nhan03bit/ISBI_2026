"""Post-prediction patient HMM / Markov-random-field prior (2026-10-01).

Replaces the in-network Markov ideas (v4 label-graph layer, v5 prior-query conditioning) with a
model that sits entirely AFTER the classifier. Plan: docs/plan/2026-10-01-markov-hmm-redesign-plan.md.

    emission    the frozen ConvNeXt (+ head) gives p_c(x) = P(y_c = 1 | x); after calibration
                (shared temperature a, per-class intercept b_c) its scaled likelihood is
                l_c(x) = logit p~_c(x) - logit pibar_c            (pibar = train prevalence)
    initial     logit pi0_c(m)   = alpha0_c + gamma0_c . phi(m)              (no earlier state)
    transition  logit pi_c(...)  = alpha_c + beta_c y'_c + kappa_c y'_c g + lambda_c g
                                   + sum_{k != c} B_ck y'_k + gamma_c . phi(m)
                y' = previous (patient, date) state, g = log1p(gap days), phi = sex / age / acuity
    fusion      z_c = logit p~_c(x) + w_h (logit q_c - logit pibar_c)   (q = transition prediction)
                z_c = logit p~_c(x) + w_0 (logit pi0_c - logit pibar_c) (no earlier state)
                w = 1 is the Bayes update; w_h, w_0 are cross-fitted on held-out patients.

History modes: "observed" -- y' is the previous state's report labels (exact); "filtered" -- y' is a
factorised (Boyen-Koller) belief from a forward pass over the patient's earlier images:
    predict  q_c = b_c sig(eta_c(1, b)) + (1 - b_c) sig(eta_c(0, b))  (exact for B = 0, else mean-field)
    update   b   = sig(logit q + w_e ebar)        ebar = mean l(x) over the state's images
Optional within-state MRF edges J (pseudo-likelihood on train states, damped mean-field, weight w_J;
an ablation -- w_J = 0 is exactly the temporal model).

All 30 per-class logistic models are fitted as one batched float64 torch model (L-BFGS, L2).
Nothing here reads PadChest metadata or labels itself -- see analysis/patient_meta.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

C = 30


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, np.float64), 1e-12, 1 - 1e-12)
    return np.log(x) - np.log1p(-x)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


# ---------------------------------------------------------------------- batched logistic regression
def fit_logistic(F: np.ndarray, Y: np.ndarray, own: np.ndarray | None = None, l2: np.ndarray | float = 0.0,
                 l2_own: np.ndarray | float = 0.0, mask: np.ndarray | None = None,
                 max_iter: int = 500) -> dict:
    """C logistic regressions sharing a design F (N, d), plus optional per-class features own (N, C, k):
        logit_nc = F_n . (W * mask)[:, c] + own_nc . V[:, c]
    loss = sum_nc BCE / N + 0.5 * sum_jc l2_j W_jc^2 + 0.5 * sum_kc l2own_k V_kc^2.
    l2 is a scalar or a (d,) vector (put 0 on the intercept). Returns {"W": (d, C), "V": (k, C)}."""
    Ft = torch.as_tensor(F, dtype=torch.float64)
    Yt = torch.as_tensor(Y, dtype=torch.float64)
    n, d = Ft.shape
    c = Yt.shape[1]
    k = 0 if own is None else own.shape[2]
    Ot = None if own is None else torch.as_tensor(own, dtype=torch.float64)
    Mt = torch.ones(d, c, dtype=torch.float64) if mask is None else torch.as_tensor(mask, dtype=torch.float64)
    L2 = torch.as_tensor(np.broadcast_to(np.asarray(l2, float), (d,)).copy(), dtype=torch.float64)[:, None]
    L2o = torch.as_tensor(np.broadcast_to(np.asarray(l2_own, float), (max(k, 1),)).copy(), dtype=torch.float64)[:k, None]
    # start from the per-class log-odds in the intercept column when there is one
    W = torch.zeros(d, c, dtype=torch.float64)
    if (F[:, 0] == 1).all():
        pbar = Yt.mean(0).clamp(1e-6, 1 - 1e-6)
        W[0] = torch.log(pbar) - torch.log1p(-pbar)
    W.requires_grad_(True)
    V = torch.zeros(k, c, dtype=torch.float64, requires_grad=True)
    params = [W, V] if k else [W]
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=max_iter, tolerance_grad=1e-9,
                            tolerance_change=1e-12, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Ft @ (W * Mt)
        if k:
            z = z + (Ot * V.T[None]).sum(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Yt, reduction="sum") / n
        loss = loss + 0.5 * (L2 * (W * Mt) ** 2).sum()
        if k:
            loss = loss + 0.5 * (L2o * V ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return {"W": (W * Mt).detach().numpy(), "V": V.detach().numpy()}


def logistic_logits(F: np.ndarray, fit: dict, own: np.ndarray | None = None) -> np.ndarray:
    z = F @ fit["W"]
    if own is not None and fit["V"].size:
        z = z + np.einsum("nck,kc->nc", own, fit["V"])
    return z


# ---------------------------------------------------------------------- prior (initial + transition)
@dataclass
class PriorSpec:
    """Which transition / initial terms are on. kind="table" is the pooled per-class 2-state chain
    (a_c = P(1|1), b_c = P(1|0), Laplace) of analysis/patient_prior.fit_chain -- the B-late anchor."""
    kind: str = "logistic"      # "table" | "logistic"
    gap: bool = False           # kappa (persistence x gap) and lambda (gap)
    meta: bool = False          # gamma . phi(m) in transition and initial
    cross: bool = False         # B: previous state of other classes
    l2: float = 1e-3            # on gap / meta weights
    l2_cross: float = 1e-2      # on B
    name: str = ""


def transition_design(prev: np.ndarray, g: np.ndarray, phi: np.ndarray, spec: PriorSpec):
    """Shared design F and per-class own features for given previous-state values `prev` (N, C)
    (0/1 labels or beliefs). Columns of F: [1, g?, phi?, prev (C)?]; own: [y'_c, y'_c g?]."""
    n = len(prev)
    cols = [np.ones((n, 1))]
    if spec.gap:
        cols.append(g[:, None])
    if spec.meta:
        cols.append(phi)
    if spec.cross:
        cols.append(prev)
    F = np.concatenate(cols, 1)
    own = [prev]
    if spec.gap:
        own.append(prev * g[:, None])
    return F, np.stack(own, -1)


def _l2_vector(spec: PriorSpec, d_phi: int, cross: bool, c: int = C) -> np.ndarray:
    v = [0.0]  # intercept
    if spec.gap:
        v.append(spec.l2)
    if spec.meta:
        v += [spec.l2] * d_phi
    if cross:
        v += [spec.l2_cross] * c
    return np.array(v)


def _cross_mask(spec: PriorSpec, d_phi: int, c: int = C) -> np.ndarray | None:
    if not spec.cross:
        return None
    d0 = 1 + int(spec.gap) + (d_phi if spec.meta else 0)
    m = np.ones((d0 + c, c))
    m[d0:] -= np.eye(c)  # own previous state enters through beta (and kappa), not B
    return m


class PriorModel:
    """Initial + transition prior P(Y_t | Y_{t-1}, m). Fit on train-fold (patient, date) states only."""

    def __init__(self, spec: PriorSpec):
        self.spec = spec

    # -- fitting
    def fit(self, prev: np.ndarray, nxt: np.ndarray, g: np.ndarray, phi: np.ndarray,
            first: np.ndarray, phi_first: np.ndarray, table: dict | None = None) -> "PriorModel":
        """prev / nxt: (P, C) 0/1 labels of consecutive train states; g, phi of the later state.
        first: (S0, C) labels of patients' first train states with phi_first."""
        s = self.spec
        self.d_phi = phi.shape[1]
        if s.kind == "table":
            assert table is not None
            self.a, self.b = np.asarray(table["a"], float), np.asarray(table["b"], float)
        else:
            F, own = transition_design(prev.astype(float), g, phi, s)
            self.trans = fit_logistic(F, nxt, own, l2=_l2_vector(s, self.d_phi, s.cross),
                                      l2_own=[0.0] + ([s.l2] if s.gap else []),
                                      mask=_cross_mask(s, self.d_phi))
        F0 = np.concatenate([np.ones((len(first), 1))] + ([phi_first] if s.meta else []), 1)
        self.init = fit_logistic(F0, first, l2=np.r_[0.0, [s.l2] * (F0.shape[1] - 1)])
        return self

    # -- prediction
    def initial_logit(self, phi: np.ndarray) -> np.ndarray:
        F0 = np.concatenate([np.ones((len(phi), 1))] + ([phi] if self.spec.meta else []), 1)
        return logistic_logits(F0, self.init)

    def predict(self, prev: np.ndarray, g: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """P(y_t,c = 1 | prev, m) for 0/1 `prev` (exact) or a factorised belief `prev` in [0, 1]
        (own state marginalised exactly, other classes by mean field)."""
        prev = np.asarray(prev, float)
        if self.spec.kind == "table":
            return prev * self.a + (1 - prev) * self.b
        F, _ = transition_design(prev, g, phi, self.spec)
        base = F @ self.trans["W"]
        out = np.zeros_like(prev)
        for s in (0.0, 1.0):
            own = np.full_like(prev, s)
            ownf = np.stack([own] + ([own * g[:, None]] if self.spec.gap else []), -1)
            eta = base + np.einsum("nck,kc->nc", ownf, self.trans["V"])
            out += (prev if s else 1 - prev) * sigmoid(eta)
        return out

    def predict_logit_observed(self, prev01: np.ndarray, g: np.ndarray, phi: np.ndarray) -> np.ndarray:
        if self.spec.kind == "table":
            return logit(self.predict(prev01, g, phi))
        F, own = transition_design(prev01.astype(float), g, phi, self.spec)
        return logistic_logits(F, self.trans, own)

    def state_dict(self) -> dict:
        d = {"spec": self.spec.__dict__, "d_phi": self.d_phi, "init": self.init}
        if self.spec.kind == "table":
            d.update(a=self.a, b=self.b)
        else:
            d["trans"] = self.trans
        return d

    @classmethod
    def from_state_dict(cls, d: dict) -> "PriorModel":
        m = cls(PriorSpec(**d["spec"]))
        m.d_phi, m.init = d["d_phi"], d["init"]
        if m.spec.kind == "table":
            m.a, m.b = d["a"], d["b"]
        else:
            m.trans = d["trans"]
        return m


# ---------------------------------------------------------------------- emission calibration
def fit_calibration(lp: np.ndarray, Y: np.ndarray, l2: float = 1e-3) -> dict:
    """logit p~ = a * logit p + b_c (shared temperature, per-class intercept with L2), by BCE."""
    x = torch.as_tensor(lp, dtype=torch.float64)
    y = torch.as_tensor(Y, dtype=torch.float64)
    a = torch.ones(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(y.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=1.0, max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a * x + b, y, reduction="sum") / len(x)
        loss = loss + 0.5 * l2 * (b ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return {"a": float(a.detach()), "b": b.detach().numpy()}


def calibrate(lp: np.ndarray, cal: dict | None) -> np.ndarray:
    return lp if cal is None else cal["a"] * lp + cal["b"]


# ---------------------------------------------------------------------- within-state MRF (ablation)
def fit_pairwise(Y: np.ndarray, l2: float = 1e-2, max_iter: int = 300) -> dict:
    """Pseudo-likelihood Ising fit on train states: logit P(y_c | y_-c) = h_c + sum_k J_ck y_k with
    J symmetric, zero diagonal."""
    y = torch.as_tensor(Y, dtype=torch.float64)
    c = y.shape[1]
    off = 1 - torch.eye(c, dtype=torch.float64)
    pbar = y.mean(0).clamp(1e-6, 1 - 1e-6)
    h = (torch.log(pbar) - torch.log1p(-pbar)).clone().requires_grad_(True)
    A = torch.zeros(c, c, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([h, A], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        J = 0.5 * (A + A.T) * off
        loss = torch.nn.functional.binary_cross_entropy_with_logits(h + y @ J, y, reduction="sum") / len(y)
        loss = loss + 0.5 * l2 * (J ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    J = (0.5 * (A + A.T) * off).detach().numpy()
    return {"h": h.detach().numpy(), "J": J}


def meanfield(z: np.ndarray, J: np.ndarray, w_J: float, iters: int = 20, damp: float = 0.5) -> np.ndarray:
    """Damped mean-field on the current state: mu <- sig(z + w_J J mu). Returns logits; w_J = 0 -> z."""
    if w_J == 0:
        return z
    mu = sigmoid(z)
    for _ in range(iters):
        mu = damp * mu + (1 - damp) * sigmoid(z + w_J * mu @ J)
    return z + w_J * mu @ J


# ---------------------------------------------------------------------- forward filtering over states
def forward_filter(prior: PriorModel, prev_idx: np.ndarray, rank: np.ndarray, g: np.ndarray,
                   phi: np.ndarray, ebar: np.ndarray, w_e: float,
                   observed: np.ndarray | None = None) -> np.ndarray:
    """Factorised belief b_s = P(y_s | images up to s) for every state s (sorted by patient, date).
    prev_idx[s] = previous state of the same patient (-1 for a first state); ebar (S, C) = mean
    calibrated scaled likelihood of the state's images (0 when the state has none). With
    `observed` (S, C) 0/1 the belief is replaced by the labels (observed-history mode)."""
    S = len(prev_idx)
    belief = np.zeros((S, C))
    first = rank == 0
    belief[first] = sigmoid(prior.initial_logit(phi[first]) + w_e * ebar[first])
    for r in range(1, int(rank.max()) + 1):
        idx = np.where(rank == r)[0]
        if not len(idx):
            continue
        pv = belief[prev_idx[idx]] if observed is None else observed[prev_idx[idx]]
        q = prior.predict(pv, g[idx], phi[idx])
        belief[idx] = sigmoid(logit(q) + w_e * ebar[idx])
    if observed is not None:
        return observed.astype(float)
    return belief


# ---------------------------------------------------------------------- fusion
@dataclass
class FusionWeights:
    w_h: float = 1.0   # history term
    w_0: float = 0.0   # initial (metadata) term for rows without an earlier state
    w_e: float = 1.0   # emission weight inside the forward filter (filtered mode)
    w_J: float = 0.0   # within-state MRF edges (ablation)


def fuse(lp_cal: np.ndarray, hist_term: np.ndarray, init_term: np.ndarray, has: np.ndarray,
         w: FusionWeights, J: np.ndarray | None = None) -> np.ndarray:
    """Fused logits. hist_term = logit q - logit pibar (rows with an earlier state),
    init_term = logit pi0(m) - logit pibar (rows without)."""
    z = lp_cal + np.where(has[:, None], w.w_h * hist_term, w.w_0 * init_term)
    if J is not None and w.w_J:
        z = meanfield(z, J, w.w_J)
    return z


@dataclass
class MarkovHMM:
    """Fitted post-prediction model: prior (train states), calibration and weights (val), optional J.
    `apply` turns an emission matrix plus the rows' prior terms into fused probabilities."""
    prior: PriorModel
    featurizer_state: dict
    pibar: np.ndarray
    cal: dict | None = None
    weights: FusionWeights = field(default_factory=FusionWeights)
    J: np.ndarray | None = None
    mode: str = "observed"

    def apply(self, probs: np.ndarray, hist_logit_q: np.ndarray, init_logit: np.ndarray,
              has: np.ndarray) -> np.ndarray:
        lp = calibrate(logit(probs), self.cal)
        lpi = logit(self.pibar)
        z = fuse(lp, hist_logit_q - lpi, init_logit - lpi, has, self.weights, self.J)
        return sigmoid(z).astype(np.float32)

    def save(self, path) -> None:
        torch.save({"prior": self.prior.state_dict(), "featurizer": self.featurizer_state,
                    "pibar": self.pibar, "cal": self.cal, "weights": self.weights.__dict__,
                    "J": self.J, "mode": self.mode}, path)

    @classmethod
    def load(cls, path) -> "MarkovHMM":
        d = torch.load(path, weights_only=False)
        return cls(PriorModel.from_state_dict(d["prior"]), d["featurizer"], d["pibar"], d["cal"],
                   FusionWeights(**d["weights"]), d["J"], d["mode"])
