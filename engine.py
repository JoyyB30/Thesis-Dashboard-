"""Live scoring engine for the Early Warning dashboard (Level 2).

It re-implements, line by line, the feature code of the thesis notebook and applies the FROZEN autoencoder and threshold.
Nothing is retrained. numpy only (no torch needed): the weights are in frozen/autoencoder_weights.npz.
"""
import hashlib, json, os, re, threading, time
import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://api.etherscan.io/v2/api"
STEP_H = 1
PAGE = 1000
MAX_TX_NORMAL = 30_000        # the study dropped ordinary wallets with a longer history
MAX_TX_ATTACK = 150_000
CAP_SENTINEL = 1e9
S_CB = ["920f5c84", "1b11d0ff", "10d1e85c", "e9cbafb0", "f04f2707", "23e30c8b"]
S_IF = ["022c0d9f", "38ed1739", "8803dbee", "0902f1ac", "414bf389", "3850c7bd", "3df02124",
        "a0712d68", "c5ebeaec", "db006a75", "852a12e3", "f5e3c462",
        "e8eda9df", "a415bcad", "ab9c4b5d", "5c38449e"]
P_LIST = [a.lower() for a in [
    "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D", "0xE592427A0AEce92De3Edee1F18E0157C05861564",
    "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45", "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",
    "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2", "0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B"]]
P_INDEX = {a: i for i, a in enumerate(P_LIST)}
FEATURE_NAMES = ["Wallet age (days)", "Funding burst", "Failed share", "Contracts deployed",
                 "Flash-loan callback", "DeFi selectors"]


class Rejected(Exception):
    """The wallet is outside the study's rules; the message says why."""


class EtherscanError(Exception):
    pass


# ---------------------------------------------------------------- frozen model
class Frozen:
    def __init__(self, folder=None):
        folder = folder or os.path.join(HERE, "frozen")
        self.s = json.load(open(os.path.join(folder, "frozen_settings.json")))
        w = np.load(os.path.join(folder, "autoencoder_weights.npz"))
        self.W = {k: w[k].astype(np.float32) for k in w.files}
        self.tau = float(self.s["tau_main"]); self.tau_ref = float(self.s["tau_ref"])
        self.cap = float(self.s["f2_cap"]); self.mu = np.array(self.s["mu"]); self.sd = np.array(self.s["sd"])
        self.sha_model = hashlib.sha256(open(os.path.join(folder, "autoencoder.pt"), "rb").read()).hexdigest()[:16] \
            if os.path.exists(os.path.join(folder, "autoencoder.pt")) else "n/a"

    def prep(self, X):
        X = np.array(X, dtype=np.float64, copy=True)
        X[:, 1] = np.minimum(X[:, 1], self.cap)
        X[:, [0, 1, 3, 5]] = np.log1p(X[:, [0, 1, 3, 5]])
        return (X - self.mu) / self.sd

    def feature_errors(self, Z):
        """Squared reconstruction error per row and per feature (shape n x 6). The score of a row is the mean over the six."""
        W = self.W; x = Z.astype(np.float32)
        h = np.maximum(x @ W["enc_0_weight"].T + W["enc_0_bias"], 0)
        z = h @ W["enc_2_weight"].T + W["enc_2_bias"]
        h = np.maximum(z @ W["dec_0_weight"].T + W["dec_0_bias"], 0)
        o = h @ W["dec_2_weight"].T + W["dec_2_bias"]
        return ((o - x) ** 2).astype(np.float64)

    def errors(self, Z):
        return self.feature_errors(Z).mean(1)

    def drivers(self, raw_rows, scores=None):
        """Share of each feature in the reconstruction error at the highest-scoring hour. Returns (shares, row index)."""
        E_ = self.feature_errors(self.prep(np.asarray(raw_rows, float)))
        i = int(np.argmax(E_.mean(1) if scores is None else np.asarray(scores)))
        e = E_[i]; t = e.sum()
        return (e / t if t > 0 else np.zeros_like(e)), i


# ---------------------------------------------------------------- features (same code as the notebook)
def has_selector(input_hex, sel):
    h = input_hex.lower(); h = h[2:] if h.startswith("0x") else h
    needle = "63" + sel; i = h.find(needle)
    while i != -1:
        if i % 2 == 0: return True
        i = h.find(needle, i + 1)
    return False


def compact(addr, rows):
    a = addr.lower()
    o = dict(ts=[], hash=[], sent=[], recv=[], val=[], err=[], creat=[], pidx=[], cb=[], ifs=[], dep=[])
    for x in rows:
        to = (x.get("to") or "").lower(); fr = x["from"].lower(); sent = fr == a
        v = int(x["value"]); err = 1 if x["isError"] == "1" else 0
        creat = sent and to == ""
        inp = x.get("input", "")
        dep = creat and err == 0 and inp not in ("", "0x")
        o["ts"].append(int(x["timeStamp"])); o["hash"].append(x["hash"].lower())
        o["sent"].append(int(sent)); o["recv"].append(int(to == a and fr != a and v > 0))
        o["val"].append(float(v)); o["err"].append(err); o["creat"].append(int(creat))
        o["pidx"].append(P_INDEX.get(to, -1) if sent else -1)
        o["cb"].append(sum(1 << j for j, s in enumerate(S_CB) if dep and has_selector(inp, s)))
        o["ifs"].append(sum(1 << j for j, s in enumerate(S_IF) if dep and has_selector(inp, s)))
        o["dep"].append(int(dep))
    return o


def _cnt(arr, t, W):
    return np.searchsorted(arr, t, "left") - np.searchsorted(arr, t - W, "right")


def horizon_features(h, anchor_t, anchor_hash, window_days=1, step_h=STEP_H):
    """Feature matrix (H x 8) for prediction times anchor - k hours, k = 1..H. Uses ONLY transactions before each time."""
    W = window_days * 86400
    H = int(window_days * 24 / step_h)
    t = anchor_t - np.arange(1, H + 1) * 3600 * step_h
    ts = np.array(h["ts"], dtype=np.int64)
    keep = (ts < anchor_t) & (np.array(h["hash"]) != (anchor_hash or "").lower()) if len(ts) else np.zeros(0, bool)
    g = lambda k, dtp=np.int64: np.array(h[k], dtype=dtp)[keep] if len(ts) else np.zeros(0, dtp)
    ts = ts[keep]
    sent, recv, val, err, creat, pidx, cb, ifs, dep = (g("sent"), g("recv"), g("val", np.float64), g("err"), g("creat"),
                                                       g("pidx"), g("cb"), g("ifs"), g("dep"))
    X = np.zeros((H, 8))
    if len(ts):
        first = ts.min(); X[:, 0] = np.where(first < t, (t - first) / 86400.0, 0.0)
    m = recv == 1
    if m.any():
        o = np.argsort(ts[m], kind="stable"); rt = ts[m][o]; rv = val[m][o]
        cs = np.concatenate([[0.0], np.cumsum(rv)])
        lo = np.searchsorted(rt, t - W, "right"); hi = np.searchsorted(rt, t, "left")
        prior = cs[lo]
        vmax = np.array([rv[a:b].max() if b > a else 0.0 for a, b in zip(lo, hi)])
        X[:, 1] = np.where(vmax == 0, 0.0, np.where(prior == 0, CAP_SENTINEL, vmax / np.where(prior == 0, 1, prior)))
    ms = sent == 1
    if ms.any():
        o = np.argsort(ts[ms], kind="stable"); st = ts[ms][o]; se = err[ms][o]
        cf = np.concatenate([[0], np.cumsum(se)])
        lo = np.searchsorted(st, t - W, "right"); hi = np.searchsorted(st, t, "left")
        n = hi - lo; fl = cf[hi] - cf[lo]
        X[:, 2] = np.where(n > 0, fl / np.maximum(n, 1), 0.0)
        mc = ms & (creat == 1)
        if mc.any(): X[:, 3] = _cnt(np.sort(ts[mc]), t, W)
    md = (sent == 1) & (dep == 1)
    if md.any():
        dts = ts[md]; dcb = cb[md]; dif = ifs[md]
        e5 = np.inf
        for j in range(len(S_CB)):
            s = dts[(dcb >> j) & 1 == 1]
            if len(s): e5 = min(e5, s.min())
        X[:, 4] = (e5 < t).astype(float)
        for j in range(len(S_IF)):
            s = dts[(dif >> j) & 1 == 1]
            if len(s): X[:, 5] += (s.min() < t)
    return X


def empty_window(h, anchor_t, anchor_hash, window_days=1):
    """True if the wallet sent nothing in the window that ends at the anchor (such a wallet cannot be scored)."""
    ts = np.array(h["ts"], dtype=np.int64); sent = np.array(h["sent"])
    if not len(ts): return True
    hs = np.array(h["hash"])
    m = (sent == 1) & (ts < anchor_t) & (ts > anchor_t - window_days * 86400) & (hs != (anchor_hash or "").lower())
    return not m.any()


# ---------------------------------------------------------------- scoring
def horizon_result(h, anchor_t, anchor_hash, model):
    """Score one 24-hour horizon, exactly as in the study. A wallet that sent nothing in the window is still
    scored (the study scored all test wallets); `quiet` marks that case."""
    quiet = empty_window(h, anchor_t, anchor_hash, 1)
    X = horizon_features(h, anchor_t, anchor_hash, 1)[:, :6]
    sc = model.errors(model.prep(X))
    al = np.where(sc >= model.tau)[0]
    return dict(features=X, scores=sc, hscore=float(sc.max()), alert=bool(len(al)),
                n_alert=int(len(al)), lead_h=int(al.max() + 1) if len(al) else None,
                persistent=bool(len(al) == len(sc)), quiet=bool(quiet))


def trajectory(h, anchor_t, anchor_hash, model, days=14, every_h=6):
    """Horizon score at earlier reference times (each uses only the history before it)."""
    rows = []
    for j in range(int(days * 24 / every_h), -1, -1):
        t = anchor_t - j * every_h * 3600
        r = horizon_result(h, t, anchor_hash if j == 0 else None, model)
        rows.append(dict(hours_before=j * every_h, hscore=None if r is None else r["hscore"],
                         alert=None if r is None else r["alert"]))
    return rows


# ---------------------------------------------------------------- Etherscan
class Client:
    def __init__(self, key, min_gap=0.22):
        if not key: raise EtherscanError("No Etherscan key found. Add ETHERSCAN_KEY to the Streamlit secrets.")
        self.key, self.gap, self.calls = key, min_gap, 0
        self._lock = threading.Lock(); self._next = 0.0

    def _wait(self):
        with self._lock:
            now = time.monotonic(); w = max(0.0, self._next - now); self._next = max(now, self._next) + self.gap
        if w: time.sleep(w)

    def call(self, **p):
        p.update(chainid=1, apikey=self.key); last = ""
        for attempt in range(6):
            self._wait(); self.calls += 1
            try:
                r = requests.get(URL, params=p, timeout=40).json()
            except Exception as e:
                last = "network error"; time.sleep(2 + attempt); continue
            res = r.get("result")
            if r.get("status") == "0" and not isinstance(res, list):
                msg = (str(res) + " " + str(r.get("message"))).lower()
                if "no transactions found" in msg or "no records found" in msg: return r
                last = re.sub(r"apikey=\S+", "", msg)[:80]
                time.sleep(2 + attempt); continue
            return r
        raise EtherscanError("Etherscan call failed after 6 tries: " + last)

    def block_at(self, ts):
        r = self.call(module="block", action="getblocknobytime", timestamp=int(ts), closest="before")
        return int(r["result"])

    def block_time(self, blk):
        b = self.call(module="proxy", action="eth_getBlockByNumber", tag=hex(blk), boolean="false").get("result")
        return int(b["timestamp"], 16)

    def is_contract(self, addr):
        code = self.call(module="proxy", action="eth_getCode", address=addr, tag="latest").get("result")
        # an ordinary wallet that delegates to a contract (EIP-7702, code 0xef0100 + address) is still a wallet, as in the study
        return code not in ("0x", None) and not str(code).lower().startswith("0xef0100")

    def history(self, addr, end_block, max_tx):
        out, start, truncated = {}, 0, False
        while True:
            r = self.call(module="account", action="txlist", address=addr, startblock=start, endblock=end_block,
                          page=1, offset=PAGE, sort="asc").get("result")
            if not isinstance(r, list) or not r: break
            for x in r: out[x["hash"]] = x
            if len(out) >= max_tx: truncated = True; break
            if len(r) < PAGE: break
            nb = int(r[-1]["blockNumber"])
            if nb == start: truncated = True; break
            start = nb
        h = compact(addr, list(out.values())); h["end_block"] = end_block; h["truncated"] = truncated
        return h


ADDR_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


def check_wallet(addr, as_of_ts, client, model, anchor_hash=None, anchor_block=None, allow_heavy=False, progress=None):
    """Full live pipeline for one wallet. Raises Rejected with a short reason if it is outside the study's rules."""
    say = progress or (lambda s: None)
    addr = addr.strip()
    if not ADDR_RE.match(addr): raise Rejected("This is not a valid Ethereum address (0x + 40 hex characters).")
    say("Checking the wallet type")
    if client.is_contract(addr): raise Rejected("This address is a contract. The study scores ordinary wallets only.")
    say("Finding the block at the reference time")
    end_block = anchor_block if anchor_block else client.block_at(as_of_ts)
    say("Downloading the public transaction history")
    h = client.history(addr, end_block, MAX_TX_ATTACK if allow_heavy else MAX_TX_NORMAL)
    if h["truncated"] and not allow_heavy:
        raise Rejected(f"Very long history (more than {MAX_TX_NORMAL:,} transactions). The study dropped such ordinary wallets.")
    if not h["ts"]: raise Rejected("No transactions found for this address before the reference time.")
    say("Scoring with the frozen model")
    res = horizon_result(h, as_of_ts, anchor_hash, model)
    out = dict(address=addr, as_of_ts=int(as_of_ts), end_block=end_block, n_tx=len(h["ts"]), calls=client.calls, hist=h, result=res)
    out["trajectory"] = trajectory(h, as_of_ts, anchor_hash, model)
    return out


# ---------------------------------------------------------------- live feed (Level 2)
THESIS_TITLE = ("Early Warning for DeFi Exploits: Detecting Anomalous On-Chain Wallet Behavior Before Attacks "
                "Using Semi-Supervised Anomaly Detection, Label-Validated")
PROTOCOLS = {  # name -> contract that users send transactions to (all in P_LIST)
    "Uniswap V2": "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",
    "Uniswap V3": "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",
    "Aave V3": "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2",
    "Aave V2": "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9",
    "Compound": "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b",
}
LABELS = ["Changed behaviour", "New wallet", "Unusual from the start", "Normal", "Not scored"]


def alert_label(res, first_ts=None, anchor_t=None):
    """Three alert kinds. res = horizon_result(...) or None.
    Unusual from the start: above the threshold in all 24 checks.
    New wallet: alert starts inside the window and the wallet's history also starts inside it.
    Changed behaviour: alert starts inside the window in a wallet that already existed before it."""
    if res is None: return "Not scored"
    if not res["alert"]: return "Normal"
    if res["persistent"]: return "Unusual from the start"
    if first_ts is not None and anchor_t is not None:
        born_inside = first_ts > anchor_t - 86400
    else:
        born_inside = bool(res["features"][-1, 0] == 0)  # row 24 (24 h before) had no history
    return "New wallet" if born_inside else "Changed behaviour"


def recent_senders(client, protocol_addr, lookback_s=1800, limit=40, now_ts=None):
    """Distinct wallets that sent a transaction to the protocol in the last lookback_s seconds (newest first)."""
    r = client.call(module="account", action="txlist", address=protocol_addr, startblock=0, endblock=99999999,
                    page=1, offset=500, sort="desc").get("result")
    now_ts = now_ts or int(time.time()); seen, out = set(), []
    for x in r if isinstance(r, list) else []:
        a = x.get("from", "").lower(); t = int(x.get("timeStamp", 0))
        if t < now_ts - lookback_s: break
        if a and a not in seen and x.get("to", "").lower() == protocol_addr.lower():
            seen.add(a); out.append((a, t))
        if len(out) >= limit: break
    return out


def feed_score(addr, now_ts, end_block, client, model):
    """Score one feed wallet at now_ts. Returns a dict with label, score, age, hourly flags and driver."""
    base = dict(address=addr, label="Not scored", score=None, age_d=None, flags=[], driver=None, reason="")
    try:
        if client.is_contract(addr): return dict(base, reason="contract")
        h = client.history(addr, end_block, MAX_TX_NORMAL)
        if h["truncated"]: return dict(base, reason="over 30,000 transactions")
        if not h["ts"]: return dict(base, reason="no history")
        res = horizon_result(h, now_ts, None, model)
        first = int(min(h["ts"]))
        sh, hi = model.drivers(res["features"], res["scores"])
        return dict(base, label=alert_label(res, first, now_ts), score=res["hscore"], age_d=(now_ts - first) / 86400,
                    flags=[bool(s >= model.tau) for s in res["scores"][::-1]], driver=int(np.argmax(sh)), res=res, reason="")
    except Rejected as e:
        return dict(base, reason=str(e))
