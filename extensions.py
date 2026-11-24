"""Extensions to the thesis, computed from the saved test results and the frozen model only.
Nothing here changes a stored score, the threshold or any thesis number.

1. Behaviour types   - what the 19 detected exploit wallets looked like before the attack
2. False warnings    - how the 15 flagged normal wallets differ from the 19 detected attackers
3. Evasion what-if   - how many features an attacker would need to change to fall below the threshold
   Contract check    - a second label on every alert: does the wallet own an attack-like contract?

Run `python extensions.py` to write the tables to data/ext_*.csv.
"""
import itertools, os
import numpy as np, pandas as pd

FCOLS = ["f1", "f2", "f3", "f4", "f5", "f6"]
FNAMES = ["Wallet age", "Funding burst", "Failed share", "Contracts deployed", "Flash-loan callback", "DeFi selectors"]

TYPES = ["Fresh wallet building contracts", "Older wallet with a flash-loan contract",
         "Older wallet with DeFi contracts", "Other unusual activity"]
TYPE_TEXT = {
    TYPES[0]: "Less than a week old and already built or owns a contract.",
    TYPES[1]: "A week or older, and owns a contract that can receive a flash loan.",
    TYPES[2]: "A week or older, owns contracts with DeFi functions, but none can receive a flash loan.",
    TYPES[3]: "No DeFi contract: plain contracts or an unusual funding or failure pattern.",
}
CONTRACT = ["Attack-like contract", "Ordinary contract", "No contract"]
CONTRACT_TEXT = {
    CONTRACT[0]: "Owns a contract that can receive a flash loan, the pattern most exploit contracts share.",
    CONTRACT[1]: "Built contracts, but none can receive a flash loan.",
    CONTRACT[2]: "Has not built a contract.",
}


def behaviour_type(x):
    """x = the six raw features at the wallet's highest-scoring check."""
    age, f4, f5, f6 = float(x[0]), float(x[3]), float(x[4]), float(x[5])
    has_contract = f4 > 0 or f6 > 0 or f5 > 0
    if age < 7 and has_contract: return TYPES[0]
    if f5 > 0: return TYPES[1]
    if f6 > 0: return TYPES[2]
    return TYPES[3]


def contract_check(x):
    if float(x[4]) > 0: return CONTRACT[0]
    if float(x[3]) > 0 or float(x[5]) > 0: return CONTRACT[1]
    return CONTRACT[2]


def peak(hw):
    """Row of a wallet's 24 hourly checks with the highest score."""
    return hw.loc[hw.score.idxmax()]


def profile(M, H, wids, group):
    """One row per wallet: features and score shares at the peak check, type and contract label."""
    rows = []
    for w in wids:
        hw = H[H.wid == w]
        r = peak(hw); x = r[FCOLS].values.astype(float)
        fe = M.feature_errors(M.prep(x[None]))[0]; sh = fe / fe.sum() if fe.sum() > 0 else np.zeros(6)
        rows.append(dict(wid=w, group=group, score=float(hw.score.max()), **dict(zip(FCOLS, x)),
                         **{f"share_{c}": float(s) for c, s in zip(FCOLS, sh)},
                         driver=FNAMES[int(np.argmax(sh))], type=behaviour_type(x), contract=contract_check(x)))
    return pd.DataFrame(rows)


def normal_reference(H, att_wids):
    """Median of each feature over normal test wallets, one hour before their reference time."""
    h1 = H[(~H.wid.isin(att_wids)) & (H.hours_before == 1)]
    return h1[FCOLS].median().values.astype(float)


def rescore(M, hw, change, ref):
    """Highest hourly score after setting the features in `change` (indices) to the normal median in every check."""
    X = hw.sort_values("hours_before")[FCOLS].values.astype(float).copy()
    for j in change: X[:, j] = ref[j]
    return float(M.errors(M.prep(X)).max())


def evasion(M, H, wids, ref, max_k=3):
    """For each detected wallet: the fewest feature changes that bring it below the threshold."""
    rows = []
    for w in wids:
        hw = H[H.wid == w]
        found, k_min, best_score = [], None, None
        for k in range(1, max_k + 1):
            for combo in itertools.combinations(range(6), k):
                s = rescore(M, hw, combo, ref)
                if s < M.tau: found.append((combo, s))
            if found: k_min = k; break
        if found:
            combo, best_score = min(found, key=lambda t: t[1])
            how = " or ".join(" + ".join(FNAMES[j] for j in c) for c, _ in found[:3]) + (" …" if len(found) > 3 else "")
        else:
            how = f"not hidden by changing up to {max_k} features"
        rows.append(dict(wid=w, score=float(hw.score.max()), changes_needed=k_min if k_min else f">{max_k}",
                         ways=len(found), example=how, score_after=best_score))
    return pd.DataFrame(rows)


def all_peaks(H, att, nor):
    """Every test wallet (36 exploit + 2,000 normal): features at its highest-scoring check, contract label, alert."""
    pk = H.loc[H.groupby("wid").score.idxmax()].set_index("wid")[FCOLS]
    both = pd.concat([att[["wid", "alert"]].assign(group="Attacker"), nor[["wid", "alert"]].assign(group="Normal")], ignore_index=True)
    d = both.join(pk, on="wid")
    d["alert"] = d.alert.astype(bool)
    d["contract"] = [contract_check(r) for r in d[FCOLS].values]
    return d


def compute(M, H, att, nor):
    det = att[att.alert.astype(bool)]; fa = nor[nor.alert.astype(bool)]
    P = pd.concat([profile(M, H, det.wid, "Detected attacker"), profile(M, H, fa.wid, "False warning")], ignore_index=True)
    P = P.merge(att[["wid", "case_id"]], on="wid", how="left")
    ref = normal_reference(H, att.wid)
    EV = evasion(M, H, det.wid, ref).merge(att[["wid", "case_id"]], on="wid", how="left")
    EV = EV.merge(P[["wid", "type"]], on="wid", how="left")
    return P, EV, ref


if __name__ == "__main__":
    import engine as E
    here = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(here, "data")
    M = E.Frozen(); H = pd.read_csv(os.path.join(D, "wallet_hours.csv"))
    att = pd.read_csv(os.path.join(D, "test_attack_horizons.csv")); nor = pd.read_csv(os.path.join(D, "test_normal_horizons.csv"))
    P, EV, ref = compute(M, H, att, nor)
    P.to_csv(os.path.join(D, "ext_profiles.csv"), index=False); EV.to_csv(os.path.join(D, "ext_evasion.csv"), index=False)
    pd.set_option("display.width", 220)
    print("normal median", ref)
    print(pd.crosstab(P.type, P.group)); print(pd.crosstab(P.contract, P.group))
    print(EV[["case_id", "type", "score", "changes_needed", "ways", "example", "score_after"]].to_string())
