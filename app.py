"""Early Warning for DeFi Exploits - research dashboard (design v5.5).
Detection Overview, Wallet Analysis, Behaviour Analysis and Model & Limits replay the stored test results.
Live Monitoring scores wallets that just used a DeFi protocol, with the same frozen model.
Run:  streamlit run app.py     (Live Monitoring needs ETHERSCAN_KEY in .streamlit/secrets.toml or as an environment variable)"""
import datetime as dt, json, os, time
import numpy as np, pandas as pd, altair as alt, streamlit as st
import engine as E
import extensions as X

D = os.path.join(E.HERE, "data")
st.set_page_config(page_title="DeFi Early Warning", page_icon=":material/shield:", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------- tokens (validated for colour-blind separation)
INK, MUTED, MUTED2, LINE, BG, CARD = "#15191D", "#4B545B", "#6B747B", "#E3E6E8", "#F3F5F8", "#FFFFFF"
NAVY, NAVY2, NAVY3 = "#0E1A2B", "#1C2C44", "#24364F"
GREY, GREY2, TRACK = "#9AA3AA", "#C5CCD1", "#EEF0F1"
ST = {  # label -> (mark colour, text colour, tint)
    "Unusual from the start": ("#E4572E", "#B23A16", "#FCE5DC"),
    "New wallet": ("#D9A21E", "#6F5205", "#FAEFCF"),
    "Changed behaviour": ("#B8389A", "#8E2477", "#F8E1F1"),
    "Normal": ("#2F6FDB", "#1F55B8", "#E4ECFB"),
    "Not scored": ("#9AA3AA", "#4B545B", "#EEF0F1"),
}
BLUE = ST["Normal"][0]
FSHORT = ["Wallet age", "Funding burst", "Failed share", "Contracts deployed", "Flash-loan callback", "DeFi selectors"]
FCOLS = ["f1", "f2", "f3", "f4", "f5", "f6"]

LOGO = ('<svg width="36" height="36" viewBox="0 0 36 36" aria-hidden="true"><rect width="36" height="36" rx="9" fill="#1C2C44"/>'
        '<path d="M18 6.5l10 5.75v11.5L18 29.5 8 23.75v-11.5z" fill="none" stroke="#8DB4F5" stroke-width="1.6" stroke-linejoin="round"/>'
        '<polyline points="10.5,18.5 14,18.5 16,14 19,23 21,16.5 22.5,18.5 25.5,18.5" fill="none" stroke="#FFFFFF" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>'
        '<circle cx="28.5" cy="7.5" r="3.5" fill="#E4572E" stroke="#0E1A2B" stroke-width="1.5"/></svg>')

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, .stApp, [class*="css"] {{ font-family:'Inter',system-ui,sans-serif; color:{INK}; font-variant-numeric:tabular-nums; }}
.stApp {{ background:{BG}; }}
[data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display:none !important; }}
[data-testid="stExpandSidebarButton"] {{ background:{NAVY} !important; border-radius:8px; color:#FFFFFF !important; }}
[data-testid="stExpandSidebarButton"] * {{ fill:#FFFFFF !important; color:#FFFFFF !important; }}
[data-testid="stExpandSidebarButton"] {{ width:40px; height:40px; }}
[data-testid="stAppDeployButton"] {{ display:none !important; }}
header[data-testid="stHeader"] {{ background:transparent; height:2rem; }}
.block-container {{ padding-top:1.6rem; padding-bottom:3rem; max-width:1280px; }}
[data-testid="stSidebar"] {{ background:{NAVY}; border-right:0; }}
[data-testid="stSidebar"] * {{ color:#E6ECF3; }}
[data-testid="stSidebarHeader"] {{ height:1rem; padding:0; }}
[data-testid="stPageLink"] a {{ border-radius:8px; padding:.55rem .75rem; min-height:40px; }}
[data-testid="stPageLink"] a:hover {{ background:{NAVY2}; }}
[data-testid="stPageLink"] a span {{ color:#C8D2DF !important; font-size:14.5px; }}
[data-testid="stPageLink"] a[aria-current="page"] {{ background:{NAVY2}; }}
[data-testid="stPageLink"] a[aria-current="page"] span {{ color:#FFFFFF !important; font-weight:600; }}
.sb-brand {{ display:flex; align-items:center; gap:12px; padding:6px 4px 22px; }}
.sb-brand b {{ font-size:15.5px; display:block; color:#FFFFFF; }}
.sb-brand span {{ font-size:12px; color:#93A3B8 !important; }}
.sb-div {{ height:1px; background:{NAVY3}; margin:12px 6px; }}
.sb-h {{ font-size:11px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:#93A3B8 !important; margin:4px 6px 10px; }}
.sb-k {{ display:flex; align-items:center; gap:10px; font-size:13px; margin:0 6px 9px; }}
.sb-k i {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
.sb-meta {{ display:grid; grid-template-columns:auto 1fr; gap:7px 14px; font-size:12.5px; margin:0 6px; }}
.sb-meta span:nth-child(odd) {{ color:#93A3B8 !important; }}
.ph {{ display:flex; flex-wrap:wrap; justify-content:space-between; align-items:flex-end; gap:16px; margin-bottom:18px; }}
.ph h1 {{ margin:0; padding:0; font-size:32px; font-weight:700; letter-spacing:-.02em; }}
.ph p {{ margin:6px 0 0; font-size:15px; color:{MUTED}; }}
.tag {{ font-size:13px; font-weight:500; padding:0 !important; border:0 !important; background:none !important; color:{MUTED} !important; display:inline-flex; align-items:center; gap:6px; margin-left:18px; }}
.tag::before {{ content:"✓"; font-weight:700; color:{BLUE}; }}
.kpi {{ background:{CARD}; border:1px solid {LINE}; border-radius:12px; padding:18px 18px 16px; height:100%; box-sizing:border-box; }}
.kpi .l {{ font-size:12px; font-weight:600; color:{MUTED}; }}
.kpi .v {{ font-size:32px; font-weight:700; letter-spacing:-.02em; line-height:1.05; margin:8px 0; }}
.kpi .v small {{ font-size:19px; color:{MUTED2}; font-weight:600; }}
.kpi .s {{ font-size:12.5px; color:{MUTED}; margin-top:8px; }}
.trk {{ height:6px; border-radius:3px; background:{TRACK}; }} .trk div {{ height:6px; border-radius:3px; }}
[class*="st-key-card"] {{ background:{CARD}; border:1px solid {LINE} !important; border-radius:14px; padding:20px 22px; }}
.ct {{ font-size:16px; font-weight:700; margin:0 0 4px; }}
.cs {{ font-size:13px; color:{MUTED}; margin:0 0 8px; }}
.big {{ font-size:22px; line-height:1.4; font-weight:500; margin:6px 0 16px; }}
.seg {{ display:flex; gap:3px; height:44px; }} .seg div {{ display:flex; align-items:center; padding-left:12px; font-weight:700; font-size:15px; color:{INK}; border-radius:3px; }}
.lg {{ display:flex; flex-wrap:wrap; gap:8px 22px; font-size:13px; color:{MUTED}; margin-top:12px; }}
.lg span {{ display:flex; align-items:center; gap:6px; }} .lg i {{ width:10px; height:10px; border-radius:2px; display:inline-block; }}
table.t {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
table.t th {{ text-align:left; color:{MUTED}; font-size:11px; text-transform:uppercase; letter-spacing:.05em; font-weight:600; padding:6px 10px 10px; border-bottom:1px solid {LINE}; }}
table.t td {{ padding:11px 10px; border-bottom:1px solid #F0F2F3; vertical-align:middle; white-space:nowrap; }}
table.t tr:last-child td {{ border-bottom:0; }}
table.t .r {{ text-align:right; }}
.mono {{ font-family:'IBM Plex Mono',monospace; font-size:13px; }}
.pill {{ font-size:11.5px; font-weight:700; padding:4px 9px; border-radius:999px; white-space:nowrap; }}
.bar {{ flex:1; height:10px; border-radius:5px; background:{TRACK}; }} .bar div {{ height:10px; border-radius:5px; }}
.hb {{ display:grid; grid-template-columns:180px 1fr 30px; gap:10px 12px; align-items:center; font-size:13px; }}
.hb .b {{ height:14px; border-radius:3px; background:{TRACK}; }} .hb .b div {{ height:14px; border-radius:3px; background:{BLUE}; }}
.note {{ font-size:12.5px; color:{MUTED2}; }}
.acard {{ border-radius:12px; padding:16px 18px; height:100%; box-sizing:border-box; }}
.acard .h {{ display:flex; justify-content:space-between; align-items:center; }}
.acard .h span {{ font-size:12px; font-weight:700; letter-spacing:.04em; text-transform:uppercase; }}
.acard .h b {{ font-size:26px; }}
.acard p {{ margin:6px 0 0; font-size:13.5px; line-height:1.45; }}
.live {{ display:inline-flex; align-items:center; gap:10px; background:{CARD}; border:1px solid {LINE}; border-radius:999px; padding:8px 14px; font-size:13px; }}
.live i {{ width:9px; height:9px; border-radius:50%; background:{BLUE}; box-shadow:0 0 0 4px #D6E3F8; display:inline-block; }}
.ev {{ display:grid; grid-template-columns:46px 14px 1fr; gap:10px; padding-bottom:14px; font-size:13px; line-height:1.45; }}
.ev i {{ width:12px; height:12px; border-radius:50%; display:inline-block; margin-top:3px; }}
.strip {{ display:flex; gap:1px; min-width:110px; }} .strip div {{ flex:1; height:14px; border-radius:2px; }}
.box3 {{ border-radius:12px; padding:18px; height:100%; box-sizing:border-box; }}
.box3 ul {{ margin:8px 0 0; padding-left:18px; font-size:13.5px; line-height:1.5; }}
[data-testid="stExpander"] details {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; }}
[data-testid="stExpander"] summary {{ font-weight:600; font-size:15px; padding:14px 18px; }}
.lead {{ font-size:15px; color:{MUTED}; margin:-6px 0 18px; }}
div[data-testid="stButton"] button[kind="primary"] {{ background:{INK}; border:0; border-radius:8px; font-weight:600; }}
div[data-testid="stButton"] button[kind="primary"] * {{ color:#FFFFFF; }}
div[data-testid="stButton"] button[kind="primary"]:disabled {{ background:{GREY2}; }}
div[data-testid="stButton"] button[kind="secondary"] {{ background:{CARD}; border:1px solid {LINE}; border-radius:10px; font-weight:600; min-height:48px; color:{INK}; }}
div[data-testid="stButton"] button[kind="secondary"]:hover {{ border-color:{BLUE}; color:{BLUE}; }}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_data
def load():
    r = lambda n: pd.read_csv(os.path.join(D, n))
    d = dict(att=r("test_attack_horizons.csv"), nor=r("test_normal_horizons.csv"), res=r("test_results.csv"),
             thr=r("threshold_table.csv"), abl=r("ablation_calibration.csv"), win=r("window_choice.csv"),
             cases=r("cases.csv"), done=json.load(open(os.path.join(D, "TEST_DONE.json"))))
    hp = os.path.join(D, "wallet_hours.csv")
    d["hours"] = pd.read_csv(hp) if os.path.exists(hp) else None
    return d


@st.cache_resource
def model():
    return E.Frozen()


S, M = load(), model()
TAU = M.tau
NOR = np.sort(S["nor"].hscore.values)
H = S["hours"]


def stored_label(row, hw):
    if not bool(row.alert):
        if hw is not None and len(hw) and (hw[FCOLS].values == 0).all(): return "Silent"
        return "Normal"
    if int(row.n_alert_rows) == int(row.n_rows): return "Unusual from the start"
    if hw is not None and len(hw) and float(hw.sort_values("hours_before").f1.values[-1]) == 0: return "New wallet"
    return "Changed behaviour"


@st.cache_data
def cohort():
    """Per exploit wallet: label, age, main driver (from wallet_hours.csv); normal reference values; age table."""
    if H is None: return None
    att, nor = S["att"], S["nor"]
    rows = []
    for r in att.itertuples():
        hw = H[H.wid == r.wid].sort_values("hours_before")
        raw = hw[FCOLS].astype(float).values
        sh, hi = M.drivers(raw, hw.score.values) if r.alert else (np.zeros(6), 0)
        rows.append(dict(wid=r.wid, case=r.case_id, hscore=r.hscore, alert=bool(r.alert), label=stored_label(r, hw),
                         nohist=bool((raw[:, 0] == 0).all()), age=float(hw[hw.hours_before == 1].f1.iloc[0]),
                         driver=int(np.argmax(sh)) if r.alert else -1))
    c = pd.DataFrame(rows)
    h1 = H[(~H.wid.isin(att.wid)) & (H.hours_before == 1)]
    ref = pd.DataFrame({"median": h1[FCOLS].median(), "p95": h1[FCOLS].quantile(.95)})
    nage = nor.set_index("wid").join(h1.set_index("wid").f1).rename(columns={"f1": "age"})
    bands = [("Under 1 day", 0, 1), ("1–7 days", 1, 7), ("7–30 days", 7, 30), ("30 days–1 year", 30, 365), ("Over 1 year", 365, 1e12)]
    tab = []
    for name, lo, hi_ in bands:
        e = c[(c.age >= lo) & (c.age < hi_)]; n = nage[(nage.age >= lo) & (nage.age < hi_)]
        tab.append(dict(band=name, det=int(e.alert.sum()), n_e=len(e), fl=float(n.alert.mean()) if len(n) else 0, n_n=len(n)))
    return c, ref, pd.DataFrame(tab)


C = cohort()


@st.cache_data
def ext():
    """Extensions: profiles of the 19 detected attackers and 15 false warnings, and the evasion table."""
    if H is None: return None
    return X.compute(M, H, S["att"], S["nor"])


XT = ext()


@st.cache_data
def peaks_all():
    return X.all_peaks(H, S["att"], S["nor"]) if H is not None else None


PK = peaks_all()


# ---------------------------------------------------------------- helpers
def html(x): st.markdown(x, unsafe_allow_html=True)


def header(title, sub, tags=""):
    html(f'<div class="ph"><div><h1>{title}</h1><p>{sub}</p></div><div>{tags}</div></div>')


_cn = [0]
def card(title=None, sub=None):
    _cn[0] += 1
    c = st.container(border=True, key=f"card{_cn[0]}")
    if title: c.markdown(f'<p class="ct">{title}</p>' + (f'<p class="cs">{sub}</p>' if sub else ""), unsafe_allow_html=True)
    return c


def kpis(items):
    cols = st.columns(len(items), gap="small")
    for c, (l, v, s, w, col) in zip(cols, items):
        bar = f'<div class="trk"><div style="width:{w}%;background:{col}"></div></div>' if w is not None else ""
        c.markdown(f'<div class="kpi"><div class="l">{l}</div><div class="v">{v}</div>{bar}<div class="s">{s}</div></div>', unsafe_allow_html=True)


def pill(label):
    m, t, b = ST.get(label, ST["Not scored"])
    return f'<span class="pill" style="background:{b};color:{t}">{label}</span>'


def open_wallet(group, wallet, whatif=False):
    """Jump to Wallet Analysis with this wallet selected. group: 'exploit' (wallet = case id) or 'normal' (wallet = id)."""
    st.session_state["goto"] = (group, wallet); st.session_state["wa_whatif"] = whatif; st.switch_page(P_WA)


def open_view(view):
    st.session_state["ba_view"] = view; st.switch_page(P_BA)


def opener(k, items, group, key, label="Open in Wallet Analysis", whatif=False):
    """A wallet picker and a button that opens the picked wallet."""
    a, b = k.columns([3, 1.3], vertical_alignment="bottom")
    pick = a.selectbox("Open a wallet", items, key=f"pick_{key}", label_visibility="collapsed")
    if b.button(label, key=f"btn_{key}", type="primary", width="stretch", icon=":material/open_in_new:"):
        open_wallet(group, pick.split(" · ")[0], whatif)


def cpill(label):
    sty = {"Attack-like contract": f"background:{INK};color:#FFFFFF", "Ordinary contract": f"background:{TRACK};color:{MUTED}",
           "No contract": f"background:#FFFFFF;color:{MUTED};border:1px solid {LINE}"}[label]
    return f'<span class="pill" style="{sty}">{label}</span>'


def sc(ch):
    return (ch.configure_axis(labelFont="Inter", titleFont="Inter", labelColor=MUTED2, titleColor=MUTED, gridColor="#EEF0F1",
                              domainColor=GREY2, tickColor=GREY2, labelFontSize=11.5, titleFontSize=12, titleFontWeight="normal")
              .configure_view(strokeWidth=0).configure(background=CARD))


LOGAX = dict(labelExpr="datum.value==-7?'1e-7':datum.value==-5?'1e-5':datum.value==-3?'0.001':datum.value==-1?'0.1':datum.value==1?'10':'1,000'",
             values=[-7, -5, -3, -1, 1, 3])


def pct_below(x): return float((NOR < x).mean() * 100)
def pctf(x): p = pct_below(x); return "&gt;99.9%" if p >= 99.95 else f"{p:.1f}%"
def xthr(x): r = x / TAU; return f"{r:,.0f}×" if r >= 100 else (f"{r:.2g}×" if r >= .01 else "&lt;0.01×")
def fmt_age(d):
    if d is None: return "—"
    return f"{d * 24:.0f} h" if d < 1 else (f"{d:.1f} d" if d < 10 else f"{d:,.0f} d")
def fmt_score(x): return "—" if x is None else (f"{x:,.0f}" if x >= 100 else f"{x:.3g}")


def hourly_chart(scores, label, h=240):
    df = pd.DataFrame({"hb": np.arange(1, len(scores) + 1), "score": np.clip(scores, 1e-7, None)})
    df["state"] = np.where(df.score >= TAU, "Above threshold", "Below")
    col = ST.get(label, ST["Unusual from the start"])[0]
    if label in ("Normal", "Silent", "Not scored"): col = ST["Unusual from the start"][0]
    x = alt.X("hb:Q", scale=alt.Scale(domain=[24, 1]), title="Hours before the reference time", axis=alt.Axis(values=[24, 18, 12, 6, 1]))
    y = alt.Y("score:Q", scale=alt.Scale(type="log", domain=[1e-7, 1e4]), title="Score (log scale)",
              axis=alt.Axis(values=[1e-6, 1e-4, 1e-2, 1, 100, 1e4], labelExpr="format(datum.value,'.0e')"))
    ln = alt.Chart(df).mark_line(color="#B8BFC4", strokeWidth=2).encode(x=x, y=y)
    pt = alt.Chart(df).mark_circle(size=70, stroke="#FFFFFF", strokeWidth=1.5, opacity=1).encode(
        x=x, y=y, color=alt.Color("state:N", scale=alt.Scale(domain=["Above threshold", "Below"], range=[col, BLUE]), legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("hb:Q", title="hours before"), alt.Tooltip("score:Q", format=".3g")])
    rule = alt.Chart(pd.DataFrame({"y": [TAU]})).mark_rule(color=INK, strokeDash=[5, 4], strokeWidth=1.5).encode(y="y:Q")
    txt = alt.Chart(pd.DataFrame({"y": [TAU], "x": [24]})).mark_text(align="left", dx=4, dy=-8, fontSize=11.5, fontWeight=600, color=INK).encode(x="x:Q", y="y:Q", text=alt.value("threshold 0.012"))
    return sc((ln + pt + rule + txt).properties(height=h))


def feature_table(vals, shares, ref, accent):
    rows = ""
    for i in range(6):
        v, med, p95 = float(vals[i]), float(ref.iloc[i]["median"]), float(ref.iloc[i]["p95"])
        if i == 0: vs = fmt_age(v) if v > 0 else "no history"; ms, ps = f"{med:,.0f} d", f"{p95:,.0f} d"
        elif i == 1: vs = "first funding" if v >= 1e8 else f"{v:.3g}"; ms, ps = f"{med:.3g}", f"{p95:.3g}"
        elif i == 2: vs, ms, ps = f"{v:.0%}", f"{med:.0%}", f"{p95:.0%}"
        elif i == 4: vs, ms, ps = ("yes" if v >= 1 else "no"), ("yes" if med >= 1 else "no"), ("yes" if p95 >= 1 else "no")
        else: vs, ms, ps = f"{v:,.0f}", f"{med:,.0f}", f"{p95:,.0f}"
        hi = i > 0 and v > p95
        w = float(shares[i]) * 100; top = shares.sum() > 0 and i == int(np.argmax(shares))
        bar = (f'<div style="display:flex;align-items:center;gap:10px"><div class="bar"><div style="width:{w:.0f}%;background:{accent if top else GREY}"></div></div>'
               f'<span style="width:38px;text-align:right;{"font-weight:700" if top else ""}">{w:.0f}%</span></div>') if shares.sum() > 0 else '<span class="note">—</span>'
        rows += (f'<tr><td><span class="note" style="margin-right:8px">F{i+1}</span><b style="font-weight:600">{FSHORT[i]}</b></td>'
                 f'<td class="r" style="font-weight:600;{"color:#B23A16" if hi else ""}">{vs}{" ▲" if hi else ""}</td><td class="r">{ms}</td><td class="r">{ps}</td><td style="width:34%">{bar}</td></tr>')
    return (f'<div style="overflow-x:auto"><table class="t" style="min-width:620px"><tr><th>Feature</th><th class="r">This wallet</th><th class="r">Normal median</th>'
            f'<th class="r">Normal 95th pct</th><th>Share of the score</th></tr>{rows}</table></div>'
            f'<p class="note" style="margin:8px 0 0">▲ above the 95th percentile of normal wallets. Share explains the score, not intent.</p>')


# ---------------------------------------------------------------- page 1
def page_overview():
    header("Detection Overview", "36 exploit wallets · 2,000 normal wallets · test period from 2023",
           f'<span class="tag" style="background:#E4ECFB;color:#1F55B8">Stored results</span><span class="tag" style="background:#fff;color:{MUTED};border:1px solid {LINE}">Model frozen before the test</span>')
    kpis([("Exploits detected", "19<small> / 36</small>", "52.8% · 95% interval 37–68%", 52.8, ST["Unusual from the start"][0]),
          ("False warnings", "0.75%", "15 of 2,000 · reference rule 3.45%", 21.7, BLUE),
          ("Ranking quality (AUC)", f"{S['done']['auc']:.2f}", "0.5 = chance · 1 = perfect", S["done"]["auc"] * 100, MUTED),
          ("Median lead time", "24 h", "Most alerts on from the first check", 100, MUTED),
          ("Isolation forest", "2<small> / 36</small>", "Comparison model", 5.6, GREY)])
    st.write("")
    if C is None:
        st.info("Put wallet_hours.csv in the data folder to see the full overview."); return
    c, ref, tab = C
    n = c.label.value_counts()
    u, nw, ch = int(n.get("Unusual from the start", 0)), int(n.get("New wallet", 0)), int(n.get("Changed behaviour", 0))
    mis = c[~c.alert]; s_old, s_new = int((~mis.nohist).sum()), int(mis.nohist.sum())
    k = card()
    k.markdown(f'<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px"><p class="ct" style="font-size:18px">What kind of alert?</p><span class="cs">All 36 exploit wallets</span></div>'
               f'<p class="big">{u + nw + ch} alerts: <b style="color:#B23A16">{u} unusual from the start</b> · <b style="color:#6F5205">{nw} new wallets</b> · <b style="color:#8E2477">{ch} changed behaviour</b></p>'
               f'<div class="seg"><div style="flex:{u};background:{ST["Unusual from the start"][0]}">{u}</div>'
               + (f'<div style="flex:{nw};background:{ST["New wallet"][0]};justify-content:center;padding:0">{nw}</div>' if nw else "")
               + (f'<div style="flex:{ch};background:{ST["Changed behaviour"][0]}">{ch}</div>' if ch else "")
               + f'<div style="flex:{s_old};background:{GREY}">{s_old}</div><div style="flex:{s_new};background:#D5DADD">{s_new}</div></div>'
               f'<div class="lg"><span><i style="background:{ST["Unusual from the start"][0]}"></i>Unusual from the start</span><span><i style="background:{ST["New wallet"][0]}"></i>New wallet</span>'
               f'<span><i style="background:{ST["Changed behaviour"][0]}"></i>Changed behaviour</span><span><i style="background:{GREY}"></i>Not flagged, looked normal</span><span><i style="background:#D5DADD"></i>Not flagged, no activity yet</span></div>', unsafe_allow_html=True)
    st.write("")
    k = card("Score distribution: exploit vs normal wallets")
    k.markdown(f'<div class="lg" style="margin:0 0 4px"><span><i style="background:{ST["Unusual from the start"][0]};border-radius:50%"></i>Unusual from the start</span>'
               f'<span><i style="background:{ST["New wallet"][0]};border-radius:50%"></i>New wallet</span><span><i style="background:{GREY};border-radius:50%"></i>Exploit, not flagged</span>'
               f'<span><i style="background:{GREY2}"></i>Normal wallets</span><span><i style="background:none;border-top:2px dashed {INK};height:0;width:16px"></i>Threshold</span></div>', unsafe_allow_html=True)
    lg = np.log10(np.clip(NOR, 1e-7, None)); cnt, ed = np.histogram(lg, bins=np.linspace(-7, 3, 41))
    hist = pd.DataFrame({"a": ed[:-1] + .02, "b": ed[1:] - .02, "n": cnt})
    xs = alt.Scale(domain=[-7.2, 3.2])
    rng = np.random.default_rng(3)
    ex = c.assign(lx=np.log10(np.clip(c.hscore, 1e-7, None)), y=rng.uniform(0, 1, len(c)),
                  grp=c.label.replace({"Silent": "Exploit, not flagged", "Normal": "Exploit, not flagged"}))
    dom = ["Unusual from the start", "New wallet", "Changed behaviour", "Exploit, not flagged"]
    rngc = [ST["Unusual from the start"][0], ST["New wallet"][0], ST["Changed behaviour"][0], GREY]
    ex["yy"] = 470 + ex.y * 150
    base_x = alt.X("lx:Q", scale=xs, axis=alt.Axis(title="Anomaly score, highest of 24 hourly checks (log scale)", grid=False, **LOGAX))
    yax = alt.Y("n:Q", scale=alt.Scale(domain=[0, 640]), axis=alt.Axis(values=[0, 100, 200, 300, 400], title="Normal wallets"))
    bars = alt.Chart(hist).mark_rect(color=GREY2, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("a:Q", scale=xs, axis=alt.Axis(title="Anomaly score, highest of 24 hourly checks (log scale)", grid=False, **LOGAX)), x2="b:Q", y=yax,
        tooltip=[alt.Tooltip("n:Q", title="normal wallets")])
    dots = alt.Chart(ex).mark_circle(size=80, opacity=1, stroke="#fff", strokeWidth=1.5).encode(
        x=base_x, y=alt.Y("yy:Q", scale=alt.Scale(domain=[0, 640])),
        color=alt.Color("grp:N", scale=alt.Scale(domain=dom, range=rngc), legend=None), tooltip=["case", "grp", alt.Tooltip("hscore:Q", format=".3g", title="score")])
    sep = alt.Chart(pd.DataFrame({"y": [445]})).mark_rule(color="#E3E6E8").encode(y="y:Q")
    lab = alt.Chart(pd.DataFrame({"x": [-7.1], "y": [628], "t": ["36 exploit wallets"]})).mark_text(align="left", fontSize=11.5, color=MUTED2).encode(x="x:Q", y="y:Q", text="t:N")
    lab2 = alt.Chart(pd.DataFrame({"x": [-7.1], "y": [430], "t": ["2,000 normal wallets"]})).mark_text(align="left", fontSize=11.5, color=MUTED2).encode(x="x:Q", y="y:Q", text="t:N")
    rule = alt.Chart(pd.DataFrame({"x": [np.log10(TAU)]})).mark_rule(color=INK, strokeDash=[5, 4], strokeWidth=1.5).encode(x="x:Q")
    k.altair_chart(sc((sep + bars + dots + rule + lab + lab2).properties(height=380)), width="stretch")
    st.write("")
    with st.expander("More detail: detection by wallet age, and what drives each alert"):
        g1, g2 = st.columns(2, gap="small")
        with g1:
            k = card("Detection by wallet age")
            old = int((c[c.alert].age >= 7).sum())
            rows = "".join(f'<tr><td style="font-weight:600">{r.band}</td><td><div style="display:flex;align-items:center;gap:8px"><div class="bar" style="height:8px"><div style="height:8px;width:{(r.det / r.n_e * 100) if r.n_e else 0:.0f}%;background:{ST["Unusual from the start"][0]}"></div></div>'
                           f'<span style="width:56px">{r.det} of {r.n_e}</span></div></td><td>{r.fl:.1%} <span class="note">of {r.n_n:,}</span></td></tr>' for r in tab.itertuples())
            k.markdown(f'<p style="font-size:14px;margin:0 0 10px"><b>Not age-driven:</b> {old} of {int(c.alert.sum())} detected wallets were older than a week.</p>'
                       f'<table class="t"><tr><th>Wallet age</th><th>Exploits detected</th><th>Normal flagged</th></tr>{rows}</table>'
                       f'<p class="note" style="margin:8px 0 0">{int(c.nohist.sum())} exploit wallets had no history yet in the window.</p>', unsafe_allow_html=True)
        with g2:
            k = card("Main driver of each alert")
            d = c[c.alert].driver.value_counts().reindex(range(6), fill_value=0).sort_values(ascending=False, kind="stable")
            mx = max(int(d.max()), 1)
            age_old = int(((c.driver == 0) & (c.age >= 7)).sum())
            k.markdown(f'<p style="font-size:14px;margin:0 0 14px"><b>Wallet age leads {int(d[0])}</b>, {age_old} of them older than a week. Funding burst leads {int(d[1])}.</p>'
                       '<div class="hb">' + "".join(f'<span>F{i+1} {FSHORT[i]}</span><div class="b"><div style="width:{v / mx * 100:.0f}%"></div></div><b>{v}</b>' for i, v in d.items()) + '</div>'
                       '<p class="note" style="margin:14px 0 0">Feature with the largest share of the score. Explains the score, not intent.</p>', unsafe_allow_html=True)
    st.write("")
    k = card("Go further", "Extensions computed from these stored results. No score or threshold changes.")
    b1, b4, b2, b3 = k.columns(4, gap="small")
    if b1.button("What attackers looked like", icon=":material/category:", width="stretch", key="ov_t"): open_view("Attacker types")
    if b4.button("Does it flag every contract builder?", icon=":material/construction:", width="stretch", key="ov_c"): open_view("Contract builders")
    if b2.button("Why 15 normal wallets alerted", icon=":material/help:", width="stretch", key="ov_f"): open_view("False warnings")
    if b3.button("Could an attacker hide?", icon=":material/visibility_off:", width="stretch", key="ov_h"): open_view("Hiding from the model")
    html('<p class="note" style="margin-top:14px">Addresses are public. A score is a research signal, never evidence against a person.</p>')


# ---------------------------------------------------------------- page 2
def page_wallet():
    header("Wallet Analysis", "Score, timing and drivers for one wallet.")
    att, nor, cs = S["att"].copy(), S["nor"].copy(), S["cases"].set_index("case_id")
    go = st.session_state.pop("goto", None)
    G = ["Exploit wallets · 36", "Normal wallets · 2,000"]
    if go: st.session_state["wa_grp"] = G[1] if go[0] == "normal" else G[0]
    if st.session_state.get("wa_grp") not in G: st.session_state["wa_grp"] = G[0]
    k = card()
    a, b = k.columns([1.35, 2], vertical_alignment="center")
    grp = a.segmented_control("Group", G, key="wa_grp", label_visibility="collapsed") or G[0]
    if grp.startswith("Exploit"):
        att = att.merge(cs[["protocol", "exploit_date"]], left_on="case_id", right_index=True, how="left")
        att["lab"] = att.case_id + " · " + att.protocol.astype(str) + " · " + att.exploit_date.astype(str) + " · " + att.wallet.str[:6] + "…" + att.wallet.str[-4:]
        labs = list(att.lab)
        if go and go[0] == "exploit" and (att.case_id == go[1]).any(): st.session_state["wa_exp"] = att.lab[att.case_id == go[1]].iloc[0]
        if st.session_state.get("wa_exp") not in labs:
            st.session_state["wa_exp"] = att.lab[att.case_id == "EXP036"].iloc[0] if (att.case_id == "EXP036").any() else labs[0]
        r = att[att.lab == b.selectbox("Wallet", labs, key="wa_exp", label_visibility="collapsed")].iloc[0]
        is_exp = True
    else:
        wids = list(nor.wid)
        if go and go[0] == "normal" and go[1] in wids: st.session_state["wa_nor"] = go[1]
        if st.session_state.get("wa_nor") not in wids: st.session_state["wa_nor"] = wids[0]
        r = nor[nor.wid == b.selectbox("Wallet", wids, key="wa_nor", label_visibility="collapsed")].iloc[0]
        is_exp = False
    hw = H[H.wid == r.wid].sort_values("hours_before") if H is not None else None
    label = stored_label(r, hw)
    score = float(r.hscore); n_al = int(r.n_alert_rows); lead = None if pd.isna(r.lead_h) else int(r.lead_h)
    ref_t = "the exploit" if is_exp else "the reference time"
    st.write("")
    sent = {"Unusual from the start": "Above the threshold in all 24 checks: already unusual when the window opened.",
            "New wallet": f"Alert began <b>{lead} h before {ref_t}</b>, when the wallet was created.",
            "Changed behaviour": f"An existing wallet crossed the threshold <b>{lead} h before {ref_t}</b>.",
            "Silent": "No activity yet: the wallet did not exist in the 24 h window, so there was nothing to flag.",
            "Normal": "Below the threshold in all 24 checks."}[label]
    lp = pill(label) if label != "Silent" else pill("Not scored").replace("Not scored", "Silent")
    facts, addr = "", r.wid
    tags = ""
    if hw is not None and len(hw) and bool(r.alert):
        xp = X.peak(hw)[FCOLS].values.astype(float)
        tags = (f'<div style="display:flex;gap:8px 10px;flex-wrap:wrap;align-items:center;margin-top:12px;font-size:13px">'
                f'<span class="note">Behaviour type</span><b>{X.behaviour_type(xp)}</b><span class="note" style="margin-left:12px">Contract check</span>{cpill(X.contract_check(xp))}</div>')
    if is_exp:
        cr = cs.loc[r.case_id]; addr = r.wallet
        facts = (f'<div style="display:grid;grid-template-columns:auto auto;gap:6px 16px;font-size:13px"><span class="note">Protocol</span><b>{cr.protocol}</b>'
                 f'<span class="note">Exploit</span><b>{cr.exploit_date}</b><span class="note">Label tier</span><b>{str(r.tier).title()}</b><span class="note">Evidence</span><b>{str(r.evidence).title()}</b></div>')
    k = card()
    k.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:24px;align-items:center;justify-content:space-between"><div style="flex:1 1 480px">'
               f'<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">{lp}<span class="mono" style="color:{MUTED}">{addr}</span></div>'
               f'<p style="font-size:19px;line-height:1.45;font-weight:500;margin:10px 0 0">{sent}</p>{tags}</div>{facts}</div>', unsafe_allow_html=True)
    st.write("")
    drv = "—"; shares = np.zeros(6); hi = 0
    if hw is not None and len(hw) and score > 0:
        raw = hw[FCOLS].astype(float).values; shares, hi = M.drivers(raw, hw.score.values)
        j = int(np.argmax(shares)); drv = f"F{j+1} {FSHORT[j]}"
    accent = ST.get(label, ST["Unusual from the start"])[0]
    kpis([("Anomaly score", fmt_score(score), f"{xthr(score)} the threshold", None, None),
          ("Normal wallets below it", pctf(score), "of 2,000 normal test wallets", None, None),
          ("Checks above", f"{n_al}<small> of 24</small>", f"first alert {lead} h before" if lead else "never above", None, None),
          ("Main driver", f'<span style="font-size:22px">{drv}</span>', f"{shares.max():.0%} of the score" if shares.sum() else "no score to explain", None, None)])
    st.write("")
    if hw is None or not len(hw):
        st.info("wallet_hours.csv is not loaded."); return
    views = ["When it alerted", "Why it alerted"] + (["What if"] if bool(r.alert) and XT is not None else [])
    kv = f"wa_view_{r.wid}"
    if st.session_state.pop("wa_whatif", False) and "What if" in views: st.session_state[kv] = "What if"
    if st.session_state.get(kv) not in views: st.session_state[kv] = views[0]
    view = st.segmented_control("Section", views, key=kv, label_visibility="collapsed") or views[0]
    st.write("")
    if view == "When it alerted":
        k = card(f"Hourly score, 24 h before {ref_t}", "Each dot is one hourly check. Above the dashed line = alert.")
        k.altair_chart(hourly_chart(hw.score.values, label), width="stretch")
    elif view == "Why it alerted":
        k = card("Feature values vs normal wallets", f"At the highest-scoring check, {int(hw.hours_before.values[hi])} h before.")
        k.markdown(feature_table(hw[FCOLS].astype(float).values[hi], shares, C[1], accent), unsafe_allow_html=True)
    else:
        whatif(hw, label)


def whatif(hw, label):
    ref = XT[2]
    k = card("What if: could this wallet hide from the model?",
             "Give chosen features the value of a typical normal wallet in all 24 checks and score again with the frozen model. Stored results do not change.")
    opts = k.multiselect("Make these features look normal", FSHORT, key=f"wi_{hw.wid.iloc[0]}", placeholder="Choose one or more features")
    idx = [FSHORT.index(o) for o in opts]
    Xh = hw.sort_values("hours_before")[FCOLS].values.astype(float).copy()
    for j in idx: Xh[:, j] = ref[j]
    sc_ = M.errors(M.prep(Xh)); new, old = float(sc_.max()), float(hw.score.max())
    hidden = new < TAU
    verdict = ('<span class="pill" style="background:#E4ECFB;color:#1F55B8">Below the threshold: hidden</span>' if hidden
               else '<span class="pill" style="background:#FCE5DC;color:#B23A16">Still above the threshold</span>')
    k.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:12px 32px;align-items:center;margin:4px 0 6px;font-size:14px">'
               f'<span><span class="note">Stored score</span> <b>{fmt_score(old)}</b></span><span><span class="note">Score after the change</span> <b>{fmt_score(new)}</b> '
               f'<span class="note">({xthr(new)} the threshold)</span></span>{verdict if idx else ""}</div>'
               f'<p class="note" style="margin:0">Typical normal wallet: {ref[0]:,.0f} days old, no funding burst, no failed transactions, no contracts.</p>', unsafe_allow_html=True)
    if idx: k.altair_chart(hourly_chart(sc_, label, 200), width="stretch")


# ---------------------------------------------------------------- page: behaviour analysis (extensions)
def two_bars(a, na, b, nb, c=None, nc=None):
    def one(v, n, col):
        return (f'<div style="display:flex;align-items:center;gap:8px"><div class="bar" style="height:8px"><div style="height:8px;width:{max(v / n * 100, 1 if v else 0):.0f}%;background:{col}"></div></div>'
                f'<span style="width:{78 if n >= 1000 else 62}px">{v:,} of {n:,}</span></div>')
    out = f'<td style="width:24%">{one(a, na, ST["Unusual from the start"][0])}</td><td style="width:24%">{one(b, nb, BLUE)}</td>'
    return out + (f'<td style="width:24%">{one(c, nc, GREY)}</td>' if c is not None else "")


def page_behaviour():
    header("Behaviour Analysis", "What the alerts look like, why some normal wallets alert, and how easily an attacker could hide.",
           f'<span class="tag" style="background:#E4ECFB;color:#1F55B8">Extension</span><span class="tag" style="background:#fff;color:{MUTED};border:1px solid {LINE}">No score or threshold changed</span>')
    if XT is None:
        st.info("Put wallet_hours.csv in the data folder to see this page."); return
    P, EV, ref = XT
    A = P[P.group == "Detected attacker"].copy(); F = P[P.group == "False warning"].copy()
    nA, nF = len(A), len(F)
    if "ba_view" in st.session_state: st.session_state["ba_seg"] = st.session_state.pop("ba_view")
    if st.session_state.get("ba_seg") is None: st.session_state["ba_seg"] = "Attacker types"
    view = st.segmented_control("View", ["Attacker types", "Contract builders", "False warnings", "Hiding from the model"], key="ba_seg", label_visibility="collapsed") or "Attacker types"
    st.write("")
    if view == "Attacker types":
        has_c = int(((A.f4 > 0) | (A.f5 > 0) | (A.f6 > 0)).sum()); fl = int((A.f5 > 0).sum())
        k = card()
        k.markdown(f'<p class="ct" style="font-size:18px">What did the {nA} detected attackers look like before the attack?</p>'
                   f'<p class="big"><b>All {has_c}</b> had built a contract, and <b>{fl}</b> owned one that can receive a flash loan.</p>', unsafe_allow_html=True)
        cc = k.columns(3, gap="small")
        for c_, t in zip(cc, X.TYPES[:3]):
            n = int((A.type == t).sum())
            c_.markdown(f'<div class="acard" style="background:#F6F7F8;border:1px solid {LINE}"><div class="h"><span style="color:{MUTED}">Type {X.TYPES.index(t) + 1}</span><b>{n}</b></div>'
                        f'<p style="font-weight:600;color:{INK};font-size:15px">{t}</p><p style="color:{MUTED};min-height:42px">{X.TYPE_TEXT[t]}</p></div>', unsafe_allow_html=True)
        st.write("")
        ex = st.expander(f"See all {nA} attackers and open one")
        k = ex.container()
        k.caption("Features at each wallet's highest-scoring check, 24 hours before the exploit at most.")
        A = A.assign(o=A.type.map({t: i for i, t in enumerate(X.TYPES)})).sort_values(["o", "score"], ascending=[True, False])
        rows = "".join(f'<tr><td style="font-weight:600">{r.case_id}</td><td>{r.type}</td><td class="r">{fmt_age(r.f1)}</td><td class="r">{r.f4:,.0f}</td>'
                       f'<td>{"yes" if r.f5 > 0 else "no"}</td><td class="r">{r.f6:,.0f}</td><td>{cpill(r.contract)}</td><td class="r">{fmt_score(r.score)}</td></tr>' for r in A.itertuples())
        k.markdown(f'<div style="overflow-x:auto"><table class="t" style="min-width:900px"><tr><th>Case</th><th>Behaviour type</th><th class="r">Age</th><th class="r">Contracts built</th>'
                   f'<th>Flash-loan contract</th><th class="r">DeFi functions</th><th>Contract check</th><th class="r">Score</th></tr>{rows}</table></div>'
                   f'<p class="note" style="margin:8px 0 0">Types are defined from the wallets\' behaviour after the test, so they describe these 19 cases; they are not a new detection rule.</p>', unsafe_allow_html=True)
        opener(k, [f"{r.case_id} · {r.type}" for r in A.itertuples()], "exploit", "types")
    elif view == "Contract builders":
        Nn = PK[PK.group == "Normal"]; Aa = PK[PK.group == "Attacker"]
        nb = int((Nn.contract != "No contract").sum()); nbf = int(((Nn.contract != "No contract") & Nn.alert).sum())
        k = card()
        k.markdown(f'<p class="ct" style="font-size:18px">Does the model flag everyone who builds a contract?</p>'
                   f'<p class="big">Building a contract is rare: only <b>{nb} of {len(Nn):,}</b> normal wallets did it, so the model sees it as unusual and flagged <b>{nbf}</b> of them. '
                   f'The contract check is what separates honest builders from attackers.</p>', unsafe_allow_html=True)
        def cell(df, lab):
            d = df[df.contract == lab]; return f'<td class="r">{len(d):,}</td><td class="r" style="font-weight:700">{int(d.alert.sum())}</td>'
        rows = "".join(f'<tr><td>{cpill(l)}</td>{cell(Nn, l)}{cell(Aa, l)}</tr>' for l in X.CONTRACT[::-1])
        k.markdown(f'<div style="overflow-x:auto"><table class="t" style="min-width:620px"><tr><th>What the wallet built</th><th class="r">Normal wallets</th><th class="r">of them flagged</th>'
                   f'<th class="r">Attackers</th><th class="r">of them flagged</th></tr>{rows}</table></div>'
                   f'<p class="note" style="margin:8px 0 0">Test set: 2,000 normal wallets and 36 attackers. Contract features at each wallet\'s highest-scoring check.</p>', unsafe_allow_html=True)
        st.write("")
        g1, g2 = st.columns(2, gap="small")
        al = PK[PK.alert]
        with g1:
            k = card("What each label means for an alarm", "Of all 34 alarms in the test (19 attackers, 15 normal wallets).")
            def share(lab):
                d = al[al.contract == lab]; n_a = int((d.group == "Attacker").sum()); return n_a, len(d)
            (a1, t1), (a2, t2) = share("Attack-like contract"), share("Ordinary contract")
            k.markdown(f'<div class="hb" style="grid-template-columns:170px 1fr 92px">'
                       f'<span>{cpill("Attack-like contract")}</span><div class="b"><div style="width:{a1 / max(t1, 1) * 100:.0f}%;background:{ST["Unusual from the start"][0]}"></div></div><b>{a1} of {t1}</b>'
                       f'<span>{cpill("Ordinary contract")}</span><div class="b"><div style="width:{a2 / max(t2, 1) * 100:.0f}%;background:{ST["Unusual from the start"][0]}"></div></div><b>{a2} of {t2}</b></div>'
                       f'<p style="font-size:13.5px;margin:14px 0 0">Bar = share of these alarms that were <b>real attackers</b>. An attack-like alarm was an attacker {a1} times out of {t1}; '
                       f'an ordinary-contract alarm only {a2} times out of {t2}, usually a developer.</p>', unsafe_allow_html=True)
        with g2:
            k = card("How to treat each alarm")
            k.markdown(f'<table class="t"><tr><td>{cpill("Attack-like contract")}</td><td style="white-space:normal"><b>Check first.</b> The contract can take a flash loan, like most attack contracts.</td></tr>'
                       f'<tr><td>{cpill("Ordinary contract")}</td><td style="white-space:normal"><b>Likely a developer.</b> Check later; do not treat as an attacker.</td></tr>'
                       f'<tr><td>{cpill("No contract")}</td><td style="white-space:normal"><b>Unusual for another reason</b>, such as funding or failed transactions.</td></tr></table>'
                       f'<p class="note" style="margin:8px 0 0">The check labels alarms; it removes none, so no real attacker is lost. Removing ordinary-contract alarms would drop 13 false alarms but also 5 attackers.</p>', unsafe_allow_html=True)
        st.write("")
        k = card("Signs that could separate honest builders even better", "Not in the study data yet: future work, no numbers.")
        cc = k.columns(4, gap="small")
        for c_, (t, d) in zip(cc, [("Published code", "Honest contracts are usually verified on Etherscan; attack contracts usually are not."),
                                   ("Used over time", "Honest contracts are used again and again; attack contracts usually once."),
                                   ("Source of money", "Funding through a mixer, such as Tornado Cash, hides who paid."),
                                   ("Who built it", "A known team or long history, versus a fresh unknown wallet.")]):
            c_.markdown(f'<div class="acard" style="background:#F6F7F8;border:1px solid {LINE}"><p style="font-weight:600;color:{INK};margin:0">{t}</p><p style="color:{MUTED}">{d}</p></div>', unsafe_allow_html=True)
        k.markdown('<p class="note" style="margin:10px 0 0">No method can prove intent before an attack. These signs, with the contract check, help a person decide which alarms to look at first.</p>', unsafe_allow_html=True)
    elif view == "False warnings":
        k = card()
        nf_fl = int((F.f5 > 0).sum()); na_fl = int((A.f5 > 0).sum())
        k.markdown(f'<p class="ct" style="font-size:18px">Why do {nF} of 2,000 normal wallets alert?</p>'
                   f'<p class="big">They are unusual too, but rarely in the attackers\' way: only <b>{nf_fl} of {nF}</b> owns a flash-loan contract, against <b>{na_fl} of {nA}</b> detected attackers.</p>', unsafe_allow_html=True)
        k.markdown(f'<div class="lg" style="margin:0 0 6px"><span><i style="background:{ST["Unusual from the start"][0]}"></i>Detected attackers ({nA})</span><span><i style="background:{BLUE}"></i>False warnings ({nF})</span><span><i style="background:{GREY}"></i>All normal wallets (2,000)</span></div>', unsafe_allow_html=True)
        cmp = [("Less than a week old", lambda d: d.f1 < 7), ("Built a contract in the last day", lambda d: d.f4 > 0),
               ("Owns contracts with DeFi functions", lambda d: d.f6 > 0), ("Owns a flash-loan contract", lambda d: d.f5 > 0),
               ("Had failed transactions", lambda d: d.f3 > 0), ("First funding (nothing received before)", lambda d: d.f2 >= 1e8)]
        NA = PK[PK.group == "Normal"]
        rows = "".join(f'<tr><td style="font-weight:600;white-space:normal">{n}</td>{two_bars(int(f(A).sum()), nA, int(f(F).sum()), nF, int(f(NA).sum()), len(NA))}</tr>' for n, f in cmp)
        k.markdown(f'<div style="overflow-x:auto"><table class="t" style="min-width:820px"><tr><th>Behaviour before the reference time</th><th>Detected attackers</th><th>False warnings</th><th>All normal wallets</th></tr>{rows}</table></div>'
                   f'<p class="note" style="margin:8px 0 0">All normal wallets: the 2,000 normal test wallets, flagged or not. Building contracts is rare among them.</p>', unsafe_allow_html=True)
        st.write("")
        g1, g2 = st.columns(2, gap="small")
        with g1:
            k = card("Same behaviour types, both groups")
            rows = "".join(f'<tr><td style="font-weight:600;white-space:normal">{t}</td><td class="r">{int((A.type == t).sum())}</td><td class="r">{int((F.type == t).sum())}</td></tr>' for t in X.TYPES)
            k.markdown(f'<table class="t"><tr><th>Type</th><th class="r">Attackers</th><th class="r">False warnings</th></tr>{rows}</table>'
                       f'<p class="note" style="margin:8px 0 0">Most false warnings are developers: they build ordinary contracts or own DeFi contracts without a flash-loan function.</p>', unsafe_allow_html=True)
        with g2:
            k = card("A second check on every alert: the contract", "It labels each alert; it does not remove any.")
            rows = "".join(f'<tr><td>{cpill(t)}</td><td class="r">{int((A.contract == t).sum())}</td><td class="r">{int((F.contract == t).sum())}</td></tr>' for t in X.CONTRACT)
            k.markdown(f'<table class="t"><tr><th>Contract check</th><th class="r">Attackers</th><th class="r">False warnings</th></tr>{rows}</table>'
                       f'<p style="font-size:13px;margin:10px 0 0">An alert with an <b>attack-like contract</b> deserves attention first. One with an ordinary contract is more likely a developer.</p>'
                       f'<p class="note" style="margin:6px 0 0">Found on the test results, so it must be confirmed on new exploits. Code alone cannot prove intent: honest trading bots also use flash loans.</p>', unsafe_allow_html=True)
        st.write("")
        ex = st.expander(f"See all {nF} false warnings and open one")
        k = ex.container()
        F = F.sort_values("score", ascending=False)
        rows = "".join(f'<tr><td class="mono">{r.wid}</td><td>{r.type}</td><td class="r">{fmt_age(r.f1)}</td><td class="r">{r.f4:,.0f}</td><td class="r">{r.f6:,.0f}</td>'
                       f'<td>{cpill(r.contract)}</td><td class="r">{fmt_score(r.score)}</td></tr>' for r in F.itertuples())
        k.markdown(f'<div style="overflow-x:auto"><table class="t" style="min-width:860px"><tr><th>Wallet</th><th>Behaviour type</th><th class="r">Age</th><th class="r">Contracts built</th>'
                   f'<th class="r">DeFi functions</th><th>Contract check</th><th class="r">Score</th></tr>{rows}</table></div>', unsafe_allow_html=True)
        opener(k, [f"{r.wid} · {r.type}" for r in F.itertuples()], "normal", "false")
    else:
        k_ = EV.changes_needed.astype(str).value_counts()
        n1, n2, n3 = int(k_.get("1", 0)), int(k_.get("2", 0)), int(k_.get("3", 0)) + int(k_.get(">3", 0))
        k = card()
        k.markdown(f'<p class="ct" style="font-size:18px">How many changes would hide a detected attacker?</p>'
                   f'<p class="big">One feature change hides <b>{n1} of {nA}</b>. <b>{n2}</b> need two changes and <b>{n3}</b> need three. Making the wallet look older is never enough on its own.</p>'
                   f'<div class="seg"><div style="flex:{n1};background:{BLUE};color:#fff">{n1}</div><div style="flex:{n2};background:#9DB9EC">{n2}</div><div style="flex:{n3};background:{GREY2};justify-content:center;padding:0">{n3}</div></div>'
                   f'<div class="lg"><span><i style="background:{BLUE}"></i>1 change</span><span><i style="background:#9DB9EC"></i>2 changes</span><span><i style="background:{GREY2}"></i>3 changes</span></div>', unsafe_allow_html=True)
        k.markdown(f'<p style="font-size:13px;margin:14px 0 0">Almost every way to hide removes the contract features. In practice that means building the attack contract from <b>another wallet</b>, so linking wallets is the natural next defence.</p>', unsafe_allow_html=True)
        st.write("")
        ex = st.expander(f"See the fewest changes for each of the {nA} attackers, and try one")
        k = ex.container()
        k.caption("Each change gives one feature the value of a typical normal wallet in all 24 checks, then the frozen model scores again.")
        EV = EV.assign(n=EV.changes_needed.astype(str)).sort_values(["n", "score"], ascending=[True, False])
        rows = "".join(f'<tr><td style="font-weight:600">{r.case_id}</td><td>{r.type}</td><td class="r">{fmt_score(r.score)}</td><td class="r" style="font-weight:700">{r.changes_needed}</td>'
                       f'<td style="white-space:normal">{r.example}</td></tr>' for r in EV.itertuples())
        k.markdown(f'<div style="overflow-x:auto"><table class="t" style="min-width:760px"><tr><th>Case</th><th>Behaviour type</th><th class="r">Score</th><th class="r">Changes needed</th><th>Change that hides it</th></tr>{rows}</table></div>'
                   f'<p class="note" style="margin:8px 0 0">Pick a wallet and press Try it in What if to change the features yourself.</p>', unsafe_allow_html=True)
        opener(k, [f"{r.case_id} · {r.changes_needed} change{'s' if str(r.changes_needed) != '1' else ''}" for r in EV.itertuples()], "exploit", "hide", "Try it in What if", True)
    html('<p class="note" style="margin-top:14px">Computed from the stored test features with the frozen model. The thesis results (19 of 36, 0.75%, AUC 0.81) are unchanged.</p>')


# ---------------------------------------------------------------- page 3
def get_key():
    key = os.environ.get("ETHERSCAN_KEY", "")
    try: key = st.secrets.get("ETHERSCAN_KEY", key)
    except Exception: pass
    return key


ALERTS = ("Unusual from the start", "New wallet", "Changed behaviour")


def run_feed(key, proto, lookback, limit, box):
    cl = E.Client(key); now = int(time.time()) - 60
    hm = dt.datetime.fromtimestamp(now, dt.timezone.utc).strftime("%H:%M")
    box.caption(f"Reading recent senders on {proto}…")
    senders = E.recent_senders(cl, E.PROTOCOLS[proto], lookback_s=lookback, limit=limit, now_ts=now)
    end_block = cl.block_at(now)
    prev = st.session_state.setdefault("feed_prev", {})
    events = st.session_state.setdefault("feed_events", [])
    first_run = not st.session_state.get("feed_started")
    out = []; bar = box.progress(0.0, text="Scoring wallets")
    for i, (a, t) in enumerate(senders):
        r = E.feed_score(a, now, end_block, cl, M)
        p = prev.get((proto, a))
        short = f"{a[:6]}…{a[-4:]}"
        if r["label"] in ALERTS:
            if p is not None and p["label"] == "Normal" and (r["age_d"] or 0) >= 1:
                # crossed the threshold between two refreshes while we were watching
                r["label"] = "Changed behaviour"; r["first_alert"] = f"{hm} · while watched"
                events.insert(0, dict(t=hm, label="Changed behaviour", text=f"{short} crossed the threshold",
                                      sub=f"{fmt_score(p['score'])} → {fmt_score(r['score'])} · {fmt_age(r['age_d'])} old"))
            elif p is not None and p["label"] in ALERTS:
                r["first_alert"] = p.get("first_alert") or "earlier scan"
            else:
                r["first_alert"] = {"Unusual from the start": "first scan", "New wallet": f"{hm} · at creation", "Changed behaviour": f"{hm} · in window"}[r["label"]]
                if r["label"] != "Unusual from the start":
                    events.insert(0, dict(t=hm, label=r["label"], text=f"{short} {'first seen, new wallet' if r['label'] == 'New wallet' else 'changed behaviour'}",
                                          sub=f"{fmt_age(r['age_d'])} old · score {fmt_score(r['score'])}"))
        prev[(proto, a)] = dict(label=r["label"], score=r["score"], first_alert=r.get("first_alert"))
        out.append(r); bar.progress((i + 1) / max(len(senders), 1), text=f"Scored {i + 1} of {len(senders)}")
    bar.empty(); box.empty()
    if first_run:
        st.session_state["feed_started"] = True
        events.append(dict(t=hm, label="Not scored", text=f"Feed started on {proto}",
                           sub=f"{sum(r['label'] == 'Unusual from the start' for r in out)} wallets already unusual on the first scan"))
    st.session_state["feed"] = dict(proto=proto, at=now, rows=out, calls=cl.calls, n_senders=len(senders))
    tot = st.session_state.setdefault("feed_totals", {})
    for r in out:
        if r["label"] != "Not scored": tot[(proto, r["address"])] = r["label"]


def feed_view():
    F = st.session_state.get("feed")
    if not F:
        k = card()
        k.markdown('<p class="ct">Press Refresh now to start.</p><p class="cs" style="margin:0">The feed reads the wallets that just used the selected protocol and scores each one '
                   'with the frozen model. Turn on auto refresh to repeat this every 5 minutes.</p>', unsafe_allow_html=True)
        return
    rows = F["rows"]; scored = [r for r in rows if r["label"] != "Not scored"]
    cnt = {l: sum(r["label"] == l for r in rows) for l in E.LABELS}
    n_al = sum(cnt[l] for l in ALERTS)
    at = dt.datetime.fromtimestamp(F["at"], dt.timezone.utc).strftime("%H:%M UTC")
    html(f'<div style="text-align:right;margin:0 0 12px"><span class="live"><i></i><b>Live</b><span style="color:{MUTED}">updated {at} · {F["calls"]} API calls</span></span></div>')
    k = card()
    segs = "".join(f'<div style="flex:{cnt[l]};background:{ST[l][0] if l != "Normal" else "#D0DFF7"};padding:0"></div>' for l in ["Unusual from the start", "New wallet", "Changed behaviour", "Normal"] if cnt[l])
    k.markdown(f'<div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:16px;align-items:flex-end"><div><p class="cs" style="margin:0;font-weight:600">Right now on {F["proto"]}</p>'
               f'<p style="margin:0;font-size:26px;font-weight:700">{n_al} alert{"s" if n_al != 1 else ""} among {len(scored)} wallets</p></div>'
               f'<div style="display:flex;gap:28px;font-size:13px;color:{MUTED}"><div>Scored<br><b style="font-size:18px;color:{INK}">{len(scored)}</b></div>'
               f'<div>Not scored<br><b style="font-size:18px;color:{INK}">{cnt["Not scored"]}</b></div><div>Alert rate<br><b style="font-size:18px;color:{INK}">{(n_al / max(len(scored), 1)):.0%}</b></div></div></div>'
               f'<div class="seg" style="height:14px;margin:16px 0">{segs}</div>', unsafe_allow_html=True)
    tot = st.session_state.get("feed_totals", {}); nt = len(tot); na = sum(v in ALERTS for v in tot.values())
    k.markdown(f'<p class="note" style="margin:0 0 14px">This session: <b>{nt}</b> distinct wallets scored, <b>{na}</b> alert{"s" if na != 1 else ""}. '
               f'Most real users should score normal: in the study only 0.75% of normal wallets alerted, about 1 in 130, so expect roughly {nt * 0.0075:.1f} alerts so far.</p>', unsafe_allow_html=True)
    cc = k.columns(3, gap="small")
    for c_, l, txt in zip(cc, ALERTS, ["Unusual in all 24 checks. <b>Anomaly.</b>", "Created inside the window. <b>Anomaly at creation.</b>",
                                       "Established wallet crossed the threshold. <b>Early-warning candidate.</b>"]):
        m, t, b = ST[l]
        border = f"2px solid {m}" if l == "Changed behaviour" else f"1px solid {b}"
        c_.markdown(f'<div class="acard" style="background:{b}80;border:{border}"><div class="h"><span style="color:{t}">{l}</span><b>{cnt[l]}</b></div><p>{txt}</p></div>', unsafe_allow_html=True)
    st.write("")
    g1 = st.container(); g2a, g2b = None, None
    order = {l: i for i, l in enumerate(E.LABELS)}
    rs = sorted(scored, key=lambda r: (order[r["label"]], -(r["score"] or 0)))
    tx = (np.log10(TAU) + 6) / 9 * 100
    with g1:
        k = card("Wallets scored, alerts first")
        show = st.session_state.get("feed_all", False)
        body = ""
        for r in (rs if show else rs[:10]):
            m = ST[r["label"]][0]; lx = min(max((np.log10(max(r["score"], 1e-7)) + 6) / 9 * 100, 0), 100)
            empty = (r["res"]["features"][::-1, 0] == 0) if r["label"] == "New wallet" else [False] * len(r["flags"])
            strip = "".join(f'<div style="background:{(m if r["label"] != "Normal" else BLUE) if f else ("#EEF0F1" if e else "#D0DFF7")}"></div>' for f, e in zip(r["flags"], empty))
            alert = r["label"] in ALERTS
            body += (f'<tr style="{"background:#FDF5FA" if r["label"] == "Changed behaviour" else ""}"><td>{pill(r["label"])}</td><td class="mono">{r["address"][:6]}…{r["address"][-4:]}</td>'
                     f'<td class="r">{fmt_age(r["age_d"])}</td><td><div style="display:flex;align-items:center;gap:8px"><span style="width:56px;font-weight:{700 if alert else 400}">{fmt_score(r["score"])}</span>'
                     f'<div style="position:relative;flex:1;height:6px;border-radius:3px;background:{TRACK};min-width:70px"><div style="position:absolute;left:{tx:.1f}%;top:-4px;width:1px;height:14px;background:{INK}"></div>'
                     f'<div style="position:absolute;left:{lx:.1f}%;top:-3px;width:12px;height:12px;margin-left:-6px;border-radius:50%;background:{m};border:2px solid #fff;box-sizing:border-box"></div></div></div></td>'
                     f'<td><div class="strip">{strip}</div></td>'
                     f'<td>{cpill(X.contract_check(r["res"]["features"][int(np.argmax(r["res"]["scores"]))])) if alert else "—"}</td>'
                     f'<td style="{"font-weight:600" if r["label"] == "Changed behaviour" else "color:" + MUTED}">{r.get("first_alert") or "—"}</td></tr>')
        k.markdown(f'<div style="overflow-x:auto"><table class="t" style="min-width:860px"><tr><th>Status</th><th>Wallet</th><th class="r">Age</th><th style="width:170px">Score vs threshold</th>'
                   f'<th style="width:150px">Last 24 checks</th><th>Contract check</th><th>First alert</th></tr>{body}</table></div>', unsafe_allow_html=True)
        if len(rs) > 10 and k.button("Show fewer wallets" if show else f"Show {len(rs) - 10} more wallets", width="stretch", key="feed_more"):
            st.session_state["feed_all"] = not show; st.rerun()
        skipped = [r for r in rows if r["label"] == "Not scored"]
        if skipped: k.caption("Not scored: " + ", ".join(sorted({r["reason"] for r in skipped})) + ".")
    st.write("")
    with st.expander("More detail: threshold crossings and wallet age"):
        g2a, g2b = st.columns(2, gap="small")
        with g2a:
            k = card("Threshold crossings while watching")
            ev = st.session_state.get("feed_events", [])[:8]
            k.markdown("".join(f'<div class="ev"><span style="font-weight:600;color:{MUTED}">{e["t"]}</span><i style="background:{ST.get(e["label"], ST["Not scored"])[0]}"></i>'
                               f'<span><b>{e["text"]}</b><br><span style="color:{MUTED}">{e["sub"]}</span></span></div>' for e in ev) or '<p class="cs">No crossings yet.</p>', unsafe_allow_html=True)
        with g2b:
            k = card("Wallet age vs score", "Above the dashed line = alert.")
            df = pd.DataFrame([dict(age=max(r["age_d"], 0.02), score=max(r["score"], 1e-7), label=r["label"], wallet=r["address"][:10]) for r in scored])
            if len(df):
                pts = alt.Chart(df).mark_circle(size=60, opacity=.95, stroke="#fff", strokeWidth=1).encode(
                    x=alt.X("age:Q", scale=alt.Scale(type="log"), title="Wallet age, days (log)", axis=alt.Axis(values=[0.1, 1, 10, 100, 1000], format="~g")), y=alt.Y("score:Q", scale=alt.Scale(type="log"), title="Score (log)", axis=alt.Axis(values=[1e-6, 1e-4, 1e-2, 1, 100, 1e4], labelExpr="format(datum.value,'.0e')")),
                    color=alt.Color("label:N", scale=alt.Scale(domain=E.LABELS[:4], range=[ST[l][0] if l != "Normal" else "#9DB9EC" for l in E.LABELS[:4]]), legend=None),
                    tooltip=["wallet", "label", alt.Tooltip("age:Q", format=".1f"), alt.Tooltip("score:Q", format=".3g")])
                rule = alt.Chart(pd.DataFrame({"y": [TAU]})).mark_rule(color=INK, strokeDash=[4, 3]).encode(y="y:Q")
                k.altair_chart(sc((pts + rule).properties(height=230)), width="stretch")


def show_result(out, expect=None, title=None):
    res = out["result"]
    if res is None: st.info(out["rejected_reason"]); return
    lab = E.alert_label(res, int(min(out["hist"]["ts"])), out["as_of_ts"])
    sh, hi = M.drivers(res["features"], res["scores"])
    k = card(title)
    when = dt.datetime.fromtimestamp(out["as_of_ts"], dt.timezone.utc).strftime("%d %b %Y %H:%M UTC")
    cchk = cpill(X.contract_check(res["features"][hi])) if lab in ALERTS else ""
    k.markdown(f'<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">{pill(lab)}{cchk}<span class="mono">{out["address"]}</span>'
               f'<span class="note">score {fmt_score(res["hscore"])} · {xthr(res["hscore"])} threshold · {res["n_alert"]} of 24 checks above · as of {when} · {out["n_tx"]:,} transactions read live</span></div>', unsafe_allow_html=True)
    if res.get("quiet"):
        k.markdown('<p class="note" style="margin:8px 0 0">No outgoing transaction in the 24 hours before this time; the wallet is scored from its history, as in the study.</p>', unsafe_allow_html=True)
    if expect is not None:
        ok = abs(expect - res["hscore"]) / max(expect, 1e-12) < 1e-3
        k.markdown(f'<p style="margin:10px 0 0;font-size:14px"><span class="pill" style="background:{"#E4ECFB" if ok else "#FCE5DC"};color:{"#1F55B8" if ok else "#B23A16"}">'
                   f'{"Matches the study" if ok else "Differs from the study"}</span> &nbsp;stored test score {fmt_score(expect)} · live score {fmt_score(res["hscore"])}</p>', unsafe_allow_html=True)
    k.altair_chart(hourly_chart(res["scores"], lab, 200), width="stretch")
    if C is not None: k.markdown(feature_table(res["features"][hi], sh, C[1], ST.get(lab, ST["Unusual from the start"])[0]), unsafe_allow_html=True)


def check_one(key, addr):
    try:
        out = E.check_wallet(addr, int(time.time()) - 120, E.Client(key), M)
    except E.Rejected as e:
        st.warning(str(e)); return
    except Exception as e:
        st.error("Could not complete the check: " + str(e)[:160]); return
    st.session_state["one"] = dict(out=out, expect=None, title="Single wallet, scored now", kind="check")


def replay_case(key, case_id):
    cs = S["cases"].set_index("case_id"); c = cs.loc[case_id]
    a = S["att"][S["att"].case_id == case_id]
    expect = float(a.hscore.iloc[0]) if len(a) else None
    try:
        cl = E.Client(key)
        with st.spinner(f"Replaying {case_id} from live Etherscan data…"):
            ts = cl.block_time(int(c.block))
            out = E.check_wallet(c.sender, ts, cl, M, anchor_hash=c.tx_hash, anchor_block=int(c.block), allow_heavy=True)
    except E.Rejected as e:
        st.warning(str(e)); return
    except Exception as e:
        st.error("Could not complete the replay: " + str(e)[:160]); return
    st.session_state["one"] = dict(out=out, expect=expect, title=f"Replay: {case_id} · {c.protocol} · just before the exploit", kind="replay")


def page_live():
    header("Live Monitoring", "Real wallets scored now by the same frozen model.")
    key = get_key()
    if not key:
        st.warning("Live Monitoring needs an Etherscan key. In Colab: add ETHERSCAN_KEY under Secrets (key icon). Locally: put it in .streamlit/secrets.toml.")
    views = ["Live feed", "Check one wallet", "Replay a known exploit"]
    view = st.segmented_control("Section", views, default="Live feed", key="live_view", label_visibility="collapsed") or "Live feed"
    st.write("")
    one = st.session_state.get("one")
    if view == "Check one wallet":
        k = card("Check one wallet", "Paste any Ethereum address. It is scored as of now, on its last 24 hours.")
        b2, b3 = k.columns([3.4, 1], vertical_alignment="bottom")
        addr = b2.text_input("Wallet address", placeholder="0x…", label_visibility="collapsed")
        if b3.button("Check", type="primary", disabled=not key or not addr, width="stretch", icon=":material/search:"): check_one(key, addr)
        one = st.session_state.get("one")
        if one and one.get("kind") == "check":
            st.write(""); show_result(one["out"], one["expect"], one["title"])
        return
    if view == "Replay a known exploit":
        k = card("Replay a known exploit", "Proof that the live part works: rebuilds an attacker wallet from live Etherscan data, just before its attack, and compares with the study.")
        cs = S["cases"]; cs = cs[cs.case_id.isin(S["att"].case_id)]
        flagged = set(S["att"][S["att"].alert.astype(bool)].case_id)
        opts = [f"{r.case_id} · {r.protocol} · {r.exploit_date}{' · flagged in the study' if r.case_id in flagged else ''}" for r in cs.itertuples()]
        dflt = next((i for i, o in enumerate(opts) if o.startswith("EXP031")), 0)
        r1, r2 = k.columns([3.4, 1], vertical_alignment="bottom")
        pick = r1.selectbox("Exploit case", opts, index=dflt, label_visibility="collapsed")
        if r2.button("Replay", type="primary", disabled=not key, width="stretch", icon=":material/replay:"): replay_case(key, pick.split(" · ")[0])
        one = st.session_state.get("one")
        if one and one.get("kind") == "replay":
            st.write(""); show_result(one["out"], one["expect"], one["title"])
        return
    k = card()
    c1, c2, c3, c4 = k.columns([4.2, 1.3, 1.1, 1.3], vertical_alignment="bottom")
    proto = c1.segmented_control("Protocol", list(E.PROTOCOLS), default="Uniswap V3") or "Uniswap V3"
    look = c2.selectbox("Look back", [("Last 30 minutes", 1800), ("Last 2 hours", 7200)], format_func=lambda x: x[0])[1]
    limit = c3.selectbox("Wallets per refresh", [20, 40, 100], index=0)
    auto = c4.toggle("Auto refresh", value=False, help="Refresh every 5 minutes")
    go = k.button("Refresh now", type="primary", disabled=not key, icon=":material/refresh:")

    @st.fragment(run_every=300 if (auto and key) else None)
    def feed_block():
        box = st.empty()
        F = st.session_state.get("feed")
        due = auto and (F is None or F["proto"] != proto or time.time() - F["at"] > 280)
        if key and (go or due):
            try: run_feed(key, proto, look, limit, box)
            except Exception as e: st.error("Feed failed: " + str(e)[:200])
        feed_view()
    st.write("")
    feed_block()


# ---------------------------------------------------------------- page 4
def page_model():
    header("Model &amp; Limits", "How a wallet becomes an alert, and what an alert can mean.")
    steps = [("1 · Learn", "Normal wallets only", "An autoencoder learns to rebuild normal behaviour.", "#1F55B8"),
             ("2 · Describe", "Six features, hourly", "Each check uses the previous day of transactions.", "#1F55B8"),
             ("3 · Score", "Rebuild error", "The highest of 24 hourly checks.", "#1F55B8"),
             ("4 · Alert", "Score ≥ 0.0120", "About 0.5% of normal calibration wallets alert.", "#B23A16"),
             ("5 · Label", "What kind of alert", "Unusual from the start, new wallet, or changed behaviour.", "#8E2477")]
    for c_, (a, b, t, col) in zip(st.columns(5, gap="small"), steps):
        c_.markdown(f'<div class="kpi"><div style="font-size:12px;font-weight:700;color:{col}">{a}</div><div style="font-size:16px;font-weight:700;margin:6px 0">{b}</div><div class="s" style="margin:0">{t}</div></div>', unsafe_allow_html=True)
    st.write("")
    g1, g2 = st.columns(2, gap="small")
    with g1:
        k = card("The six features")
        defs = [("Wallet age", "Days since the wallet's first transaction"), ("Funding burst", "Largest transfer received in the window ÷ everything received before it"),
                ("Failed share", "Share of the wallet's outgoing transactions that failed"), ("Contracts deployed", "Contracts the wallet created in the window"),
                ("Flash-loan callback", "Whether a contract it deployed has a flash-loan callback function"), ("DeFi selectors", "How many of 16 DeFi function signatures its contracts contain")]
        k.markdown('<table class="t">' + "".join(f'<tr><td class="note" style="font-weight:600">F{i+1}</td><td style="font-weight:600;white-space:nowrap">{a}</td><td style="color:#34393E;white-space:normal">{b}</td></tr>' for i, (a, b) in enumerate(defs)) + '</table>'
                   '<p class="note" style="margin:8px 0 0">Two earlier features (F7, F8) were removed after a control check on development data.</p>', unsafe_allow_html=True)
    with g2:
        k = card("Choices made on calibration data, before the test")
        w = S["win"]; ab = S["abl"].set_index("model").auc_calibration
        def bars(items):
            return ('<div class="hb" style="grid-template-columns:120px 1fr 48px;margin-bottom:16px">' + "".join(
                f'<span style="{"font-weight:700" if best else ""}">{n}</span><div class="b"><div style="width:{v * 100:.1f}%;background:{BLUE if best else GREY}"></div></div>'
                f'<span style="{"font-weight:700" if best else ""}">{v:.3f}</span>' for n, v, best in items) + '</div>')
        k.markdown('<p class="cs" style="font-weight:600">Window length · calibration AUC</p>' +
                   bars([(f"{int(r.window_days)} day{'s' if r.window_days > 1 else ''}", float(r.auc_calibration), r.window_days == 1) for r in w.itertuples()]) +
                   '<p class="cs" style="font-weight:600">What carries the signal · calibration AUC</p>' +
                   bars([("All 6", float(ab.iloc[0]), True), ("Without F5–F6", float(ab[ab.index.str.contains("F5")].iloc[0]), False),
                         ("Without F1–F4", float(ab[ab.index.str.contains("F1")].iloc[0]), False), ("8 features", 0.52, False)]) +
                   '<p style="font-size:13px;margin:0">Budgets of 0.5%, 1% and 2% all caught 2 of 12 calibration exploits, so the strictest, <b>0.5%</b>, sets the threshold.</p>', unsafe_allow_html=True)
    st.write("")
    k = card("What the model can and cannot establish")
    cc = k.columns(3, gap="small")
    boxes = [("Can establish", "#1F55B8", "#EEF3FC", ["A wallet looks unlike normal wallets on six public features.", "Whether that was already true or started in the last 24 hours.", "Which features drive the score."]),
             ("Cannot establish", MUTED, "#F2F3F4", ["Intent, or that an attack is coming.", "That a wallet below the threshold is safe: 17 of 36 exploit wallets were not flagged, 10 had no activity yet.", "Anything about skipped wallets: contracts, over 30,000 transactions."]),
             ("What the study shows", "#B23A16", "#FDF1EC", ["17 of 19 caught attackers looked unusual for the whole day before the attack.", "2 new wallets were flagged from creation, 3 h and 13 h before the attack.", "The six-feature design was fixed after a first test run, on the same 48 cases."])]
    for c_, (t, col, bg, items) in zip(cc, boxes):
        c_.markdown(f'<div class="box3" style="background:{bg}"><span style="font-size:12px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:{col}">{t}</span><ul>' + "".join(f"<li>{x}</li>" for x in items) + "</ul></div>", unsafe_allow_html=True)
    st.write("")
    k = st.expander("Frozen settings and model checksum")
    k.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:12px 34px;font-size:13px">'
               f'<span><span class="note">Main threshold</span> <span class="mono">{TAU:.6f}</span></span><span><span class="note">Reference threshold</span> <span class="mono">{M.tau_ref:.6f}</span></span>'
               f'<span><span class="note">Network</span> <b>6 → 16 → 3 → 16 → 6</b></span><span><span class="note">Window</span> <b>1 day, step 1 h</b></span>'
               f'<span><span class="note">Model checksum</span> <span class="mono">{M.sha_model}</span></span><span><span class="note">Frozen</span> <b>{M.s["frozen_at_utc"][:10]}</b></span></div>'
               f'<p class="note" style="margin:12px 0 0">Thesis: {E.THESIS_TITLE}</p>', unsafe_allow_html=True)


# ---------------------------------------------------------------- navigation
P_OV = st.Page(page_overview, title="Detection Overview", icon=":material/dashboard:", url_path="overview", default=True)
P_WA = st.Page(page_wallet, title="Wallet Analysis", icon=":material/account_balance_wallet:", url_path="wallet")
P_BA = st.Page(page_behaviour, title="Behaviour Analysis", icon=":material/insights:", url_path="behaviour")
P_MO = st.Page(page_model, title="Model & Limits", icon=":material/tune:", url_path="model")
P_LI = st.Page(page_live, title="Live Monitoring", icon=":material/sensors:", url_path="live")
pg = st.navigation([P_OV, P_WA, P_BA, P_MO, P_LI], position="hidden")
with st.sidebar:
    html(f'<div class="sb-brand">{LOGO}<div><b>DeFi Early Warning</b><span>Exploit monitoring on Ethereum</span></div></div>')
    for p in (P_OV, P_WA, P_BA, P_MO): st.page_link(p)
    html('<div class="sb-div"></div>')
    st.page_link(P_LI)
    html('<div class="sb-div" style="margin-top:18px"></div><div class="sb-h">Status key</div>' +
         "".join(f'<div class="sb-k"><i style="background:{ST[l][0]}"></i>{l if l != "Not scored" else "Silent or not scored"}</div>' for l in ["Unusual from the start", "New wallet", "Changed behaviour", "Normal", "Not scored"]) +
         f'<div class="sb-div" style="margin-top:14px"></div><div class="sb-meta"><span>Threshold</span><span>{TAU:.4f}</span><span>Window</span><span>1 day · 24 checks</span>'
         f'<span>Features</span><span>6</span><span>Frozen</span><span>{M.s["frozen_at_utc"][:10]}</span></div>')
pg.run()
