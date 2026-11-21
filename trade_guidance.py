# ==================== COMPONENT 5: TRADE GUIDANCE SYSTEM (rule-based, not predictive) ====================
# The trading decision model: at the close of a 15-minute candle it reads only completed
# candles and returns WAIT, SETUP, ENTER, NO TRADE or EXIT with an explanation.
# It reuses the same rules as the back-test (View, find_ob, simulate_exit).

WINDOW_DAYS = 150          # history used each time (covers the longest setup, 62-day holding period and warm-up)
EXIT_TEXT = {"stop": "stop-loss hit", "target": "take-profit hit", "htf_reversal": "four-hour structure reversed", "open_at_end": "maximum holding period reached"}

def _completed(df1, t):
    """Only completed candles: 1-minute candles that ended by t, and 15-minute and four-hour candles that ended by t."""
    i0 = df1.index.searchsorted(t - pd.Timedelta(days=WINDOW_DAYS)); i1 = df1.index.searchsorted(t); d = df1.iloc[i0:i1]
    if len(d) < 60 * 24 * 20: return None
    m15 = rs(d, "15min"); h4 = rs(d, "4h")
    m15 = m15[m15.index + pd.Timedelta("15min") <= t].iloc[1:]; h4 = h4[h4.index + pd.Timedelta("4h") <= t].iloc[1:]
    if len(m15) < 200 or len(h4) < 30: return None
    return {"1m": d, "15m": m15, "H4": h4}

def trace(V, st, mode, n, buf):
    """Follows one setup through the SAME rules as the back-test, but only up to the last completed 15-minute candle."""
    o, h, l, c = V.L; N = len(h); last = N - 1; sp = V.Lsp; sh15 = V.sw15(n)
    zl, zh, tgt = st["zl"], st["zh"], st["target"]; base = dict(zl=zl, zh=zh, target=tgt, act=st["act"], last=last)
    closed = lambda why, at: dict(base, kind="closed", why=why, at=at)
    j = int(np.searchsorted(V.Lt, st["act"]))
    if j > last: return dict(base, kind="await_touch")
    touch = None
    for jj in range(j, min(j + EXP_A, N)):
        if st["inval"] and V.kmap[jj] > st["inval"][1] and V.H[3][V.kmap[jj]] < st["inval"][0]: return closed("price returned inside the range", jj)
        if c[jj] < zl: return closed("a 15-minute candle closed beyond the far side of the zone", jj)
        if l[jj] <= zh: touch = jj; break
    if touch is None: return closed("price did not reach the zone in time", j + EXP_A - 1) if j + EXP_A <= N else dict(base, kind="await_touch")
    trig = None
    for jj in range(touch, min(touch + EXP_B, N)):
        if c[jj] < zl: return closed("a 15-minute candle closed beyond the far side of the zone", jj)
        lo_i = np.searchsorted(sh15, touch - 3, side="left"); hi_i = np.searchsorted(sh15, jj - n, side="right")
        if hi_i > lo_i and c[jj] > h[sh15[hi_i - 1]]: trig = (jj, sh15[hi_i - 1]); break
    if trig is None: return closed("no confirmation within 24 hours", touch + EXP_B - 1) if touch + EXP_B <= N else dict(base, kind="await_trigger")
    jj, ish = trig; extra = {}
    if mode == "momentum":
        b0 = l[touch:jj + 1].min()
        if jj + 1 <= last: e = jj + 1; entry = o[e] + sp[e]; stop = b0 - buf * sp[e]; kind = "entered"; t0 = V.Lt[e]
        else: entry = c[jj] + sp[jj]; stop = b0 - buf * sp[jj]; kind = "enter_now"; t0 = V.Lt[jj] + np.timedelta64(15, "m")
        decided = jj
    else:
        Lidx = ish + int(np.argmin(l[ish:jj + 1])); ob = find_ob(o, c, h, l, Lidx + 1, Lidx - 6)
        if ob is None: return closed("no order block found for the entry", jj)
        zl2, zh2 = l[ob], h[ob]; fill = None
        for j2 in range(jj + 1, min(jj + 1 + EXP_C, N)):
            if c[j2] < zl2: return closed("a 15-minute candle closed beyond the entry zone before the retest", j2)
            if l[j2] <= zh2: fill = j2; break
        if fill is None:
            if jj + 1 + EXP_C <= N: return closed("no retest of the entry zone within 24 hours", jj + EXP_C)
            entry = zh2 + sp[last]; stop = zl2 - buf * sp[last]; kind = "pending"; t0 = None; decided = jj
        else:
            entry = min(zh2, o[fill]) + sp[fill]; stop = zl2 - buf * sp[fill]; t0 = V.Lt[fill] + np.timedelta64(15, "m"); kind = "enter_now" if fill == last else "entered"; decided = fill
    rr = (tgt - entry) / (entry - stop) if entry > stop else np.nan
    if not (entry > stop and tgt > entry) or not rr >= MIN_RR: return closed(f"reward-to-risk is below the 3:1 minimum ({rr:.1f}:1)" if rr == rr else "invalid stop and target levels", decided)
    return dict(base, kind=kind, entry=entry, stop=stop, rr=rr, hl=st["hl"], t0=t0)

def _exit_state(V, c0):
    s = int(np.searchsorted(V.Mt, c0["t0"]))
    if s >= len(V.Mt) - 1: return dict(open=True)
    ex = simulate_exit(V, c0); avail = len(V.Mt) - s
    if ex["why"] == "open_at_end" and avail < CAP: return dict(open=True, R_now=ex["R"])
    return dict(open=False, code=ex["why"], R=ex["R"], t_exit=pd.Timestamp(ex["t_exit"]))

def guidance(df1, t, code, equity=START_EQ):
    """One decision from completed candles only. Returns WAIT, SETUP, ENTER, NO TRADE or EXIT with a short explanation."""
    sel = SEL.get(code)
    if not sel: raise RuntimeError(f"No strategy was selected for {code}. Run the back-test first; nothing is guessed.")
    point = INSTRUMENTS[code]["point"]; dp = max(0, int(round(-np.log10(point)))); f = lambda x: f"{x:.{dp}f}"; t = pd.Timestamp(t)
    res = dict(time=t, instrument=code, strategy=LABEL[sel["strategy"]], settings=f"swing lookback {sel['n']}, stop buffer {sel['buf']}x spread", decision="WAIT", direction="-",
               entry=None, stop=None, target=None, reward_to_risk=None, risk_amount=None, size_units=None, reason_code=None, explanation="")
    T = _completed(df1, t)
    if T is None: res["explanation"] = "Not enough completed candles before this time to assess the strategy."; return res
    entered, other = [], []
    for mirror in (False, True):
        V = View(T, mirror, point)
        for st in V.setups(sel["mode"]):
            tr = trace(V, st, sel["mode"], sel["n"], sel["buf"]); tr["V"] = V; tr["m"] = V.m
            if tr["kind"] == "entered":
                tr["c0"] = dict(t0=tr["t0"], entry=tr["entry"], stop=tr["stop"], target=tr["target"], hl=tr["hl"]); tr["ex"] = _exit_state(V, tr["c0"]); entered.append(tr)
            else: other.append(tr)
    acc, free_at = [], pd.Timestamp.min                       # one position at a time, in order of entry time
    for tr in sorted(entered, key=lambda x: pd.Timestamp(x["t0"])):
        if pd.Timestamp(tr["t0"]) >= free_at: acc.append(tr); free_at = pd.Timestamp.max if tr["ex"]["open"] else tr["ex"]["t_exit"]
    d = lambda tr: "BUY" if tr["m"] == 1 else "SELL"; lv = lambda tr, k: tr["m"] * tr[k]
    def fill_levels(tr):
        en, sp_, tg = lv(tr, "entry"), lv(tr, "stop"), lv(tr, "target"); risk_amt = RISK * equity
        res.update(direction=d(tr), entry=en, stop=sp_, target=tg, reward_to_risk=tr["rr"], risk_amount=risk_amt, size_units=risk_amt / abs(en - sp_))
    if acc:
        tr = acc[-1]; ex = tr["ex"]
        if not ex["open"] and ex["t_exit"] > t - pd.Timedelta("15min"):
            res.update(decision="EXIT", direction=d(tr), reason_code=ex["code"]); fill_levels(tr)
            res["explanation"] = f"Close the {d(tr)} position: {EXIT_TEXT[ex['code']]}. Result about {ex['R']:+.2f}R."; return res
        if ex["open"]:
            fill_levels(tr); rn = f" Open result about {ex['R_now']:+.2f}R." if "R_now" in ex else ""
            res.update(decision="WAIT", direction=d(tr)); res["explanation"] = f"Holding the {d(tr)} position opened {pd.Timestamp(tr['t0']):%d %b %H:%M}. Stop {f(res['stop'])}, target {f(res['target'])}.{rn}"; return res
    now = [x for x in other if x["kind"] == "enter_now"]
    if now:
        tr = now[0]; fill_levels(tr); res.update(decision="ENTER")
        how = ("Enter at the next candle's open (estimated " + f(res["entry"]) + ")" if sel["mode"] == "momentum" else "Price retested the entry zone on the last completed candle; entry level " + f(res["entry"]))
        res["explanation"] = f"{d(tr)}: {how}. Stop {f(res['stop'])}, target {f(res['target'])}, reward-to-risk {tr['rr']:.1f}:1. Risk 1% = {res['risk_amount']:.2f}, size = {res['size_units']:.4f} index units."; return res
    rank = {"pending": 3, "await_trigger": 2, "await_touch": 1}; act = [x for x in other if x["kind"] in rank]
    if act:
        tr = max(act, key=lambda x: (rank[x["kind"]], pd.Timestamp(x["act"]))); lo, hi = sorted([tr["m"] * tr["zl"], tr["m"] * tr["zh"]]); extra = f" ({len(act) - 1} other active setups.)" if len(act) > 1 else ""
        res.update(decision="SETUP", direction=d(tr))
        if tr["kind"] == "pending":
            fill_levels(tr); res["explanation"] = f"{d(tr)} confirmed. Pending entry {f(res['entry'])} on a retest of the entry zone, stop {f(res['stop'])}, target {f(res['target'])}, reward-to-risk {tr['rr']:.1f}:1. Valid until a candle closes beyond the zone or 24 hours pass.{extra}"
        elif tr["kind"] == "await_trigger": res["explanation"] = f"{d(tr)} setup: price is inside the zone {f(lo)} to {f(hi)}. Waiting for a 15-minute close beyond the latest swing {'high' if tr['m'] == 1 else 'low'} to confirm.{extra}"
        else: res["explanation"] = f"{d(tr)} setup: four-hour zone {f(lo)} to {f(hi)}, target {f(tr['m'] * tr['target'])}. Waiting for price to trade into the zone.{extra}"
        return res
    rec = [x for x in other if x["kind"] == "closed" and x["at"] == x["last"]]
    if rec: tr = rec[-1]; res.update(decision="NO TRADE", direction=d(tr)); res["explanation"] = f"{d(tr)} setup rejected: {tr['why']}."; return res
    res["explanation"] = "No active setup. Waiting for a new four-hour zone to form."; return res

def latest_guidance(code, path=None, equity=START_EQ):
    """Decision at the last completed 15-minute candle. Pass the path of a fresh M1 export to use newer data."""
    d_ = D[code]["df"] if path is None else load(path)
    return guidance(d_, (d_.index[-1] + pd.Timedelta("1min")).floor("15min"), code, equity)

def show_guidance(g):
    print(f"[{g['time']:%d %b %Y %H:%M}] {g['instrument']} | {g['strategy']} ({g['settings']})\n  {g['decision']}" + (f" ({g['direction']})" if g["direction"] != "-" else "") + f": {g['explanation']}")
