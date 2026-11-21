# ==================== COMPONENT 3: STRATEGY ENGINE ====================
# Turns an H4 setup into a trade (15-minute confirmation, entry, stop, target),
# simulates the exit on one-minute data, and runs a strategy over the full history.

def ltf_candidate(V, st, mode, n, buf):
    """15-minute confirmation: zone touch, close beyond latest swing high, then entry/stop/target."""
    o, h, l, c = V.L; N = len(h); sp = V.Lsp; sh15 = V.sw15(n)
    j = int(np.searchsorted(V.Lt, st["act"])); zl, zh = st["zl"], st["zh"]; touch = None
    for jj in range(j, min(j + EXP_A, N - 2)):
        if st["inval"] and V.kmap[jj] > st["inval"][1] and V.H[3][V.kmap[jj]] < st["inval"][0]: return None
        if c[jj] < zl: return None
        if l[jj] <= zh: touch = jj; break
    if touch is None: return None
    trig = None
    for jj in range(touch, min(touch + EXP_B, N - 2)):
        if c[jj] < zl: return None
        lo_i = np.searchsorted(sh15, touch - 3, side="left"); hi_i = np.searchsorted(sh15, jj - n, side="right")
        if hi_i > lo_i and c[jj] > h[sh15[hi_i - 1]]: trig = (jj, sh15[hi_i - 1]); break
    if trig is None: return None
    jj, ish = trig
    if mode == "momentum":
        e = jj + 1; entry = o[e] + sp[e]; stop = l[touch:jj + 1].min() - buf * sp[e]; t_scan = V.Lt[e]
    else:
        Lidx = ish + int(np.argmin(l[ish:jj + 1])); ob = find_ob(o, c, h, l, Lidx + 1, Lidx - 6)
        if ob is None: return None
        zl2, zh2 = l[ob], h[ob]; fill = None
        for j2 in range(jj + 1, min(jj + 1 + EXP_C, N - 1)):
            if c[j2] < zl2: return None
            if l[j2] <= zh2: fill = j2; break
        if fill is None: return None
        entry = min(zh2, o[fill]) + sp[fill]; stop = zl2 - buf * sp[fill]; t_scan = V.Lt[fill] + np.timedelta64(15, "m")
    tgt = st["target"]
    if not (entry > stop and tgt > entry) or (tgt - entry) / (entry - stop) < MIN_RR: return None
    return dict(t0=t_scan, entry=entry, stop=stop, target=tgt, hl=st["hl"])

def simulate_exit(V, c0, rev=True):
    """Walks forward on one-minute data until stop, target, H4 reversal or the holding limit. Returns the result in R."""
    s = int(np.searchsorted(V.Mt, c0["t0"])); e = min(s + CAP, len(V.Mt))
    if s >= len(V.Mt) - 1: return None
    lo, hi = V.M[2][s:e], V.M[1][s:e]; big = 10**9
    sm, tm = lo <= c0["stop"], hi >= c0["target"]
    si = int(np.argmax(sm)) if sm.any() else big; ti = int(np.argmax(tm)) if tm.any() else big; ri = big
    if rev and c0["hl"] is not None:
        i0 = int(np.searchsorted(V.Hend, c0["t0"])); w = np.where(V.H[3][i0:] < c0["hl"])[0]
        if len(w): ri = max(int(np.searchsorted(V.Mt, V.Hend[i0 + w[0]])) - s, 0); rpx = V.H[3][i0 + w[0]]
    risk = c0["entry"] - c0["stop"]
    if si < big and si <= ti and si <= ri: R, why, k = -1.0, "stop", si
    elif ti < big and ti < ri: R, why, k = (c0["target"] - c0["entry"]) / risk, "target", ti
    elif ri < big: R, why, k = (rpx - c0["entry"]) / risk, "htf_reversal", min(ri, e - s - 1)
    else: R, why, k = (V.M[3][e - 1] - c0["entry"]) / risk, "open_at_end", e - s - 1
    return dict(R=R, why=why, t_exit=V.Mt[min(s + k, len(V.Mt) - 1)] + np.timedelta64(1, "m"))

def run_strategy(views, mode, n, buf_mult):
    """Back-test one strategy and setting over the full history (buys and sells), one position at a time."""
    rows = []
    for V in views:
        for st in V.setups(mode):
            c0 = ltf_candidate(V, st, mode, n, buf_mult)
            if c0 is None: continue
            ex = simulate_exit(V, c0)
            if ex is None: continue
            m = V.m
            rows.append(dict(entry_time=pd.Timestamp(c0["t0"]), direction="buy" if m == 1 else "sell", entry=m * c0["entry"], stop=m * c0["stop"],
                             target=m * c0["target"], exit_time=pd.Timestamp(ex["t_exit"]), exit_reason=ex["why"], R=ex["R"]))
    cols = ["entry_time", "direction", "entry", "stop", "target", "exit_time", "exit_reason", "R"]
    tr = pd.DataFrame(rows, columns=cols).sort_values("entry_time").reset_index(drop=True)
    keep, last = [], pd.Timestamp.min
    for i, r_ in tr.iterrows():
        if r_.entry_time >= last: keep.append(i); last = r_.exit_time
    return tr.loc[keep].reset_index(drop=True)
