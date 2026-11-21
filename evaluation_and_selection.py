# ==================== COMPONENT 4: PERFORMANCE EVALUATION AND STRATEGY SELECTION ====================
# Profitability and risk metrics, in-sample selection, and the random-entry benchmark.

def metrics(R, start=START_EQ):
    """Net profit, maximum drawdown and return-to-drawdown from a list of trade results in R (compounded equity)."""
    R = np.asarray(R, float); eq = [start]
    for x in R: eq.append(eq[-1] * (1 + RISK * x))
    eq = np.array(eq); peak = np.maximum.accumulate(eq); dd = ((peak - eq) / peak).max() * 100; net = (eq[-1] - start) / start * 100
    streak = best = 0
    for x in R:
        streak = streak + 1 if x < 0 else 0; best = max(best, streak)
    return dict(trades=len(R), win_rate=float((R > 0).mean() * 100) if len(R) else np.nan, net_profit_pct=net, final_equity=eq[-1], max_dd_pct=dd,
                ret_dd=(net / dd) if dd > 0 else (np.inf if net > 0 else np.nan), avg_R=float(R.mean()) if len(R) else np.nan,
                longest_loss_streak=best, low_confidence=len(R) < 30), eq

def split(tr):
    """In-sample (before the split date) and out-of-sample (from the split date), by entry time."""
    return tr[tr.entry_time < SPLIT], tr[tr.entry_time >= SPLIT]

def score(tr):
    """The fixed selection criterion, on in-sample trades only: return-to-drawdown ratio, at least 5 trades."""
    m, _ = metrics(split(tr)[0].R)
    if m["trades"] < 5: return -np.inf
    return m["ret_dd"] if np.isfinite(m["ret_dd"]) else m["net_profit_pct"]

def select_overall(runs):
    """Choose ONE strategy and its settings from in-sample data only (highest score; ties go to the first in the fixed order)."""
    best = None
    for s in STRATS:
        for g in GRID:
            sc = score(runs[s][g])
            if np.isfinite(sc) and (best is None or sc > best["score"]): best = dict(strategy=s, mode=STRATS[s], n=g[0], buf=g[1], score=float(sc), in_sample_trades=len(split(runs[s][g])[0]))
    return best

def random_benchmark(views, tr_oos, sims=100):
    """Random entries with the same stop distances and 3:1 target, out-of-sample, to test the result against chance."""
    V0, V1 = views; rng = np.random.default_rng(SEED)
    idx = np.where(V0.Lt >= np.datetime64(SPLIT))[0]; idx = idx[idx < len(V0.Lt) - 200]
    dists = (tr_oos.entry - tr_oos.stop).abs().values
    if len(dists) == 0: return None
    res = []
    for _ in range(sims):
        Rs = []
        for _ in range(len(dists)):
            V = V0 if rng.random() < .5 else V1; j = rng.choice(idx); d = rng.choice(dists); entry = V.L[0][j] + V.Lsp[j]
            ex = simulate_exit(V, dict(t0=V.Lt[j], entry=entry, stop=entry - d, target=entry + MIN_RR * d, hl=None), rev=False)
            if ex: Rs.append(ex["R"])
        res.append((np.mean(np.array(Rs) > 0) * 100, np.mean(Rs)))
    a = np.array(res)
    return dict(win_rate_mean=a[:, 0].mean(), win_rate_p5=np.percentile(a[:, 0], 5), win_rate_p95=np.percentile(a[:, 0], 95), avgR_mean=a[:, 1].mean(), avgR_p95=np.percentile(a[:, 1], 95))
