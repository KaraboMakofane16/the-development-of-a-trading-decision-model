# ==================== COMPONENT 6: VALIDATION OF THE GUIDANCE MODEL ====================
# Evaluation on the unseen period only. Nothing here is used to select or tune the strategy.
# Checks: (1) ENTER matches the back-test, (2) EXIT matches the back-test,
# (3) look-ahead test: decisions do not change when all later candles are scrambled.

def validate_guidance(code, max_checks=15):
    sel = SEL.get(code)
    if not sel: print(f"{code}: no strategy was selected, so there is nothing to validate."); return
    df = D[code]["df"]; O = out(code)
    tr = pd.read_csv(f"{O}/trades_{sel['strategy']}.csv", parse_dates=["entry_time", "exit_time"]); oos = tr[(tr.entry_time >= SPLIT) & (tr.exit_time < df.index[-1] - pd.Timedelta("1D"))].reset_index(drop=True)
    print(f"\n{code}: {LABEL[sel['strategy']]} | {len(oos)} out-of-sample trades available, checking {min(len(oos), max_checks)}")
    rows = []
    for i in np.unique(np.linspace(0, len(oos) - 1, min(len(oos), max_checks)).astype(int)) if len(oos) else []:
        r_ = oos.iloc[i]; g = guidance(df, r_.entry_time, code); te = r_.exit_time.ceil("15min"); g2 = guidance(df, te, code)
        win = df[df.index >= r_.entry_time - pd.Timedelta(days=WINDOW_DAYS + 5)].copy(); fut = win.index >= r_.entry_time   # look-ahead test: scramble every candle from the decision time onward
        win.loc[fut, ["open", "high", "low", "close"]] = win.loc[fut, ["open", "high", "low", "close"]].values[::-1] * 1.7
        g3 = guidance(win, r_.entry_time, code)
        same = (g3["decision"] == g["decision"]) and (g3["entry"] is None or np.isclose(g3["entry"], g["entry"])) and (g3["stop"] is None or np.isclose(g3["stop"], g["stop"]))
        rows.append(dict(entry_time=r_.entry_time, backtest_dir=r_.direction.upper(), guidance_entry=g["decision"], dir_ok=g["direction"] == r_.direction.upper(),
                         target_ok=g["target"] is not None and np.isclose(g["target"], r_.target), stop_gap=abs(g["stop"] - r_.stop) if g["stop"] is not None else np.nan,
                         exit_reason=r_.exit_reason, guidance_exit=g2["decision"], exit_ok=(g2["decision"] == "EXIT" and g2["reason_code"] == r_.exit_reason), no_lookahead=same))
    V_ = pd.DataFrame(rows)
    if len(V_):
        display(V_)
        print(f"{code}: ENTER matched the back-test on {int(((V_.guidance_entry == 'ENTER') & V_.dir_ok & V_.target_ok).sum())}/{len(V_)} | EXIT matched on {int(V_.exit_ok.sum())}/{len(V_)} | unchanged when future candles were scrambled: {int(V_.no_lookahead.sum())}/{len(V_)}")
        print("Largest stop-level gap (momentum uses an estimated entry because the next open is not yet known):", V_.stop_gap.max())
    rng = np.random.default_rng(1); span = int((df.index[-1] - SPLIT) / pd.Timedelta("15min")) - 96
    times = [SPLIT + pd.Timedelta("15min") * int(k) for k in rng.integers(0, span, 40)]
    states = pd.Series([guidance(df, t, code)["decision"] for t in times]).value_counts()
    print(f"{code}: decisions at 40 random out-of-sample 15-minute candles:", states.to_dict())
