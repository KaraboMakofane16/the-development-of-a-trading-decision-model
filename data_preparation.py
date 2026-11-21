# ==================== COMPONENT 1: DATA PREPARATION ====================
# Loads the MT5 one-minute export and builds the higher timeframes from it.
# Uses constants defined in the project settings (not included here).

def load(path):
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["dt"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
    n_raw = len(df)
    df = df.drop_duplicates("dt").sort_values("dt").set_index("dt")
    df.attrs["n_raw"] = n_raw
    return df[["open", "high", "low", "close", "tickvol", "vol", "spread"]]

def rs(df, rule, **kw):
    """Resample one-minute candles to a larger timeframe."""
    return df.resample(rule, **kw).agg({"open": "first", "high": "max", "low": "min", "close": "last", "spread": "mean"}).dropna()

def build_tfs(df):
    return {"1m": df, "5m": rs(df, "5min"), "15m": rs(df, "15min"), "30m": rs(df, "30min"), "H4": rs(df, "4h"),
            "D1": rs(df, "1D"), "W1": rs(df, "W-SUN", closed="left", label="left")}

def infer_point(prices):
    """Detect the size of one point from the price decimals (V10 = 0.001)."""
    s = prices.dropna().values[:200000]
    for d in range(0, 8):
        if np.abs(np.round(s, d) - s).max() < 1e-8: return 10.0 ** -d
    return 1e-8
