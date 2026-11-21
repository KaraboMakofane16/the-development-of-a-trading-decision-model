# ==================== COMPONENT 2: MARKET STRUCTURE ====================
# Swing points, order blocks, and the multi-timeframe view that sets the
# directional bias (Dow Theory: HH/HL = up, LH/LL = down) and creates setups.

def swings(h, l, n):
    """Swing highs/lows: a candle that is the unique extreme of the 2n+1 candles around it."""
    w = 2 * n + 1
    if len(h) < w: return np.array([], int), np.array([], int)
    Wh, Wl = sliding_window_view(h, w), sliding_window_view(l, w)
    ch, cl = h[n:len(h) - n], l[n:len(l) - n]
    sh = np.where((ch == Wh.max(1)) & ((Wh == ch[:, None]).sum(1) == 1))[0] + n
    sl = np.where((cl == Wl.min(1)) & ((Wl == cl[:, None]).sum(1) == 1))[0] + n
    return sh, sl

def find_ob(o, c, h, l, hi_idx, lo_idx):
    """Order block: the last bearish candle found scanning back from hi_idx to lo_idx."""
    for i in range(hi_idx, max(lo_idx, 0) - 1, -1):
        if c[i] < o[i]: return i
    return None

class View:
    """Long-only view. mirror=True negates prices so shorts reuse the long logic."""
    def __init__(self, tfs, mirror, point):
        self.m = -1 if mirror else 1
        def a(d):
            o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
            return (-o, -l, -h, -c) if mirror else (o, h, l, c)
        self.M = a(tfs["1m"]); self.Mt = tfs["1m"].index.values
        self.H = a(tfs["H4"]); self.Ht = tfs["H4"].index.values; self.Hend = self.Ht + np.timedelta64(4, "h")
        self.L = a(tfs["15m"]); self.Lt = tfs["15m"].index.values; self.Lsp = tfs["15m"]["spread"].values * point
        self.kmap = np.searchsorted(self.Hend, self.Lt, side="right") - 1
        self._cache = {}
        self._htf()
    def sw15(self, n):
        if n not in self._cache: self._cache[n] = swings(self.L[1], self.L[2], n)[0]
        return self._cache[n]
    def _htf(self):
        """Higher-timeframe (H4) bias from the sequence of swing highs and lows."""
        o, h, l, c = self.H; n = N_H
        sh, sl = swings(h, l, n); K = len(h)
        self.bias = np.zeros(K, int); self.info = [None] * K
        for k in range(K):
            a, b = sh[:np.searchsorted(sh, k - n, side="right")], sl[:np.searchsorted(sl, k - n, side="right")]
            if len(a) >= 2 and len(b) >= 2:
                if h[a[-1]] > h[a[-2]] and l[b[-1]] > l[b[-2]]: self.bias[k] = 1
                elif h[a[-1]] < h[a[-2]] and l[b[-1]] < l[b[-2]]: self.bias[k] = -1
                self.info[k] = (a[-1], b[-1], a[-2])
    def setups(self, mode):
        if mode not in self._cache: self._cache[mode] = self._make_setups(mode)
        return self._cache[mode]
    def _make_setups(self, mode):
        """H4 setups: order-block zone and target for trend/momentum, or range breakout for breakout-and-retest."""
        o, h, l, c = self.H; out_ = []; used = set()
        for k in range(len(h) - 1):
            if self.info[k] is None: continue
            ih, il1, ih2 = self.info[k]
            if mode in ("trend", "momentum") and self.bias[k] == 1 and ih not in used:
                used.add(ih)
                prev_lows = [i for i in range(max(ih - 60, 0), ih) if l[i] == l[max(i - N_H, 0):i + N_H + 1].min()]
                if not prev_lows: continue
                il = prev_lows[-1]
                ob = find_ob(o, c, h, l, min(il + 1, k), il - 3)
                if ob is None: continue
                out_.append(dict(zl=l[ob], zh=h[ob], target=h[ih], hl=l[il1], act=self.Hend[k], inval=None))
            if mode == "breakout" and self.bias[k] == 0:
                rh, rl = h[ih], l[il1]
                if rh > rl and c[k + 1] > rh and rh not in used:
                    used.add(rh)
                    ob = find_ob(o, c, h, l, k, k - 6)
                    if ob is None: continue
                    out_.append(dict(zl=l[ob], zh=h[ob], target=rh + (rh - rl), hl=rh, act=self.Hend[k + 1], inval=(rh, k + 1)))
        return out_
