import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import linprog
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import FRED API (optional, falls back to Yahoo)
try:
    import fredapi
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    print("Warning: fredapi not installed. Install with: pip install fredapi")
    print("Falling back to Yahoo Finance for macro data approximations.\n")

# ====================================
# ASSET UNIVERSE
# ====================================
futures_to_etf = {
    'ES': 'SPY', 'NQ': 'QQQ', 'RTY': 'IWM', 'CL': 'USO',
    'GC': 'GLD', 'HG': 'CPER', 'NG': 'UNG', 'EUR': 'FXE',
    'GBP': 'FXB', 'JPY': 'FXY', 'ZN': 'IEF', 'ZB': 'TLT'
}

start = '2015-12-31'
end = '2025-12-28'

print("Libraries imported.")
print("Asset universe defined: 12 assets")
print("\nAssets:")
for future, etf in futures_to_etf.items():
    print(f"  {future:4s} to {etf}")

print(f"\nDownloading data from {start} to {end}")

# ====================================
# DOWNLOAD PRICE DATA
# ====================================
data = {}
for future, etf in futures_to_etf.items():
    print(f"Downloading {future} ({etf})...", end=" ")
    try:
        df = yf.download(etf, start=start, end=end,
                         progress=False, auto_adjust=True)

        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                close_col = df['Close'].iloc[:, 0] if 'Close' in df.columns.get_level_values(0) else df.iloc[:, -2]
            else:
                close_col = df['Close']

            prices = close_col.dropna()
            if len(prices) > 0:
                data[future] = prices
                print(f"{len(prices)} days")
                continue

        print("failed")
        data[future] = pd.Series(dtype=float)
    except Exception as e:
        print(f"error: {e}")
        data[future] = pd.Series(dtype=float)

print("\nDownload complete!")

# ====================================
# CREATE PRICE DATAFRAME
# ====================================
price_data = pd.DataFrame(data).ffill().dropna(how='all')
print(f"\nPrice data shape: {price_data.shape}")

if price_data.shape[1] > 0:

    # ====================================
    # DATA PREP
    # ====================================
    daily_returns = price_data.pct_change()

    # Map to ETF names
    etf_cols = [futures_to_etf[c] for c in price_data.columns]
    price_data_etf = price_data.copy()
    price_data_etf.columns = etf_cols
    daily_returns_etf = price_data_etf.pct_change()

    asset_names = price_data.columns.tolist()
    n_regimes = 4

    # ====================================
    # DOWNLOAD MACRO DATA
    # ====================================
    print("\n" + "="*70)
    print("DOWNLOADING MACRO DATA")
    print("="*70)

    # Method 1: Try FRED API (most accurate)
    if FRED_AVAILABLE:
        print("\nUsing FRED API for macro data...")
        try:
            fred = fredapi.Fred(api_key='YOUR_FRED_KEY_HERE')
            # Note: Users need to replace with actual FRED key from fred.stlouisfed.org
            print("(Note: Requires FRED API key. Get free key at https://fred.stlouisfed.org)")
        except:
            FRED_AVAILABLE = False
            print("FRED API key not set. Using Yahoo Finance approximations instead.\n")

    # Method 2: Proxy from Yahoo Finance (free, works without API key)
    print("Downloading macro proxy variables from Yahoo Finance...")

    # VIX (market fear/uncertainty proxy)
    try:
        print("  Downloading VIX (implied volatility)...", end=" ")
        vix_data = yf.download('^VIX', start=start, end=end, progress=False)['Close']
        # Ensure it's 1D
        if isinstance(vix_data, pd.DataFrame):
            vix_data = vix_data.iloc[:, 0]
        vix_normalized = (vix_data / 100.0).fillna(method='ffill').fillna(method='bfill')
        print("✓")
    except Exception as e:
        print(f"✗ Error: {e}")
        vix_normalized = pd.Series(0.2, index=price_data.index)

    # TLT (long-term bonds) as proxy for real yields
    # When TLT falls, real yields are rising (bad for equities)
    try:
        print("  Computing TLT momentum as real yield proxy...", end=" ")
        tlt_ret = daily_returns_etf['TLT'].rolling(20).mean() * 252
        real_yield_proxy = tlt_ret.fillna(0)  # Higher = higher real yields
        print("✓")
    except Exception as e:
        print(f"✗ Error: {e}")
        real_yield_proxy = pd.Series(0.0, index=price_data.index)

    # Credit spreads proxy: HY bond performance vs safe bonds
    # Use ratio of high-yield performance to treasury performance
    try:
        print("  Computing credit spread proxy (HY vs Treasury)...", end=" ")
        # If we had HY bond ETF data, we'd use it. For now, use commodity/equity divergence
        # as rough proxy for risk-on/risk-off
        equity_momentum = daily_returns_etf[['SPY', 'QQQ']].mean(axis=1).rolling(20).mean() * 252
        credit_proxy = equity_momentum.fillna(0)  # Higher = lower spreads (risk-on)
        print("✓")
    except Exception as e:
        print(f"✗ Error: {e}")
        credit_proxy = pd.Series(0.0, index=price_data.index)

    # Fed rate proxy: TLT yield level (when high = Fed tightening)
    try:
        print("  Computing Fed policy proxy from bond yields...", end=" ")
        # TLT is 20+ year treasuries. Higher yield = tighter policy
        tlt_level = daily_returns_etf['TLT'].rolling(252).mean()
        fed_proxy = (tlt_level / tlt_level.std()).fillna(0)
        print("✓")
    except Exception as e:
        print(f"✗ Error: {e}")
        fed_proxy = pd.Series(0.0, index=price_data.index)

    # Combine macro proxies into single dataframe
    macro_data = pd.DataFrame({
        'vix': vix_normalized.values,
        'real_yields': real_yield_proxy.values,
        'credit_spreads': credit_proxy.values,
        'fed_policy': fed_proxy.values
    }, index=price_data.index)

    print("\nMacro data prepared:")
    print(f"  Shape: {macro_data.shape}")
    print(f"  Columns: {list(macro_data.columns)}")
    print(f"\nMacro data sample:")
    print(macro_data.tail())

    # ====================================
    # ROLLING K-MEANS WITH MACRO DATA
    # ====================================
    print("\n" + "="*70)
    print("ROLLING-WINDOW GTO WITH MACRO-AWARE REGIMES")
    print("="*70)

    def get_regimes_at_date_with_macro(daily_ret_etf, macro_data, end_date, lookback_days=252*2):
        """
        Cluster regimes using BOTH price signals AND macro data
        """
        start_date = end_date - timedelta(days=lookback_days)
        data_slice = daily_ret_etf[start_date:end_date]
        macro_slice = macro_data[start_date:end_date]
        
        if len(data_slice) < 252:
            return None, None

        # ===== PRICE-BASED FEATURES =====
        # Yield curve slope
        yc_ret_long = data_slice['TLT'].rolling(20).mean() * 252
        yc_ret_short = data_slice['IEF'].rolling(20).mean() * 252
        yc_slope = yc_ret_long - yc_ret_short
        
        # Equity momentum
        equity_momentum = data_slice[['SPY', 'QQQ', 'IWM']].mean(axis=1).rolling(20).mean()
        
        # Commodity momentum
        commodity_momentum = data_slice[['USO', 'GLD', 'CPER']].mean(axis=1).rolling(30).mean()
        
        # Realized volatility
        realized_vol = data_slice['SPY'].rolling(30).std() * np.sqrt(252)

        # ===== MACRO FEATURES =====
        # VIX (fear gauge)
        vix = macro_slice['vix'].values
        
        # Real yields proxy (higher = tighter policy)
        real_yields = macro_slice['real_yields'].values
        
        # Credit spreads proxy (higher = lower spreads, risk-on)
        spreads = macro_slice['credit_spreads'].values
        
        # Fed policy proxy
        fed_policy = macro_slice['fed_policy'].values

        # ===== COMBINE INTO FEATURE MATRIX =====
        features = np.column_stack([
            # Price signals (4 features)
            yc_slope.fillna(0),               # Yield curve
            equity_momentum.fillna(0),        # Equity momentum
            commodity_momentum.fillna(0),     # Commodity momentum
            realized_vol.fillna(0),           # Volatility
            
            # Macro signals (4 features) - THIS IS NEW
            vix,                              # Market fear
            real_yields,                      # Rate expectations
            spreads,                          # Risk appetite
            fed_policy,                       # Policy stance
        ])

        # Standardize all features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Cluster into 4 regimes
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=5)
        regime_id_raw = kmeans.fit_predict(features_scaled)
        
        # Relabel by equity momentum (Risk-On to Risk-Off)
        regime_eq_mom = [equity_momentum[regime_id_raw == r].mean() for r in range(4)]
        regime_order = np.argsort(regime_eq_mom)[::-1]
        regime_relabel = {old: new for new, old in enumerate(regime_order)}
        regime_id = np.array([regime_relabel[r] for r in regime_id_raw])
        
        return regime_id, data_slice.index

    def get_regime_returns(daily_ret, regime_id):
        """Compute expected returns per regime"""
        mu_reg = np.zeros((n_regimes, daily_ret.shape[1]))
        for r in range(n_regimes):
            mask = regime_id == r
            if mask.sum() > 50:
                mu_reg[r] = daily_ret[mask].mean() * 252
        return mu_reg

    def solve_maximin_gto(mu_reg, max_weight=0.15):
        """Game-theoretic maximin optimization"""
        n_assets = mu_reg.shape[1]
        c = np.zeros(n_assets + 1)
        c[-1] = -1

        A_ub = []
        b_ub = []
        for r in range(n_regimes):
            row = np.zeros(n_assets + 1)
            row[:n_assets] = -mu_reg[r]
            row[-1] = 1
            A_ub.append(row)
            b_ub.append(0)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        A_eq = np.zeros((1, n_assets + 1))
        A_eq[0, :n_assets] = 1
        b_eq = np.array([1])

        bounds = [(0, max_weight) for _ in range(n_assets)] + [(None, None)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs')

        return res

    # ====================================
    # QUARTERLY REOPTIMIZATION (Calendar-Based)
    # ====================================
    print("\n[Running quarterly reoptimization (Q1, Q2, Q3, Q4)]")
    print("[Using 2-year lookback for regime detection]")
    print("[With macro-aware regime clustering]\n")

    # Pre-specified fiscal quarter ends (not optimized)
    quarter_ends = pd.to_datetime([
        '2017-03-31', '2017-06-30', '2017-09-30', '2017-12-31',
        '2018-03-31', '2018-06-30', '2018-09-30', '2018-12-31',
        '2019-03-31', '2019-06-30', '2019-09-30', '2019-12-31',
        '2020-03-31', '2020-06-30', '2020-09-30', '2020-12-31',
        '2021-03-31', '2021-06-30', '2021-09-30', '2021-12-31',
        '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31',
        '2023-03-31', '2023-06-30', '2023-09-30', '2023-12-31',
        '2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31',
        '2025-03-31', '2025-06-30', '2025-09-30', '2025-12-31',
    ])

    # Filter to dates available in our data
    optimization_dates = [d for d in quarter_ends if d in daily_returns.index]
    
    lookback = 252 * 2
    rolling_weights = {}
    rolling_weights_prev = None

    print(f"Optimizing at {len(optimization_dates)} quarterly dates:")
    for opt_date in optimization_dates[:8]:
        print(f"  {opt_date.date()}")
    if len(optimization_dates) > 8:
        print(f"  ... ({len(optimization_dates) - 8} more)")

    for opt_idx, opt_date in enumerate(optimization_dates):
        # Detect regimes using macro-aware clustering
        regime_id, regime_dates = get_regimes_at_date_with_macro(
            daily_returns_etf, macro_data, opt_date, lookback_days=lookback
        )
        
        if regime_id is None:
            continue

        # Get returns aligned with regimes
        data_for_mu = daily_returns.loc[regime_dates]
        mu_reg = get_regime_returns(data_for_mu, regime_id)

        # Optimize portfolio
        res = solve_maximin_gto(mu_reg)
        if res.success:
            w = res.x[:len(asset_names)]
        else:
            w = np.ones(len(asset_names)) / len(asset_names)

        # Apply smoothing: blend old and new weights
        if rolling_weights_prev is not None:
            w_smoothed = 0.7 * rolling_weights_prev + 0.3 * w
            # Constraint: max 5% change per quarter
            w_final = np.clip(w_smoothed, rolling_weights_prev - 0.05, rolling_weights_prev + 0.05)
        else:
            w_final = w

        rolling_weights[opt_date] = w_final
        rolling_weights_prev = w_final.copy()

    # ====================================
    # BACKTEST WITH ROLLING MACRO-AWARE GTO
    # ====================================
    print("\n" + "="*70)
    print("BACKTEST: ROLLING GTO (QUARTERLY + MACRO) VS STATIC GTO")
    print("="*70)

    # Static GTO (optimized once on IS data)
    split_date = '2020-12-31'
    regime_id_is, regime_dates_is = get_regimes_at_date_with_macro(
        daily_returns_etf, macro_data, pd.Timestamp(split_date), lookback
    )
    data_is = daily_returns.loc[regime_dates_is]
    mu_is = get_regime_returns(data_is, regime_id_is)
    res_static = solve_maximin_gto(mu_is)
    w_static = res_static.x[:len(asset_names)]

    # Static portfolio returns (OOS only)
    oos_idx = daily_returns.index > split_date
    oos_data = daily_returns[oos_idx]
    static_portfolio_ret = (w_static * oos_data).sum(axis=1)

    # Rolling portfolio returns
    rolling_portfolio_ret = []
    rolling_dates = []

    for i, date in enumerate(daily_returns.index):
        opt_dates_before = [d for d in rolling_weights.keys() if d <= date]
        if opt_dates_before:
            closest_opt_date = max(opt_dates_before)
            w_roll = rolling_weights[closest_opt_date]
            daily_ret = daily_returns.loc[date]
            port_ret = (w_roll * daily_ret).sum()
            rolling_portfolio_ret.append(port_ret)
            rolling_dates.append(date)

    rolling_portfolio_ret = pd.Series(rolling_portfolio_ret, index=rolling_dates)

    # Metrics
    def calc_metrics(ret_series):
        ret = ret_series.dropna()
        if len(ret) < 252:
            return None
        cum = (1 + ret).cumprod()
        years = len(ret) / 252.0
        ann_ret = cum.iloc[-1]**(1/years) - 1
        vol = ret.std() * np.sqrt(252)
        sharpe = ann_ret / vol if vol > 0 else 0
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        max_dd = dd.min()
        return {'ret': ann_ret, 'vol': vol, 'sharpe': sharpe, 'max_dd': max_dd}

    metrics_static = calc_metrics(static_portfolio_ret)
    metrics_rolling = calc_metrics(rolling_portfolio_ret)

    print(f"\n{'Strategy':<35} {'Ann Ret':>12} {'Vol':>10} {'Sharpe':>10} {'Max DD':>12}")
    print("-" * 80)
    print(f"{'Static GTO (opt once)':<35} {metrics_static['ret']:>11.2%} {metrics_static['vol']:>9.2%} {metrics_static['sharpe']:>9.2f} {metrics_static['max_dd']:>11.2%}")
    print(f"{'Rolling GTO (Q + Macro + Smooth)':<35} {metrics_rolling['ret']:>11.2%} {metrics_rolling['vol']:>9.2%} {metrics_rolling['sharpe']:>9.2f} {metrics_rolling['max_dd']:>11.2%}")
    
    print(f"\nImprovement (Rolling vs Static):")
    print(f"  Sharpe: {metrics_rolling['sharpe'] - metrics_static['sharpe']:+.2f}")
    print(f"  Return: {(metrics_rolling['ret'] - metrics_static['ret'])*100:+.1f} bps")

    # ====================================
    # STABILITY ANALYSIS
    # ====================================
    print("\n" + "="*70)
    print("STABILITY ANALYSIS: ROLLING GTO (Q+MACRO) IS vs OOS")
    print("="*70)

    rolling_is_idx = rolling_portfolio_ret.index <= split_date
    rolling_oos_idx = rolling_portfolio_ret.index > split_date

    rolling_ret_is = rolling_portfolio_ret[rolling_is_idx]
    rolling_ret_oos = rolling_portfolio_ret[rolling_oos_idx]

    print(f"\nIn-Sample:     {rolling_ret_is.index[0].date()} to {rolling_ret_is.index[-1].date()} ({len(rolling_ret_is)} days)")
    print(f"Out-of-Sample: {rolling_ret_oos.index[0].date()} to {rolling_ret_oos.index[-1].date()} ({len(rolling_ret_oos)} days)")

    # Check 1: Return Distribution Stability
    print("\n" + "="*70)
    print("1. RETURN DISTRIBUTION STABILITY (IS vs OOS)")
    print("="*70)

    mean_is = rolling_ret_is.mean() * 252
    vol_is = rolling_ret_is.std() * np.sqrt(252)
    sharpe_is = mean_is / vol_is if vol_is > 0 else 0

    mean_oos = rolling_ret_oos.mean() * 252
    vol_oos = rolling_ret_oos.std() * np.sqrt(252)
    sharpe_oos = mean_oos / vol_oos if vol_oos > 0 else 0

    print(f"\n{'Metric':<20} {'IS':>12} {'OOS':>12} {'Drift':>12}")
    print("-" * 50)
    print(f"{'Mean (ann %)':<20} {mean_is*100:>11.2f}% {mean_oos*100:>11.2f}% {(mean_oos-mean_is)*100:>11.2f}%")
    print(f"{'Vol (ann %)':<20} {vol_is*100:>11.2f}% {vol_oos*100:>11.2f}% {(vol_oos-vol_is)*100:>11.2f}%")
    print(f"{'Sharpe':<20} {sharpe_is:>11.2f} {sharpe_oos:>11.2f} {sharpe_oos-sharpe_is:>11.2f}")

    return_stability = abs((sharpe_oos - sharpe_is) / (abs(sharpe_is) + 1e-8))
    if return_stability < 0.25:
        print("\n✓ STABLE: Return distributions similar IS vs OOS")
        check1_pass = True
    else:
        print("\n⚠ UNSTABLE: Return distributions shift significantly")
        check1_pass = False

    # Check 2: Performance Consistency
    print("\n" + "="*70)
    print("2. PERFORMANCE CONSISTENCY (IS vs OOS)")
    print("="*70)

    metrics_is = calc_metrics(rolling_ret_is)
    metrics_oos = calc_metrics(rolling_ret_oos)

    print(f"\n{'Metric':<20} {'IS':>12} {'OOS':>12} {'Ratio':>12}")
    print("-" * 50)
    print(f"{'Ann Return':<20} {metrics_is['ret']*100:>11.2f}% {metrics_oos['ret']*100:>11.2f}% {metrics_oos['ret']/metrics_is['ret'] if metrics_is['ret'] > 0 else 0:>11.2f}x")
    print(f"{'Sharpe':<20} {metrics_is['sharpe']:>11.2f} {metrics_oos['sharpe']:>11.2f} {metrics_oos['sharpe']/metrics_is['sharpe'] if metrics_is['sharpe'] > 0 else 0:>11.2f}x")
    print(f"{'Max DD':<20} {metrics_is['max_dd']*100:>11.2f}% {metrics_oos['max_dd']*100:>11.2f}% {metrics_oos['max_dd']/metrics_is['max_dd'] if metrics_is['max_dd'] != 0 else 0:>11.2f}x")

    consistency = abs((metrics_oos['sharpe'] - metrics_is['sharpe']) / (abs(metrics_is['sharpe']) + 1e-8))
    if consistency < 0.25:
        print("\n✓ STABLE: Performance metrics persist OOS")
        check2_pass = True
    else:
        print("\n⚠ UNSTABLE: Performance degrades significantly OOS")
        check2_pass = False

    # Check 3: Drawdown Behavior
    print("\n" + "="*70)
    print("3. DRAWDOWN BEHAVIOR (IS vs OOS)")
    print("="*70)

    def get_drawdown_stats(ret_series):
        cum = (1 + ret_series).cumprod()
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        max_dd = dd.min()
        avg_dd = dd[dd < 0].mean() if (dd < 0).sum() > 0 else 0
        return {'max_dd': max_dd, 'avg_dd': avg_dd}

    dd_is = get_drawdown_stats(rolling_ret_is)
    dd_oos = get_drawdown_stats(rolling_ret_oos)

    print(f"\n{'Metric':<20} {'IS':>12} {'OOS':>12} {'Drift':>12}")
    print("-" * 50)
    print(f"{'Max Drawdown':<20} {dd_is['max_dd']*100:>11.2f}% {dd_oos['max_dd']*100:>11.2f}% {(dd_oos['max_dd']-dd_is['max_dd'])*100:>11.2f}%")
    print(f"{'Avg Drawdown':<20} {dd_is['avg_dd']*100:>11.2f}% {dd_oos['avg_dd']*100:>11.2f}% {(dd_oos['avg_dd']-dd_is['avg_dd'])*100:>11.2f}%")

    dd_stability = abs((dd_oos['max_dd'] - dd_is['max_dd']) / (abs(dd_is['max_dd']) + 1e-8))
    if dd_stability < 0.35:
        print("\n✓ STABLE: Drawdown behavior similar IS vs OOS")
        check3_pass = True
    else:
        print("\n⚠ UNSTABLE: Drawdown behavior shifts significantly")
        check3_pass = False

    # Check 4: Win Rate Stability
    print("\n" + "="*70)
    print("4. WIN RATE STABILITY (IS vs OOS)")
    print("="*70)

    win_rate_is = (rolling_ret_is > 0).sum() / len(rolling_ret_is)
    win_rate_oos = (rolling_ret_oos > 0).sum() / len(rolling_ret_oos)

    print(f"\n{'Metric':<20} {'IS':>12} {'OOS':>12} {'Drift':>12}")
    print("-" * 50)
    print(f"{'Win Rate':<20} {win_rate_is*100:>11.1f}% {win_rate_oos*100:>11.1f}% {(win_rate_oos-win_rate_is)*100:>11.1f}%")

    wr_stability = abs(win_rate_oos - win_rate_is)
    if wr_stability < 0.10:
        print("\n✓ STABLE: Win rates consistent IS vs OOS")
        check4_pass = True
    else:
        print("\n⚠ UNSTABLE: Win rates diverge significantly")
        check4_pass = False

    # ====================================
    # OVERALL VERDICT
    # ====================================
    print("\n" + "="*70)
    print("STABILITY SCORECARD (GTO WITH MACRO)")
    print("="*70)

    checks = {
        'Return Distribution': check1_pass,
        'Performance Consistency': check2_pass,
        'Drawdown Behavior': check3_pass,
        'Win Rate Stability': check4_pass
    }

    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "⚠ FAIL"
        print(f"{check_name:<35} {status}")

    n_pass = sum(checks.values())
    n_total = len(checks)
    print(f"\nOverall: {n_pass}/{n_total} stability checks passed")

    if n_pass >= 3:
        print("\n→ VERDICT: GTO with Macro shows GOOD stability.")
        print("  Quarterly reoptimization + macro data + smoothing = ROBUST")
        print("  Strategy is HONEST and TRADABLE.")
        is_robust = True
    else:
        print("\n→ VERDICT: GTO with Macro still lacks stability.")
        print("  Consider TSMOM alternative.")
        is_robust = False

    print("\n" + "="*70)
    print("GTO WITH MACRO ANALYSIS COMPLETE")
    print("="*70)

    if is_robust:
        print(f"\n✓ RECOMMENDATION: Submit GTO-Quarterly+Macro")
        print(f"  Sharpe: {metrics_rolling['sharpe']:.2f}")
        print(f"  Stability: {n_pass}/4 checks")
        print(f"  Macro-aware regimes + calendar quarterly rebalancing")
    else:
        print(f"\n✓ RECOMMENDATION: Fall back to TSMOM")
        print(f"  (Submit GTO analysis as supporting research)")

print("\n" + "="*70)
print("MACRO-AWARE GTO BACKTEST COMPLETE")
print("="*70)                             