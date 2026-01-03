import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import linprog
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

# ====================================
# ASSET UNIVERSE
# ====================================
futures_to_etf = {
    'ES': 'SPY', 'NQ': 'QQQ', 'RTY': 'IWM', 'CL': 'USO',
    'GC': 'GLD', 'HG': 'CPER', 'NG': 'UNG', 'EUR': 'FXE',
    'GBP': 'FXB', 'JPY': 'FXY', 'ZN': 'IEF', 'ZB': 'TLT'
}

start = '2015-12-31'
end   = '2025-12-28'

print("Libraries imported.")
print("Asset universe defined: 12 assets")
print("\nAssets:")
for future, etf in futures_to_etf.items():
    print(f"  {future:4s} to {etf}")

print(f"\nDownloading data from {start} to {end}")

# ====================================
# DOWNLOAD DATA
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
    regime_names = {0: 'Risk-On', 1: 'Risk-Mid', 2: 'Risk-Low', 3: 'Risk-Off'}

    # ====================================
    # ROLLING K-MEANS REGIME DETECTION
    # ====================================
    print("\n" + "="*70)
    print("ROLLING-WINDOW GTO: COMPLETE ANALYSIS")
    print("="*70)

    def get_regimes_at_date(daily_ret_etf, end_date, lookback_days=252*2):
        """Cluster regimes using data up to end_date with lookback"""
        start_date = end_date - timedelta(days=lookback_days)
        data_slice = daily_ret_etf[start_date:end_date]
        
        if len(data_slice) < 252:
            return None, None

        # Features
        yc_ret_long = data_slice['TLT'].rolling(20).mean() * 252
        yc_ret_short = data_slice['IEF'].rolling(20).mean() * 252
        yc_slope = yc_ret_long - yc_ret_short
        
        equity_momentum = data_slice[['SPY', 'QQQ', 'IWM']].mean(axis=1).rolling(20).mean()
        commodity_momentum = data_slice[['USO', 'GLD', 'CPER']].mean(axis=1).rolling(30).mean()
        realized_vol = data_slice['SPY'].rolling(30).std() * np.sqrt(252)

        features = np.column_stack([
            yc_slope.fillna(0),
            equity_momentum.fillna(0),
            commodity_momentum.fillna(0),
            realized_vol.fillna(0)
        ])

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=5)
        regime_id_raw = kmeans.fit_predict(features_scaled)
        
        # Relabel
        regime_eq_mom = [equity_momentum[regime_id_raw == r].mean() for r in range(4)]
        regime_order = np.argsort(regime_eq_mom)[::-1]
        regime_relabel = {old: new for new, old in enumerate(regime_order)}
        regime_id = np.array([regime_relabel[r] for r in regime_id_raw])
        
        return regime_id, data_slice.index

    def get_regime_returns(daily_ret, regime_id):
        """Compute mu_r for regime_id"""
        mu_reg = np.zeros((n_regimes, daily_ret.shape[1]))
        for r in range(n_regimes):
            mask = regime_id == r
            if mask.sum() > 50:
                mu_reg[r] = daily_ret[mask].mean() * 252
        return mu_reg

    def solve_maximin_gto(mu_reg, max_weight=0.15):
        """Maximin optimization"""
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
    # ROLLING WINDOW OPTIMIZATION
    # ====================================
    print("\n[Running rolling-window optimization...]")
    print("[Reoptimizing every 60 trading days]")
    print("[Using 2-year lookback for regime detection]\n")

    # Parameters
    reopt_freq = 30
    lookback = 252 * 2

    # Storage
    rolling_weights = {}
    rolling_regimes = {}

    # Iterate through dates
    optimization_dates = []
    for idx, date in enumerate(daily_returns.index[lookback:]):
        if (idx + 1) % reopt_freq == 0:
            optimization_dates.append(date)
            if len(optimization_dates) > 50:
                break

    print(f"Optimizing at {len(optimization_dates)} dates:")
    for opt_date in optimization_dates[:10]:
        print(f"  {opt_date.date()}")
    if len(optimization_dates) > 10:
        print(f"  ... ({len(optimization_dates) - 10} more)")

    for opt_date in optimization_dates:
        regime_id, regime_dates = get_regimes_at_date(daily_returns_etf, opt_date, lookback_days=lookback)
        if regime_id is None:
            continue

        data_for_mu = daily_returns.loc[regime_dates]
        mu_reg = get_regime_returns(data_for_mu, regime_id)

        res = solve_maximin_gto(mu_reg)
        if res.success:
            w = res.x[:len(asset_names)]
        else:
            w = np.ones(len(asset_names)) / len(asset_names)

        rolling_weights[opt_date] = w
        rolling_regimes[opt_date] = regime_id

    # ====================================
    # BACKTEST WITH ROLLING WEIGHTS
    # ====================================
    print("\n" + "="*70)
    print("BACKTEST: ROLLING-WINDOW GTO VS STATIC GTO")
    print("="*70)

    split_date = '2020-12-31'
    is_idx = daily_returns.index <= split_date
    oos_idx = daily_returns.index > split_date

    regime_id_is, regime_dates_is = get_regimes_at_date(daily_returns_etf, pd.Timestamp(split_date), lookback)
    data_is = daily_returns.loc[regime_dates_is]
    mu_is = get_regime_returns(data_is, regime_id_is)
    res_static = solve_maximin_gto(mu_is)
    w_static = res_static.x[:len(asset_names)]

    # Static portfolio
    oos_data = daily_returns[oos_idx]
    static_portfolio_ret = (w_static * oos_data).sum(axis=1)

    # Rolling portfolio
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

    print(f"\n{'Strategy':<25} {'Ann Ret':>12} {'Vol':>10} {'Sharpe':>10} {'Max DD':>12}")
    print("-" * 70)
    print(f"{'Static GTO (opt once)':<25} {metrics_static['ret']:>11.2%} {metrics_static['vol']:>9.2%} {metrics_static['sharpe']:>9.2f} {metrics_static['max_dd']:>11.2%}")
    print(f"{'Rolling GTO (opt 60d)':<25} {metrics_rolling['ret']:>11.2%} {metrics_rolling['vol']:>9.2%} {metrics_rolling['sharpe']:>9.2f} {metrics_rolling['max_dd']:>11.2%}")
    
    print(f"\nImprovement (Rolling vs Static):")
    print(f"  Sharpe: {metrics_rolling['sharpe'] - metrics_static['sharpe']:+.2f}")
    print(f"  Return: {(metrics_rolling['ret'] - metrics_static['ret'])*100:+.1f} bps")

    # ====================================
    # STABILITY ANALYSIS FRAMEWORK
    # ====================================
    print("\n" + "="*70)
    print("STABILITY ANALYSIS: IS VS OOS WITH ROLLING OPTIMIZATION")
    print("="*70)

    # For rolling GTO, test stability by looking at performance in fixed windows
    rolling_is_idx = rolling_portfolio_ret.index <= split_date
    rolling_oos_idx = rolling_portfolio_ret.index > split_date

    rolling_ret_is = rolling_portfolio_ret[rolling_is_idx]
    rolling_ret_oos = rolling_portfolio_ret[rolling_oos_idx]

    print(f"\nIn-Sample:     {rolling_ret_is.index[0].date()} to {rolling_ret_is.index[-1].date()} ({len(rolling_ret_is)} days)")
    print(f"Out-of-Sample: {rolling_ret_oos.index[0].date()} to {rolling_ret_oos.index[-1].date()} ({len(rolling_ret_oos)} days)")

    # ====================================
    # CHECK 1: RETURN DISTRIBUTION STABILITY
    # ====================================
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

    # ====================================
    # CHECK 2: PERFORMANCE CONSISTENCY
    # ====================================
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

    # ====================================
    # CHECK 3: DRAWDOWN BEHAVIOR
    # ====================================
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

    # ====================================
    # CHECK 4: WIN RATE STABILITY
    # ====================================
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
    print("STABILITY SCORECARD (ROLLING GTO)")
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
        print("\n→ VERDICT: Rolling GTO shows GOOD stability.")
        print("  Continuous reoptimization maintains edge OOS.")
        print("  Strategy is STABLE and TRADABLE.")
        is_stable = True
    else:
        print("\n→ VERDICT: Rolling GTO lacks sufficient stability.")
        print("  Need to add constraints or smoothing.")
        is_stable = False

    # ====================================
    # TRANSACTION COST ANALYSIS
    # ====================================
    if is_stable:
        print("\n" + "="*70)
        print("TRANSACTION COST ANALYSIS")
        print("="*70)

        opt_dates_sorted = sorted(rolling_weights.keys())
        w_list = [rolling_weights[d] for d in opt_dates_sorted]
        w_diffs = []
        for i in range(1, len(w_list)):
            diff = np.sum(np.abs(w_list[i] - w_list[i-1]))
            w_diffs.append(diff)

        avg_weight_drift = np.mean(w_diffs)

        scenarios = [0, 5, 10, 20]
        
        print(f"\n{'Cost Scenario':<30} {'Sharpe':>10} {'Ann Ret':>12} {'vs Static':>12}")
        print("-" * 70)
        print(f"{'No transaction costs':<30} {metrics_rolling['sharpe']:>10.2f} {metrics_rolling['ret']:>11.2%} {metrics_rolling['sharpe']/metrics_static['sharpe']:.2f}x")

        best_sharpe_with_cost = 0
        for cost_bps in scenarios:
            rolling_portfolio_ret_costs = rolling_portfolio_ret.copy()
            
            for i in range(1, len(opt_dates_sorted)):
                prev_date = opt_dates_sorted[i-1]
                curr_date = opt_dates_sorted[i]
                
                w_prev = rolling_weights[prev_date]
                w_curr = rolling_weights[curr_date]
                drift = np.sum(np.abs(w_curr - w_prev))
                cost = drift * (cost_bps / 10000)
                
                mask = (rolling_portfolio_ret_costs.index > prev_date) & (rolling_portfolio_ret_costs.index <= curr_date)
                if mask.sum() > 0:
                    first_day_idx = rolling_portfolio_ret_costs.index[mask][0]
                    idx_in_series = rolling_portfolio_ret_costs.index.get_loc(first_day_idx)
                    rolling_portfolio_ret_costs.iloc[idx_in_series] -= cost

            metrics_with_cost = calc_metrics(rolling_portfolio_ret_costs)
            ratio = metrics_with_cost['sharpe'] / metrics_static['sharpe'] if metrics_static['sharpe'] > 0 else 0
            print(f"{cost_bps} bps per 1% drift{'':<14} {metrics_with_cost['sharpe']:>10.2f} {metrics_with_cost['ret']:>11.2%} {ratio:.2f}x")
            
            if cost_bps == 5:
                best_sharpe_with_cost = metrics_with_cost['sharpe']

        print(f"\n" + "="*70)
        print("REALISTIC SCENARIO: 5 bps per 1% weight change")
        print("="*70)
        print(f"  Assumption: Bid-ask spread + slippage on rebalance")
        print(f"  Average turnover per reopt: {avg_weight_drift:.1%}")
        print(f"  Estimated cost per reopt: {avg_weight_drift * 5:.0f} bps")
        print(f"  Rebalances per year: ~{252/reopt_freq:.0f}")
        print(f"  Total annual cost: ~{avg_weight_drift * 5 * (252/reopt_freq):.0f} bps")
        
        print(f"\nFinal Comparison (after realistic 5 bps costs):")
        print(f"{'Strategy':<30} {'Sharpe':>12} {'Ann Ret':>12} {'vs Static':>12}")
        print("-" * 70)
        print(f"{'Static GTO':<30} {metrics_static['sharpe']:>12.2f} {metrics_static['ret']:>11.2%} {'baseline':>11}")
        print(f"{'Rolling GTO (5 bps costs)':<30} {best_sharpe_with_cost:>12.2f} {metrics_rolling['ret']:>11.2%} {best_sharpe_with_cost/metrics_static['sharpe']:.2f}x")

print("\n" + "="*70)
print("ROLLING-WINDOW GTO ANALYSIS COMPLETE")
print("="*70)
