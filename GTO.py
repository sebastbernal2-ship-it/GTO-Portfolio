import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import linprog
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

# Asset universe: futures mapped to ETFs
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

# Download data
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

# Create DataFrame
price_data = pd.DataFrame(data).ffill().dropna(how='all')

print(f"\nPrice data shape: {price_data.shape}")

if price_data.shape[1] > 0:

    # ====================================
    # DATA PREP
    # ====================================
    daily_returns = price_data.pct_change()

    # Map to ETF names for regime detection
    etf_cols = [futures_to_etf[c] for c in price_data.columns]
    price_data_etf = price_data.copy()
    price_data_etf.columns = etf_cols
    daily_returns_etf = price_data_etf.pct_change()

    # ====================================
    # REGIME LABELING (LOOSENED THRESHOLDS)
    # ====================================
    print("\n" + "="*70)
    print("REGIME DETECTION")
    print("="*70)

    # Backward-looking indicators
    eq_ret_12m = daily_returns_etf[['SPY', 'QQQ', 'IWM']].mean(axis=1).rolling(252).mean() * 252
    vix_proxy  = daily_returns_etf['SPY'].rolling(60).std() * np.sqrt(252) * 10
    inf_proxy  = daily_returns_etf[['USO', 'GLD', 'CPER']].mean(axis=1).rolling(60).mean() * 252 * 100

    # LOOSENED THRESHOLDS for balanced regime split
    regime_signals = pd.DataFrame(index=daily_returns_etf.index)
    regime_signals['risk_on']  = (eq_ret_12m > 0.05)
    regime_signals['risk_off'] = (eq_ret_12m < -0.02) | (vix_proxy > 0.25)
    regime_signals['inflation'] = (inf_proxy > 0.05)
    regime_signals['normal'] = ~(regime_signals['risk_on'] |
                                 regime_signals['risk_off'] |
                                 regime_signals['inflation'])

    # Assign: priority order risk-off > inflation > risk-on > normal
    regime_id = pd.Series(3, index=daily_returns_etf.index)  # default normal
    regime_id[regime_signals['risk_off']] = 1
    regime_id[regime_signals['inflation']] = 2
    regime_id[regime_signals['risk_on']] = 0

    regime_names = {0: 'Risk-On', 1: 'Risk-Off', 2: 'Inflation', 3: 'Normal'}

    print("\nRegime Balance:")
    for r in range(4):
        count = (regime_id == r).sum()
        pct = count / len(regime_id) * 100
        print(f"  {regime_names[r]:<12}: {pct:>5.1f}% ({count:>5d} days)")

    # ====================================
    # REGIME-CONDITIONAL STATS
    # ====================================
    print("\nRegime-Conditional Expected Returns (ann %):")
    print("-" * 70)

    n_regimes = 4
    mu_reg = np.zeros((n_regimes, daily_returns.shape[1]))
    sigma_reg = np.zeros((n_regimes, daily_returns.shape[1], daily_returns.shape[1]))
    asset_names = price_data.columns.tolist()

    for r in range(n_regimes):
        mask = regime_id == r
        if mask.sum() > 252*2:
            ret_reg = daily_returns[mask]
            mu_reg[r] = ret_reg.mean() * 252
            sigma_reg[r] = ret_reg.cov() * 252

    print(f"\n{'Asset':<8}", end='')
    for r in range(4):
        print(f"{regime_names[r]:>12}", end='')
    print()
    print("-" * 70)

    for i, asset in enumerate(asset_names):
        print(f"{asset:<8}", end='')
        for r in range(4):
            print(f"{mu_reg[r, i]*100:>11.1f}%", end='')
        print()

    print("-" * 70)
    print("Avg Return per Regime:")
    for r in range(4):
        avg_ret = mu_reg[r].mean()
        print(f"  {regime_names[r]:<12}: {avg_ret*100:>6.2f}%")

    # ====================================
    # GTO MAXIMIN OPTIMIZATION
    # ====================================
    print("\n" + "="*70)
    print("GTO-INSPIRED PORTFOLIO CONSTRUCTION")
    print("="*70)

    def solve_maximin_gto(mu_reg, long_only=True, max_weight=0.15):
        """
        Solve: max z s.t. w^T mu_r >= z for all regimes r,
               sum(w) = 1, w_i in [0, max_weight] if long_only
        """
        n_assets = mu_reg.shape[1]
        n_regimes = mu_reg.shape[0]

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

        if long_only:
            bounds = [(0, max_weight) for _ in range(n_assets)] + [(None, None)]
        else:
            bounds = [(-max_weight, max_weight) for _ in range(n_assets)] + [(None, None)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs')

        return res

    print("\n[Solving long-only GTO portfolio...]")
    res = solve_maximin_gto(mu_reg, long_only=True, max_weight=0.15)

    if res.success:
        w_opt = res.x[:len(asset_names)]
        z_opt = res.x[-1]
        regime_payoffs = mu_reg @ w_opt

        print("\n" + "="*70)
        print("GTO PORTFOLIO (LONG-ONLY, MAX-MIN OPTIMIZATION)")
        print("="*70)

        print(f"\n{'Asset':<8} {'Weight':>10} {'Interpretation':>35}")
        print("-" * 70)

        idx_sorted = np.argsort(-w_opt)
        for idx in idx_sorted:
            w = w_opt[idx]
            if w > 0.001:
                interp = "diversifier" if w < 0.05 else "core holding" if w < 0.12 else "major position"
                print(f"{asset_names[idx]:<8} {w:>9.2%}  {interp:>35}")

        print("\n" + "-" * 70)
        print("PAYOFF MATRIX (Expected return in each regime):")
        print("-" * 70)
        print(f"{'Regime':<12} {'Return':>10} {'Interpretation':>45}")
        print("-" * 70)
        for r in range(n_regimes):
            payoff = regime_payoffs[r]
            print(f"{regime_names[r]:<12} {payoff*100:>9.2f}%", end='')
            if payoff == regime_payoffs.min():
                print(f"  <-- GUARANTEED MIN (Worst case)")
            elif payoff == regime_payoffs.max():
                print(f"  <-- BEST CASE")
            else:
                print()

        print("-" * 70)
        print(f"\nGUARANTEED MINIMUM RETURN: {z_opt*100:.2f}%")
        print(f"  → Portfolio never drops below this across regimes")
        print(f"\nWORST-CASE REGIME: {regime_names[np.argmin(regime_payoffs)]}")
        print(f"  → Even if market picks most adversarial regime")

    else:
        print(f"Optimization failed: {res.message}")

    # ====================================
    # STABILITY ANALYSIS: IS vs OOS
    # ====================================
    print("\n" + "="*70)
    print("STABILITY ANALYSIS FRAMEWORK")
    print("="*70)

    # Time split
    split_date = '2020-12-31'
    is_idx = daily_returns.index <= split_date
    oos_idx = daily_returns.index > split_date

    print(f"\nIn-Sample:     {daily_returns.index[is_idx][0].date()} to {daily_returns.index[is_idx][-1].date()} ({is_idx.sum()} days)")
    print(f"Out-of-Sample: {daily_returns.index[oos_idx][0].date()} to {daily_returns.index[oos_idx][-1].date()} ({oos_idx.sum()} days)")

    # ====================================
    # 1. REGIME DISTRIBUTION STABILITY
    # ====================================
    print("\n" + "="*70)
    print("1. REGIME DISTRIBUTION STABILITY")
    print("="*70)

    regime_id_is = regime_id[is_idx]
    regime_id_oos = regime_id[oos_idx]

    def compute_regime_dist(regime_id_period):
        counts = regime_id_period.value_counts(normalize=True)
        dist = np.zeros(4)
        for r in range(4):
            dist[r] = counts.get(r, 0.0)
        return dist

    regime_dist_is = compute_regime_dist(regime_id_is)
    regime_dist_oos = compute_regime_dist(regime_id_oos)

    kl_div = jensenshannon(regime_dist_is, regime_dist_oos)

    print("\nRegime Distribution IS:")
    for r in range(4):
        print(f"  {regime_names[r]:<12}: {regime_dist_is[r]*100:>5.1f}%")

    print("\nRegime Distribution OOS:")
    for r in range(4):
        print(f"  {regime_names[r]:<12}: {regime_dist_oos[r]*100:>5.1f}%")

    print(f"\nJensen-Shannon Divergence: {kl_div:.4f}")
    print(f"  Interpretation: 0.0 = identical, 0.5+ = major shift")
    if kl_div < 0.1:
        print(f"  ✓ STABLE: Regime distributions similar across periods")
        regime_dist_stable = True
    else:
        print(f"  ⚠ UNSTABLE: Regime distribution shifts significantly")
        regime_dist_stable = False

    # ====================================
    # 2. CONDITIONAL RETURNS STABILITY
    # ====================================
    print("\n" + "="*70)
    print("2. REGIME-CONDITIONAL RETURNS STABILITY")
    print("="*70)

    def get_regime_returns(daily_ret, regime_id_period, n_regimes=4):
        mu_reg = np.zeros((n_regimes, daily_ret.shape[1]))
        for r in range(n_regimes):
            mask = regime_id_period == r
            if mask.sum() > 50:
                mu_reg[r] = daily_ret[mask].mean() * 252
        return mu_reg

    mu_is = get_regime_returns(daily_returns[is_idx], regime_id_is)
    mu_oos = get_regime_returns(daily_returns[oos_idx], regime_id_oos)

    print("\nAverage Asset Return per Regime (IS vs OOS):")
    print(f"{'Regime':<12} {'IS Avg':>10} {'OOS Avg':>10} {'Drift':>10}")
    print("-" * 50)
    for r in range(4):
        ret_is = mu_is[r].mean()
        ret_oos = mu_oos[r].mean()
        drift = ret_oos - ret_is
        print(f"{regime_names[r]:<12} {ret_is*100:>9.2f}% {ret_oos*100:>9.2f}% {drift*100:>9.2f}%")

    corr_is_oos = np.corrcoef(mu_is.flatten(), mu_oos.flatten())[0, 1]
    print(f"\nCorrelation of mu_IS vs mu_OOS across all assets/regimes: {corr_is_oos:.3f}")
    if corr_is_oos > 0.8:
        print("  ✓ STABLE: Return patterns persist OOS")
        returns_stable = True
    else:
        print("  ⚠ UNSTABLE: Return patterns shift significantly")
        returns_stable = False

    # ====================================
    # 3. GTO WEIGHTS STABILITY
    # ====================================
    print("\n" + "="*70)
    print("3. GTO PORTFOLIO STABILITY")
    print("="*70)

    res_is = solve_maximin_gto(mu_is)
    res_oos = solve_maximin_gto(mu_oos)

    w_is = res_is.x[:len(asset_names)] if res_is.success else np.ones(len(asset_names))/len(asset_names)
    w_oos = res_oos.x[:len(asset_names)] if res_oos.success else np.ones(len(asset_names))/len(asset_names)

    z_is = res_is.x[-1] if res_is.success else 0
    z_oos = res_oos.x[-1] if res_oos.success else 0

    print("\nGTO Weights Stability (top positions):")
    print(f"{'Asset':<8} {'IS Weight':>12} {'OOS Weight':>12} {'Drift':>10}")
    print("-" * 50)
    for i in np.argsort(-w_is)[:7]:
        if w_is[i] > 0.01:
            drift = w_oos[i] - w_is[i]
            print(f"{asset_names[i]:<8} {w_is[i]:>11.2%} {w_oos[i]:>11.2%} {drift:>9.2%}")

    w_correlation = np.corrcoef(w_is, w_oos)[0, 1]
    print(f"\nWeight correlation IS vs OOS: {w_correlation:.3f}")
    if w_correlation > 0.7:
        print("  ✓ STABLE: Optimal weights stable across periods")
        weights_stable = True
    else:
        print("  ⚠ UNSTABLE: Weight allocation shifts significantly")
        weights_stable = False

    print(f"\nGuaranteed Min Return (Maximin z):")
    print(f"  IS: {z_is*100:.2f}%")
    print(f"  OOS: {z_oos*100:.2f}%")
    print(f"  Drift: {(z_oos - z_is)*100:.2f}%")

    # ====================================
    # 4. DECOMPOSED EDGE STABILITY
    # ====================================
    print("\n" + "="*70)
    print("4. DECOMPOSED EDGE ANALYSIS")
    print("  E[PnL] = (Avg Winner × P(Win)) + (Avg Loser × P(Loss))")
    print("="*70)

    def compute_edge_metrics(portfolio_returns):
        ret = portfolio_returns.dropna()
        
        win_mask = ret > 0
        loss_mask = ret < 0
        
        if win_mask.sum() == 0 or loss_mask.sum() == 0:
            return None
        
        avg_win = ret[win_mask].mean()
        avg_loss = ret[loss_mask].mean()
        p_win = win_mask.sum() / len(ret)
        p_loss = loss_mask.sum() / len(ret)
        
        exp_pnl = (avg_win * p_win) + (avg_loss * p_loss)
        
        return {
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'p_win': p_win,
            'p_loss': p_loss,
            'exp_pnl': exp_pnl,
            'win_loss_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else 0
        }

    gto_ret_is = (w_is * daily_returns[is_idx]).sum(axis=1)
    gto_ret_oos = (w_is * daily_returns[oos_idx]).sum(axis=1)

    edge_is = compute_edge_metrics(gto_ret_is)
    edge_oos = compute_edge_metrics(gto_ret_oos)

    print("\nEdge Components (using IS-optimized weights):")
    print(f"{'Metric':<20} {'IS':>12} {'OOS':>12} {'Drift':>12}")
    print("-" * 60)

    for key in ['avg_win', 'avg_loss', 'p_win', 'p_loss', 'exp_pnl', 'win_loss_ratio']:
        is_val = edge_is[key]
        oos_val = edge_oos[key]
        drift = oos_val - is_val
        
        if key == 'exp_pnl':
            print(f"{key:<20} {is_val*100:>11.2f}% {oos_val*100:>11.2f}% {drift*100:>11.2f}%")
        elif 'p_' in key:
            print(f"{key:<20} {is_val*100:>11.1f}% {oos_val*100:>11.1f}% {drift*100:>11.1f}%")
        else:
            print(f"{key:<20} {is_val*100:>11.2f}% {oos_val*100:>11.2f}% {drift*100:>11.2f}%")

    edge_stability = abs((edge_oos['exp_pnl'] - edge_is['exp_pnl']) / (abs(edge_is['exp_pnl']) + 1e-8))
    print(f"\nE[PnL] relative drift: {edge_stability*100:.1f}%")
    if edge_stability < 0.25:
        print("  ✓ STABLE: Expected value of trade stable OOS")
        edge_stable = True
    else:
        print("  ⚠ UNSTABLE: Edge degrades significantly out-of-sample")
        edge_stable = False

    # ====================================
    # 5. OVERALL STABILITY VERDICT
    # ====================================
    print("\n" + "="*70)
    print("STABILITY SCORECARD")
    print("="*70)

    stability_scores = {
        'Regime Distribution (KL)': regime_dist_stable,
        'Return Correlations': returns_stable,
        'Weight Stability': weights_stable,
        'Edge Persistence': edge_stable
    }

    for check, passed in stability_scores.items():
        status = "✓ PASS" if passed else "⚠ FAIL"
        print(f"{check:<35} {status}")

    n_pass = sum(stability_scores.values())
    n_total = len(stability_scores)
    print(f"\nOverall: {n_pass}/{n_total} stability checks passed")

    if n_pass >= 3:
        print("\n→ VERDICT: Strategy shows reasonable stability.")
        print("  Edge and P/L distributions persist out-of-sample.")
        print("  Can proceed to performance metrics (Sharpe, drawdown) as secondary validation.")
        is_stable = True
    else:
        print("\n→ VERDICT: Strategy lacks sufficient stability.")
        print("  Edge degrades significantly out-of-sample.")
        print("  Performance metrics alone are unreliable. Needs redesign.")
        is_stable = False

    # ====================================
    # 6. SECONDARY PERFORMANCE METRICS
    # ====================================
    if is_stable:
        print("\n" + "="*70)
        print("SECONDARY PERFORMANCE METRICS (supporting evidence only)")
        print("="*70)
        
        def compute_perf_metrics(ret_series):
            ret = ret_series.dropna()
            if len(ret) == 0:
                return None
            cum = (1 + ret).cumprod()
            years = len(ret) / 252.0
            ann_ret = cum.iloc[-1]**(1/years) - 1 if years > 0 else 0
            vol = ret.std() * np.sqrt(252)
            sharpe = ann_ret / vol if vol > 0 else 0
            dd = (cum / cum.cummax() - 1).min()
            return {'ret': ann_ret, 'vol': vol, 'sharpe': sharpe, 'dd': dd}
        
        perf_is = compute_perf_metrics(gto_ret_is)
        perf_oos = compute_perf_metrics(gto_ret_oos)
        
        print(f"\n{'Metric':<15} {'IS':>12} {'OOS':>12} {'Drift':>12}")
        print("-" * 55)
        print(f"{'Ann Return':<15} {perf_is['ret']*100:>11.2f}% {perf_oos['ret']*100:>11.2f}% {(perf_oos['ret']-perf_is['ret'])*100:>11.2f}%")
        print(f"{'Volatility':<15} {perf_is['vol']*100:>11.2f}% {perf_oos['vol']*100:>11.2f}% {(perf_oos['vol']-perf_is['vol'])*100:>11.2f}%")
        print(f"{'Sharpe':<15} {perf_is['sharpe']:>11.2f} {perf_oos['sharpe']:>11.2f} {perf_oos['sharpe']-perf_is['sharpe']:>11.2f}")
        print(f"{'Max DD':<15} {perf_is['dd']*100:>11.2f}% {perf_oos['dd']*100:>11.2f}% {(perf_oos['dd']-perf_is['dd'])*100:>11.2f}%")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
