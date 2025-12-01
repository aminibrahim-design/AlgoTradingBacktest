import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Dict, Callable, Tuple, Optional

st.set_page_config(page_title="Plotly Algo Backtester", layout="wide")


# =============================
# Data Loading (ROBUST)
# =============================
def load_data(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column"
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]

    # Standardize names
    clean_cols = []
    for c in df.columns:
        c2 = c.lower().replace("adj close", "close")
        c2 = c2.split("_")[0]  # strip ticker suffix like close_spy
        clean_cols.append(c2.title())

    df.columns = clean_cols

    # Ensure required columns
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(df.columns):
        possible_close = [c for c in df.columns if "Close" in c]
        if possible_close:
            df = df.rename(columns={possible_close[0]: "Close"})

    if "Close" not in df.columns:
        raise ValueError(f"'Close' column missing. Columns: {df.columns.tolist()}")

    return df.dropna()


def to_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().fillna(0.0)


def annualize_factor(interval: str) -> float:
    if interval == "1d":
        return 252
    if interval == "1h":
        return 252 * 6.5
    if interval == "30m":
        return 252 * 13
    if interval == "15m":
        return 252 * 26
    if interval == "5m":
        return 252 * 78
    return 252


# =============================
# Backtest Result Container
# =============================
@dataclass
class BacktestResult:
    df: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, float]


def compute_metrics(df: pd.DataFrame, ann_factor: float, rf_annual: float = 0.0) -> Dict[str, float]:
    strat_ret = df["StrategyRet"]
    bh_ret = df["BHRet"]

    equity = (1 + strat_ret).cumprod()
    bh_equity = (1 + bh_ret).cumprod()

    total_return = equity.iloc[-1] - 1
    bh_total_return = bh_equity.iloc[-1] - 1

    cagr = (equity.iloc[-1]) ** (ann_factor / len(df)) - 1 if len(df) > 1 else 0.0
    bh_cagr = (bh_equity.iloc[-1]) ** (ann_factor / len(df)) - 1 if len(df) > 1 else 0.0

    vol = strat_ret.std() * np.sqrt(ann_factor)
    bh_vol = bh_ret.std() * np.sqrt(ann_factor)

    rf_daily = (1 + rf_annual) ** (1 / ann_factor) - 1
    sharpe = ((strat_ret.mean() - rf_daily) / (strat_ret.std() + 1e-12)) * np.sqrt(ann_factor)

    roll_max = equity.cummax()
    dd = equity / roll_max - 1
    max_dd = dd.min()

    hit_rate = (strat_ret[df.get("Position", 0).shift(1) == 1] > 0).mean() \
        if "Position" in df.columns and (df["Position"].shift(1) == 1).any() else 0.0

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Hit Rate (on days in trade)": hit_rate,
        "BH Total Return": bh_total_return,
        "BH CAGR": bh_cagr,
        "BH Volatility": bh_vol,
    }


def trades_from_position(df: pd.DataFrame) -> pd.DataFrame:
    pos = df["Position"].fillna(0)
    change = pos.diff().fillna(0)

    entries = df.index[change == 1]
    exits = df.index[change == -1]

    if len(exits) and len(entries) and exits[0] < entries[0]:
        exits = exits[1:]

    if len(entries) > len(exits):
        entries = entries[:len(exits)]
    else:
        exits = exits[:len(entries)]

    trades = []
    for en, ex in zip(entries, exits):
        en_price = df.loc[en, "Close"]
        ex_price = df.loc[ex, "Close"]
        ret = ex_price / en_price - 1
        trades.append({
            "Entry Time": en,
            "Exit Time": ex,
            "Entry Price": en_price,
            "Exit Price": ex_price,
            "Return": ret,
            "Bars Held": df.loc[en:ex].shape[0]
        })
    return pd.DataFrame(trades)


# =============================
# Strategy Library (Single Asset)
# =============================
def sma_crossover(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    sma_fast = df["Close"].rolling(fast).mean()
    sma_slow = df["Close"].rolling(slow).mean()
    return (sma_fast > sma_slow).astype(int)


def ema_crossover(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    return (ema_fast > ema_slow).astype(int)


def rsi_reversion(df: pd.DataFrame, length: int, buy_below: float, sell_above: float) -> pd.Series:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))

    sig = pd.Series(0, index=df.index)
    sig[rsi < buy_below] = 1
    sig[rsi > sell_above] = 0
    return sig.ffill().fillna(0).astype(int)


def bollinger_reversion(df: pd.DataFrame, length: int, n_std: float) -> pd.Series:
    ma = df["Close"].rolling(length).mean()
    sd = df["Close"].rolling(length).std()
    upper = ma + n_std * sd
    lower = ma - n_std * sd

    sig = pd.Series(0, index=df.index)
    sig[df["Close"] < lower] = 1
    sig[df["Close"] > ma] = 0
    return sig.ffill().fillna(0).astype(int)


def zscore_reversion(df: pd.DataFrame, lookback: int, entry_z: float, exit_z: float) -> pd.Series:
    price = df["Close"]
    mean = price.rolling(lookback).mean()
    std = price.rolling(lookback).std()
    z = (price - mean) / (std + 1e-12)

    sig = pd.Series(0, index=df.index)
    sig[z < -entry_z] = 1
    sig[z > -exit_z] = 0
    return sig.ffill().fillna(0).astype(int)


def breakout(df: pd.DataFrame, lookback: int) -> pd.Series:
    high_roll = df["High"].rolling(lookback).max()
    low_roll = df["Low"].rolling(lookback).min()

    sig = pd.Series(0, index=df.index)
    sig[df["Close"] > high_roll.shift(1)] = 1
    sig[df["Close"] < low_roll.shift(1)] = 0
    return sig.ffill().fillna(0).astype(int)


# =============================
# Pairs Trading Strategy
# =============================
def pairs_zscore_signal(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    lookback: int,
    entry_z: float,
    exit_z: float,
    hedge_ratio: float = 1.0,
) -> pd.DataFrame:
    a = df_a["Close"]
    b = df_b["Close"]

    spread = a - hedge_ratio * b
    m = spread.rolling(lookback).mean()
    s = spread.rolling(lookback).std()
    z = (spread - m) / (s + 1e-12)

    pos_a = pd.Series(0, index=spread.index)
    pos_b = pd.Series(0, index=spread.index)

    long_spread = z < -entry_z
    short_spread = z > entry_z

    exit_long = z > -exit_z
    exit_short = z < exit_z

    pos_a[long_spread] = 1
    pos_b[long_spread] = -1

    pos_a[short_spread] = -1
    pos_b[short_spread] = 1

    pos_a[exit_long | exit_short] = 0
    pos_b[exit_long | exit_short] = 0

    pos_a = pos_a.replace(0, np.nan).ffill().fillna(0)
    pos_b = pos_b.replace(0, np.nan).ffill().fillna(0)

    return pd.DataFrame({"Spread": spread, "Z": z, "PosA": pos_a, "PosB": pos_b})


# =============================
# Backtest Engines
# =============================
def run_backtest(df: pd.DataFrame, signal: pd.Series, fee_bps: float, slippage_bps: float, ann_factor: float) -> BacktestResult:
    out = df.copy()
    out["Signal"] = signal.reindex(out.index).fillna(0).astype(int)

    out["Position"] = out["Signal"].shift(1).fillna(0)
    out["Ret"] = to_returns(out["Close"])
    out["BHRet"] = out["Ret"]

    trade = out["Position"].diff().abs().fillna(0)
    fee = (fee_bps + slippage_bps) / 10000.0
    out["Cost"] = trade * fee

    out["StrategyRet"] = out["Position"] * out["Ret"] - out["Cost"]
    out["Equity"] = (1 + out["StrategyRet"]).cumprod()
    out["BHEquity"] = (1 + out["BHRet"]).cumprod()

    trades = trades_from_position(out)
    metrics = compute_metrics(out, ann_factor)

    return BacktestResult(out, trades, metrics)


def run_pairs_backtest(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    pairs_df: pd.DataFrame,
    fee_bps: float,
    slippage_bps: float,
    ann_factor: float,
) -> BacktestResult:
    out = pd.DataFrame(index=pairs_df.index).copy()
    out["CloseA"] = df_a["Close"]
    out["CloseB"] = df_b["Close"]

    out["RetA"] = to_returns(out["CloseA"])
    out["RetB"] = to_returns(out["CloseB"])

    out["PosA"] = pairs_df["PosA"].shift(1).fillna(0)
    out["PosB"] = pairs_df["PosB"].shift(1).fillna(0)

    trade_a = out["PosA"].diff().abs().fillna(0)
    trade_b = out["PosB"].diff().abs().fillna(0)

    fee = (fee_bps + slippage_bps) / 10000.0
    out["Cost"] = (trade_a + trade_b) * fee

    out["StrategyRet"] = 0.5 * out["PosA"] * out["RetA"] + 0.5 * out["PosB"] * out["RetB"] - out["Cost"]
    out["BHRet"] = 0.5 * out["RetA"] + 0.5 * out["RetB"]

    out["Equity"] = (1 + out["StrategyRet"]).cumprod()
    out["BHEquity"] = (1 + out["BHRet"]).cumprod()

    # Fake Position column for metrics so hit-rate doesn't crash
    metrics = compute_metrics(out.assign(Position=(out["PosA"] != 0).astype(int)), ann_factor)
    trades = pd.DataFrame()

    return BacktestResult(out, trades, metrics)


# =============================
# Plotly Charts
# =============================
def plot_price_signals(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
    ))

    entries = df.index[df["Signal"].diff().fillna(0) == 1]
    exits = df.index[df["Signal"].diff().fillna(0) == -1]

    fig.add_trace(go.Scatter(
        x=entries, y=df.loc[entries, "Close"],
        mode="markers", marker=dict(symbol="triangle-up", size=10),
        name="Entry"
    ))
    fig.add_trace(go.Scatter(
        x=exits, y=df.loc[exits, "Close"],
        mode="markers", marker=dict(symbol="triangle-down", size=10),
        name="Exit"
    ))

    fig.update_layout(title=title, height=520, xaxis_rangeslider_visible=False)
    return fig


def plot_equity(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Equity"], name="Strategy", mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BHEquity"], name="Buy & Hold", mode="lines"))
    fig.update_layout(title=title, height=420)
    return fig


# =============================
# Strategy Registry
# =============================
STRATEGIES: Dict[str, Tuple[Optional[Callable], Dict]] = {
    "SMA Crossover": (sma_crossover, {"fast": 20, "slow": 50}),
    "EMA Crossover": (ema_crossover, {"fast": 12, "slow": 26}),
    "RSI Mean Reversion": (rsi_reversion, {"length": 14, "buy_below": 30.0, "sell_above": 70.0}),
    "Bollinger Band Reversion": (bollinger_reversion, {"length": 20, "n_std": 2.0}),
    "Donchian Breakout": (breakout, {"lookback": 20}),
    "Z-Score Reversion": (zscore_reversion, {"lookback": 60, "entry_z": 2.0, "exit_z": 0.5}),
    "Pairs Trading (Z-Score Spread)": (None, {"lookback": 60, "entry_z": 2.0, "exit_z": 0.5, "hedge_ratio": 1.0}),
}


# =============================
# UI
# =============================
st.title("📈 Plotly Algo Trading Backtester")

with st.sidebar:
    st.header("Data")
    ticker = st.text_input("Ticker", value="SPY")
    col1, col2 = st.columns(2)
    start = col1.date_input("Start", value=pd.to_datetime("2020-01-01"))
    end = col2.date_input("End", value=pd.to_datetime("today"))
    interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], index=0, help="Note: Intraday data (intervals <1d) is limited to last 60 days by Yahoo Finance.")

    st.divider()
    st.header("Strategy")
    strat_name = st.selectbox("Choose strategy", list(STRATEGIES.keys()))
    is_pairs = strat_name.startswith("Pairs Trading")

    pair_ticker = None
    if is_pairs:
        pair_ticker = st.text_input("Second ticker (pairs)", value="IVV")

    func, defaults = STRATEGIES[strat_name]

    params = {}
    for k, v in defaults.items():
        if isinstance(v, int):
            params[k] = st.number_input(k, min_value=1, value=v, step=1)
        else:
            params[k] = st.number_input(k, value=float(v))

    st.divider()
    st.header("Costs")
    fee_bps = st.number_input("Fee (bps per trade)", min_value=0.0, value=1.0, step=0.5)
    slippage_bps = st.number_input("Spread", min_value=0.0, value=1.0, step=0.5)

    st.divider()
    run_btn = st.button("🚀 Backtest")


if run_btn:
    df = load_data(ticker, str(start), str(end), interval)
    if df.empty:
        st.error("No data returned. Try another ticker/period.")
        st.stop()

    ann_factor = annualize_factor(interval)

    if not is_pairs:
        signal = func(df, **params)
        result = run_backtest(df, signal, fee_bps, slippage_bps, ann_factor)

        left, right = st.columns([1.2, 0.8], gap="large")
        with left:
            st.subheader("Price + Signals")
            st.plotly_chart(plot_price_signals(result.df, f"{ticker} — {strat_name}"), use_container_width=True)

            st.subheader("Equity Curve")
            st.plotly_chart(plot_equity(result.df, f"{ticker} — Equity"), use_container_width=True)

        with right:
            st.subheader("Performance Metrics")
            m = result.metrics

            def fmt(x):
                return f"{x*100:,.2f}%" if np.isfinite(x) else "n/a"

            st.metric("Total Return", fmt(m["Total Return"]))
            st.metric("CAGR", fmt(m["CAGR"]))
            st.metric("Volatility", fmt(m["Volatility"]))
            st.metric("Sharpe", f"{m['Sharpe']:.2f}")
            st.metric("Max Drawdown", fmt(m["Max Drawdown"]))
            st.metric("Hit Rate", fmt(m["Hit Rate (on days in trade)"]))

            st.divider()
            st.caption("Buy & Hold Benchmarks")
            st.metric("BH Total Return", fmt(m["BH Total Return"]))
            st.metric("BH CAGR", fmt(m["BH CAGR"]))
            st.metric("BH Volatility", fmt(m["BH Volatility"]))

            st.divider()
            st.subheader("Trades")
            if result.trades.empty:
                st.info("No completed trades in this period.")
            else:
                trades_df = result.trades.copy()
                trades_df["Return"] = trades_df["Return"].map(lambda r: f"{r*100:.2f}%")
                st.dataframe(trades_df, use_container_width=True)

        st.divider()
        st.subheader("Raw Backtest Data (last 200 rows)")
        st.dataframe(result.df.tail(200), use_container_width=True)

    else:
        df_b = load_data(pair_ticker, str(start), str(end), interval)
        if df_b.empty:
            st.error("Second ticker has no data.")
            st.stop()

        common = df.index.intersection(df_b.index)
        df_a_aligned = df.loc[common].copy()
        df_b_aligned = df_b.loc[common].copy()

        pairs_df = pairs_zscore_signal(
            df_a_aligned, df_b_aligned,
            lookback=int(params["lookback"]),
            entry_z=float(params["entry_z"]),
            exit_z=float(params["exit_z"]),
            hedge_ratio=float(params["hedge_ratio"]),
        )

        result = run_pairs_backtest(df_a_aligned, df_b_aligned, pairs_df, fee_bps, slippage_bps, ann_factor)

        left, right = st.columns([1.2, 0.8], gap="large")
        with left:
            st.subheader("Pairs Spread & Z-Score")
            spread_fig = go.Figure()
            spread_fig.add_trace(go.Scatter(x=pairs_df.index, y=pairs_df["Spread"], name="Spread"))
            spread_fig.add_trace(go.Scatter(x=pairs_df.index, y=pairs_df["Z"], name="Z-Score", yaxis="y2"))

            spread_fig.update_layout(
                height=420,
                yaxis=dict(title="Spread"),
                yaxis2=dict(title="Z-Score", overlaying="y", side="right"),
                title=f"{ticker} vs {pair_ticker} Spread & Z"
            )
            st.plotly_chart(spread_fig, use_container_width=True)

            st.subheader("Equity Curve")
            st.plotly_chart(plot_equity(result.df, f"{ticker}/{pair_ticker} — Equity"), use_container_width=True)

        with right:
            st.subheader("Performance Metrics")
            m = result.metrics

            def fmt(x):
                return f"{x*100:,.2f}%" if np.isfinite(x) else "n/a"

            st.metric("Total Return", fmt(m["Total Return"]))
            st.metric("CAGR", fmt(m["CAGR"]))
            st.metric("Volatility", fmt(m["Volatility"]))
            st.metric("Sharpe", f"{m['Sharpe']:.2f}")
            st.metric("Max Drawdown", fmt(m["Max Drawdown"]))

            st.divider()
            st.caption("Buy & Hold Benchmarks (50/50 long both)")
            st.metric("BH Total Return", fmt(m["BH Total Return"]))
            st.metric("BH CAGR", fmt(m["BH CAGR"]))
            st.metric("BH Volatility", fmt(m["BH Volatility"]))

        st.divider()
        st.subheader("Pairs Backtest Data (last 200 rows)")
        st.dataframe(result.df.tail(200), use_container_width=True)


# =============================
# Comparison Mode (single-asset only)
# =============================
with st.expander("🔁 Compare all single-asset strategies"):
    st.write("Runs every single-asset strategy using default params + your cost settings.")
    if st.button("Run Comparison"):
        df = load_data(ticker, str(start), str(end), interval)
        if df.empty:
            st.error("No data returned.")
            st.stop()

        ann_factor = annualize_factor(interval)

        rows = []
        equity_fig = go.Figure()
        equity_fig.add_trace(go.Scatter(
            x=df.index, y=(1 + to_returns(df["Close"])).cumprod(), name="Buy & Hold"
        ))

        for name, (f, defs) in STRATEGIES.items():
            if f is None:
                continue
            sig = f(df, **defs)
            res = run_backtest(df, sig, fee_bps, slippage_bps, ann_factor)

            rows.append({
                "Strategy": name,
                "Total Return": res.metrics["Total Return"],
                "CAGR": res.metrics["CAGR"],
                "Sharpe": res.metrics["Sharpe"],
                "Max DD": res.metrics["Max Drawdown"],
                "# Trades": len(res.trades),
            })

            equity_fig.add_trace(go.Scatter(x=df.index, y=res.df["Equity"], name=name))

        comp = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
        st.dataframe(
            comp.style.format({
                "Total Return": "{:.2%}",
                "CAGR": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Max DD": "{:.2%}",
            }),
            use_container_width=True
        )

        equity_fig.update_layout(title="Equity Curves Comparison", height=520)
        st.plotly_chart(equity_fig, use_container_width=True)
