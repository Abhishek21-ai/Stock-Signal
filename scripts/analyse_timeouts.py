import re
import pandas as pd

# Paste your raw terminal trade log inside these triple quotes
terminal_output = """
── Sample Trades (last 10) ────────────────────────── Symbol Entry Exit Return Exit reason ------------------------------------------------------- COALINDIA 2026-04-28 2026-05-12 +1.30% TIMEOUT COALINDIA 2026-04-30 2026-05-07 -5.12% STOP SBIN 2026-05-13 2026-05-27 -0.69% TIMEOUT SBIN 2026-05-14 2026-05-26 -2.05% STOP POWERGRID 2026-05-20 2026-05-21 -2.21% STOP POWERGRID 2026-06-03 2026-06-08 -2.21% STOP SBIN 2026-06-09 2026-06-23 +3.93% TIMEOUT SBIN 2026-06-11 2026-06-25 +4.69% TIMEOUT EICHERMOT 2026-06-23 2026-06-29 -2.87% TIMEOUT SUNPHARMA 2026-06-25 2026-06-29 +0.06% TIMEOUT     
"""

def parse_and_diagnose(text_data: str):
    # Regex to capture: Symbol, Entry Date, Exit Date, Return %, Exit Reason
    pattern = r"([A-Z&\-]+)\s+(\d{4}-\d{2}-\d{2})\s+(\d{4}-\d{2}-\d{2})\s+([+\-]?\d+\.\d+)%\s+([A-Z_]+)"
    matches = re.findall(pattern, text_data)
    
    if not matches:
        print("❌ Error: No trade logs could be parsed from the text provided. Check formatting.")
        return
        
    # Convert regex matches into a structured DataFrame
    trades = []
    for match in matches:
        symbol, entry, exit, ret, reason = match
        entry_dt = pd.to_datetime(entry)
        exit_dt = pd.to_datetime(exit)
        holding_days = max(1, (exit_dt - entry_dt).days) # Avoid zero-division on intraday
        
        trades.append({
            "symbol": symbol,
            "return": float(ret),
            "exit_reason": reason,
            "holding_days": holding_days
        })
        
    df = pd.DataFrame(trades)
    
    print("\n=========================================================")
    print("          TIMEOUT CAPITAL EFFICIENCY DIAGNOSTICS         ")
    print("=========================================================")
    
    timeout_trades = df[df["exit_reason"] == "TIMEOUT"]
    total_trades = len(df)
    num_timeouts = len(timeout_trades)
    
    if num_timeouts == 0:
        print(f"Parsed {total_trades} trades successfully, but found 0 TIMEOUT entries.")
        return

    timeout_pct = (num_timeouts / total_trades) * 100
    print(f"Total Trades Parsed: {total_trades} | TIMEOUT Trades: {num_timeouts} ({timeout_pct:.1f}%)")
    print("---------------------------------------------------------")
    
    stats = timeout_trades["return"].describe()
    win_rate = (timeout_trades["return"] > 0).mean() * 100
    total_portfolio_pnl = df["return"].sum()
    pnl_share = (timeout_trades["return"].sum() / total_portfolio_pnl * 100) if total_portfolio_pnl != 0 else 0
    
    print(f"Mean Return:        {stats['mean']:.2f}%")
    print(f"Median Return:      {stats['50%']:.2f}%")
    print(f"Win Rate:           {win_rate:.1f}%")
    print(f"PnL Contribution:   {pnl_share:.1f}% of total returns")
    print("---------------------------------------------------------")
    
    print("CAPITAL EFFICIENCY MATRIX (Avg Return per Day Held):")
    for reason in df["exit_reason"].unique():
        sub = df[df["exit_reason"] == reason]
        avg_days = sub["holding_days"].mean()
        avg_ret = sub["return"].mean()
        efficiency = avg_ret / avg_days
        print(f"  * {reason:<10} -> Avg Days: {avg_days:>4.1f} | Avg Return: {avg_ret:>5.2f}% | Efficiency: {efficiency:>6.3f}")
    print("=========================================================\n")

# Execute
parse_and_diagnose(terminal_output)