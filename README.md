
# Trade Journal (macOS-friendly)

A local, simple trading journal you run on your Mac. Add screenshots, write reviews, record R-multiples and $ PnL, and track performance over time. Each trade has its own page with Next/Prev navigation.

## Quick Start (macOS)

1. **Install Python 3.9+** (macOS usually has Python 3; if not, get it from python.org).
2. Open Terminal and run:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   streamlit run streamlit_app.py
   ```
3. Your browser will open at `http://localhost:8501`.

## Features

- New Trade form with instrument, direction, entry/stop/target/exit, qty, fees.
- Auto-calculated risk ($), realized R, potential profit, and realized PnL.
- Upload multiple screenshots per trade; files saved under `data/screenshots/`.
- Browse Trades with **Prev/Next** controls (like flipping trade pages).
- Edit trades inline.
- Dashboard with win rate, average R, expectancy (R), total $ PnL, cumulative charts.
- CSV export of all trades.
- All data stored locally in `data/journal.db` (SQLite).

## Notes

- Tick/Point Value lets you translate price distance to $ risk/PnL (e.g., for futures).
- Your data never leaves your machine; this is a local app.
- Back up the `data/` folder if you want to save or move your journal.
