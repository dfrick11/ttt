
import os
import io
import math
import json
import sqlite3
from datetime import datetime, date
from typing import List, Tuple, Optional

import streamlit as st
from PIL import Image

DB_PATH = os.path.join("data", "journal.db")
SS_DIR = os.path.join("data", "screenshots")
os.makedirs("data", exist_ok=True)
os.makedirs(SS_DIR, exist_ok=True)

# -------------------------- DB Helpers --------------------------

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        instrument TEXT,
        direction TEXT CHECK(direction IN ('Long','Short')),
        session TEXT,
        strategy TEXT,
        entry REAL,
        stop REAL,
        target REAL,
        exit REAL,
        qty REAL,
        fees REAL DEFAULT 0,
        risk_amount REAL,               -- $ risk per trade
        planned_r REAL,                 -- target R planned
        realized_r REAL,                -- actual R result
        potential_profit REAL,          -- $ potential if plan hit
        realized_pnl REAL,              -- $ realized
        review TEXT,
        tags TEXT                       -- comma separated
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS screenshots(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_id INTEGER,
        path TEXT,
        FOREIGN KEY(trade_id) REFERENCES trades(id) ON DELETE CASCADE
    )
    """)
    conn.commit()
    conn.close()

def insert_trade(d):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO trades(ts,instrument,direction,session,strategy,entry,stop,target,exit,qty,fees,
                           risk_amount,planned_r,realized_r,potential_profit,realized_pnl,review,tags)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        d["ts"], d["instrument"], d["direction"], d["session"], d["strategy"],
        d["entry"], d["stop"], d["target"], d["exit"], d["qty"], d["fees"],
        d["risk_amount"], d["planned_r"], d["realized_r"], d["potential_profit"],
        d["realized_pnl"], d["review"], d["tags"]
    ))
    trade_id = cur.lastrowid
    for p in d.get("screens", []):
        cur.execute("INSERT INTO screenshots(trade_id,path) VALUES(?,?)", (trade_id, p))
    conn.commit()
    conn.close()
    return trade_id

def update_trade(trade_id, d):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        UPDATE trades SET ts=?,instrument=?,direction=?,session=?,strategy=?,entry=?,stop=?,target=?,exit=?,qty=?,fees=?,
                          risk_amount=?,planned_r=?,realized_r=?,potential_profit=?,realized_pnl=?,review=?,tags=?
        WHERE id=?
    """, (
        d["ts"], d["instrument"], d["direction"], d["session"], d["strategy"],
        d["entry"], d["stop"], d["target"], d["exit"], d["qty"], d["fees"],
        d["risk_amount"], d["planned_r"], d["realized_r"], d["potential_profit"],
        d["realized_pnl"], d["review"], d["tags"], trade_id
    ))
    # manage screenshots (simple: replace all for now)
    cur.execute("DELETE FROM screenshots WHERE trade_id=?", (trade_id,))
    for p in d.get("screens", []):
        cur.execute("INSERT INTO screenshots(trade_id,path) VALUES(?,?)", (trade_id, p))
    conn.commit()
    conn.close()

def fetch_trade(trade_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM trades WHERE id=?", (trade_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return None, []
    cur.execute("SELECT * FROM screenshots WHERE trade_id=?", (trade_id,))
    shots = cur.fetchall()
    conn.close()
    return row, shots

def list_trade_ids(filters=None, order="DESC"):
    q = "SELECT id FROM trades"
    params = []
    if filters and filters.get("instrument"):
        q += " WHERE instrument LIKE ?"
        params.append(f"%{filters['instrument']}%")
    q += f" ORDER BY datetime(ts) {order}"
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(q, params)
    ids = [r["id"] for r in cur.fetchall()]
    conn.close()
    return ids

def fetch_all_trades():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM trades ORDER BY datetime(ts) DESC")
    rows = cur.fetchall()
    conn.close()
    return rows

def save_uploaded_images(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[str]:
    saved = []
    for f in files or []:
        # unique name
        base = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        ext = os.path.splitext(f.name)[1].lower() or ".png"
        path = os.path.join(SS_DIR, f"{base}{ext}")
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        saved.append(path)
    return saved

# -------------------------- Metrics --------------------------

def compute_metrics(rows: List[sqlite3.Row]):
    n = len(rows)
    if n == 0:
        return {"count":0}
    wins = [r for r in rows if (r["realized_r"] or 0) > 0]
    losses = [r for r in rows if (r["realized_r"] or 0) < 0]
    win_rate = len(wins)/n if n else 0
    avg_r = sum([r["realized_r"] or 0 for r in rows]) / n
    total_r = sum([r["realized_r"] or 0 for r in rows])
    exp = (sum([r["realized_r"] or 0 for r in wins])/len(wins) if wins else 0) * win_rate \
          + (sum([r["realized_r"] or 0 for r in losses])/len(losses) if losses else 0) * (1 - win_rate)
    total_pnl = sum([r["realized_pnl"] or 0 for r in rows])
    return {
        "count": n,
        "win_rate": win_rate,
        "avg_r": avg_r,
        "total_r": total_r,
        "expectancy_r": exp,
        "total_pnl": total_pnl
    }

# -------------------------- UI --------------------------

st.set_page_config(page_title="Trade Journal", page_icon="📒", layout="wide")
init_db()

# Sidebar navigation
st.sidebar.title("📒 Trade Journal")
page = st.sidebar.radio("Navigate", ["➕ New Trade","📈 Dashboard","📚 Browse Trades"], label_visibility="collapsed")
st.sidebar.markdown("---")

# Filters
with st.sidebar:
    st.subheader("Filters")
    instr_filter = st.text_input("Instrument contains", value="")
    sort_order = st.selectbox("Sort by date", ["DESC (Newest first)","ASC (Oldest first)"])
    order_val = "DESC" if "DESC" in sort_order else "ASC"
    st.session_state.setdefault("filters", {})
    st.session_state["filters"] = {"instrument": instr_filter}

def r_or_none(x):
    try:
        return float(x)
    except:
        return None

def calc_risk_amount(entry, stop, qty, tick_value=None):
    if entry is None or stop is None or qty is None:
        return None
    # simple $ risk = |entry - stop| * qty * (tick_value or 1)
    # user can treat prices as $ if forex/CFD not tick-based
    return abs(entry - stop) * qty * (tick_value or 1.0)

def calc_r_multiple(entry, stop, exit):
    if None in (entry, stop, exit):
        return None
    risk = abs(entry - stop)
    if risk == 0:
        return None
    # Long if exit > entry; Short if exit < entry; sign captured by (exit-entry)/risk
    return (exit - entry) / risk

def calc_potential_profit(risk_amount, planned_r):
    if risk_amount is None or planned_r is None:
        return None
    return risk_amount * planned_r

def calc_realized_pnl(risk_amount, realized_r, fees):
    if risk_amount is None or realized_r is None:
        return None
    pnl = risk_amount * realized_r - (fees or 0)
    return pnl

# -------------------------- New/Edit Trade Form --------------------------

def trade_form(existing: Optional[sqlite3.Row] = None, existing_shots: Optional[List[sqlite3.Row]] = None):
    st.subheader("Trade Details")
    cols = st.columns(4)
    ts = cols[0].date_input("Trade Date", value=(date.fromisoformat(existing["ts"][:10]) if existing else date.today()))
    ttime = cols[1].text_input("Trade Time (HH:MM)", value=(existing["ts"][11:16] if existing else datetime.now().strftime("%H:%M")))
    instrument = cols[2].text_input("Instrument", value=(existing["instrument"] if existing else ""))
    direction = cols[3].selectbox("Direction", ["Long","Short"], index=(["Long","Short"].index(existing["direction"]) if existing and existing["direction"] in ["Long","Short"] else 0))

    cols2 = st.columns(4)
    session = cols2[0].selectbox("Session", ["Asia","London","NY","Overnight","Other"], index=(["Asia","London","NY","Overnight","Other"].index(existing["session"]) if existing and existing["session"] else 0))
    strategy = cols2[1].text_input("Strategy/Setup", value=(existing["strategy"] if existing else ""))
    qty = cols2[2].number_input("Qty / Contracts", value=float(existing["qty"]) if existing and existing["qty"] is not None else 1.0, step=1.0, format="%.2f")
    fees = cols2[3].number_input("Fees/Comm ($)", value=float(existing["fees"]) if existing and existing["fees"] is not None else 0.0, step=0.01, format="%.2f")

    cols3 = st.columns(4)
    entry = cols3[0].number_input("Entry Price", value=float(existing["entry"]) if existing and existing["entry"] is not None else 0.0, step=0.01, format="%.5f")
    stop = cols3[1].number_input("Stop Price", value=float(existing["stop"]) if existing and existing["stop"] is not None else 0.0, step=0.01, format="%.5f")
    target = cols3[2].number_input("Target Price (Plan)", value=float(existing["target"]) if existing and existing["target"] is not None else 0.0, step=0.01, format="%.5f")
    exitp = cols3[3].number_input("Exit Price (Actual)", value=float(existing["exit"]) if existing and existing["exit"] is not None else 0.0, step=0.01, format="%.5f")

    cols4 = st.columns(3)
    planned_r = cols4[0].number_input("Planned R (e.g., 2 = 2R)", value=float(existing["planned_r"]) if existing and existing["planned_r"] is not None else 2.0, step=0.25, format="%.2f")
    realized_r = cols4[1].number_input("Realized R (auto if exit/entry/stop)", value=float(existing["realized_r"]) if existing and existing["realized_r"] is not None else 0.0, step=0.25, format="%.2f")
    tick_value = cols4[2].number_input("Tick/Point Value (optional)", value=1.0, step=0.01, format="%.2f", help="If futures or instruments with tick value, set this to compute $ risk/PnL.")

    # Auto-calc
    risk_amount = calc_risk_amount(entry, stop, qty, tick_value)
    auto_realized_r = calc_r_multiple(entry, stop, exitp)
    if auto_realized_r is not None:
        realized_r = auto_realized_r
    potential_profit = calc_potential_profit(risk_amount, planned_r)
    realized_pnl = calc_realized_pnl(risk_amount, realized_r, fees)

    st.markdown(f"**$ Risk/Trade:** {risk_amount if risk_amount is not None else '—'}  |  **Potential $ Profit (plan):** {potential_profit if potential_profit is not None else '—'}  |  **Realized R:** {round(realized_r,2) if realized_r is not None else '—'}  |  **Realized $ PnL:** {round(realized_pnl,2) if realized_pnl is not None else '—'}")

    tags = st.text_input("Tags (comma separated)", value=(existing["tags"] if existing and existing["tags"] else ""))
    review = st.text_area("Trade Review / Lessons", value=(existing["review"] if existing and existing["review"] else ""), height=160)

    st.subheader("Screenshots")
    uploaded = st.file_uploader("Upload 1–6 images (PNG/JPG)", accept_multiple_files=True, type=["png","jpg","jpeg"])
    existing_paths = [r["path"] for r in (existing_shots or [])]
    if existing_paths:
        st.caption("Existing screenshots:")
        for p in existing_paths:
            try:
                st.image(p, use_column_width=True)
            except:
                st.text(f"(missing) {p}")

    # Prepare payload
    ts_str = f"{ts.isoformat()} {ttime}:00"
    payload = {
        "ts": ts_str,
        "instrument": instrument.strip(),
        "direction": direction,
        "session": session,
        "strategy": strategy.strip(),
        "entry": float(entry) if entry is not None else None,
        "stop": float(stop) if stop is not None else None,
        "target": float(target) if target is not None else None,
        "exit": float(exitp) if exitp is not None else None,
        "qty": float(qty) if qty is not None else None,
        "fees": float(fees) if fees is not None else 0.0,
        "risk_amount": risk_amount,
        "planned_r": float(planned_r) if planned_r is not None else None,
        "realized_r": float(realized_r) if realized_r is not None else None,
        "potential_profit": potential_profit,
        "realized_pnl": realized_pnl,
        "review": review.strip(),
        "tags": tags.strip(),
        "screens": []
    }

    if uploaded:
        saved_paths = save_uploaded_images(uploaded)
        payload["screens"].extend(saved_paths)
    else:
        # keep existing if editing
        payload["screens"].extend(existing_paths)

    return payload

# -------------------------- Page: New Trade --------------------------
if page == "➕ New Trade":
    st.title("➕ Log a New Trade")
    payload = trade_form()
    if st.button("Save Trade", type="primary"):
        tid = insert_trade(payload)
        st.success(f"Trade saved (ID #{tid}). Go to Browse to view.")
        st.balloons()

# -------------------------- Page: Browse Trades --------------------------
elif page == "📚 Browse Trades":
    st.title("📚 Browse Trades")
    ids = list_trade_ids(filters=st.session_state.get("filters"), order=order_val)

    if "idx" not in st.session_state:
        st.session_state["idx"] = 0
    # keep idx in range
    st.session_state["idx"] = min(st.session_state["idx"], max(0, len(ids)-1))

    colA, colB, colC, colD = st.columns([1,1,1,4])
    with colA:
        if st.button("⟵ Prev", disabled=(len(ids)==0 or st.session_state["idx"]<=0)):
            st.session_state["idx"] -= 1
    with colB:
        if st.button("Next ⟶", disabled=(len(ids)==0 or st.session_state["idx"]>=len(ids)-1)):
            st.session_state["idx"] += 1
    with colC:
        st.write(f"{(st.session_state['idx']+1) if ids else 0} / {len(ids)}")

    if not ids:
        st.info("No trades yet. Add one in 'New Trade'.")
    else:
        tid = ids[st.session_state["idx"]]
        row, shots = fetch_trade(tid)
        st.subheader(f"Trade #{tid} — {row['instrument']} ({row['direction']})  •  {row['ts']}")
        cols = st.columns(4)
        cols[0].metric("Realized R", f"{row['realized_r']:.2f}" if row['realized_r'] is not None else "—")
        cols[1].metric("$ Risk/Trade", f"{row['risk_amount']:.2f}" if row['risk_amount'] is not None else "—")
        cols[2].metric("Potential $ Profit", f"{row['potential_profit']:.2f}" if row['potential_profit'] is not None else "—")
        cols[3].metric("Realized $ PnL", f"{row['realized_pnl']:.2f}" if row['realized_pnl'] is not None else "—")

        st.markdown("**Session/Strategy:** " + f"{row['session'] or ''} / {row['strategy'] or ''}")
        st.markdown("**Entry / Stop / Target / Exit:** " + " / ".join([
            str(row['entry'] or "—"),
            str(row['stop'] or "—"),
            str(row['target'] or "—"),
            str(row['exit'] or "—")
        ]))
        st.markdown("**Qty / Fees:** " + f"{row['qty'] or '—'} / {row['fees'] or '—'}")
        st.markdown("**Tags:** " + f"{row['tags'] or ''}")

        st.markdown("**Review / Notes**")
        st.write(row["review"] or "—")

        if shots:
            st.markdown("**Screenshots**")
            for s in shots:
                try:
                    st.image(s["path"], use_column_width=True)
                except:
                    st.text(f"(missing) {s['path']}")

        with st.expander("✏️ Edit this trade"):
            updated = trade_form(existing=row, existing_shots=shots)
            if st.button("Save Changes"):
                update_trade(tid, updated)
                st.success("Updated. Use Prev/Next to refresh view.")

# -------------------------- Page: Dashboard --------------------------
elif page == "📈 Dashboard":
    st.title("📈 Performance Dashboard")
    rows = fetch_all_trades()
    m = compute_metrics(rows)
    if m.get("count",0) == 0:
        st.info("No trades logged yet.")
    else:
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Trades", m["count"])
        c2.metric("Win Rate", f"{m['win_rate']*100:.1f}%")
        c3.metric("Avg R", f"{m['avg_r']:.2f}")
        c4.metric("Expectancy (R)", f"{m['expectancy_r']:.2f}")
        c5.metric("Total $ PnL", f"{m['total_pnl']:.2f}")

        # Table
        st.markdown("---")
        st.subheader("Recent Trades")
        import pandas as pd
        df = pd.DataFrame([dict(r) for r in rows])
        st.dataframe(df, use_container_width=True)

        # Simple cumulative curves
        try:
            import matplotlib.pyplot as plt
            df2 = df[::-1].reset_index(drop=True)  # oldest to newest
            df2["cum_r"] = df2["realized_r"].fillna(0).cumsum()
            df2["cum_pnl"] = df2["realized_pnl"].fillna(0).cumsum()

            st.subheader("Cumulative R")
            fig1 = plt.figure()
            plt.plot(df2.index, df2["cum_r"])
            plt.xlabel("Trade # (old→new)")
            plt.ylabel("Cumulative R")
            st.pyplot(fig1)

            st.subheader("Cumulative $ PnL")
            fig2 = plt.figure()
            plt.plot(df2.index, df2["cum_pnl"])
            plt.xlabel("Trade # (old→new)")
            plt.ylabel("Cumulative $")
            st.pyplot(fig2)
        except Exception as e:
            st.warning(f"Could not render charts: {e}")

    st.markdown("---")
    st.download_button("Export CSV", data=df.to_csv(index=False) if m.get("count",0)>0 else "", file_name="trades_export.csv", disabled=(m.get("count",0)==0))

st.markdown("---")
st.caption("Built with ❤️ in Streamlit. Data stored locally in ./data")
