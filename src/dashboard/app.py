"""
ShiftGuard Dashboard
Run: streamlit run src/dashboard/app.py
"""
import streamlit as st
import pandas as pd
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# --- helpers ---
def load_csv(path):
    full = os.path.join(ROOT, path)
    return pd.read_csv(full) if os.path.exists(full) else pd.DataFrame()

def load_decisions(pair):
    path = os.path.join(ROOT, 'results', 'decisions', f'{pair}_decisions.csv')
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=['datetime_utc','decision','notes'])

def save_decision(pair, dt, decision, notes):
    os.makedirs(os.path.join(ROOT, 'results', 'decisions'), exist_ok=True)
    path = os.path.join(ROOT, 'results', 'decisions', f'{pair}_decisions.csv')
    old = load_decisions(pair)
    new = pd.concat([old, pd.DataFrame([{'datetime_utc':dt,'decision':decision,'notes':notes}])], ignore_index=True)
    new.to_csv(path, index=False)

# --- page setup ---
st.set_page_config(page_title="ShiftGuard", layout="wide")

# --- sidebar ---
pair = st.sidebar.selectbox("Select pair", ['EURUSD','GBPJPY','XAUUSD'],
                            format_func=lambda x: {'EURUSD':'EUR/USD','GBPJPY':'GBP/JPY','XAUUSD':'XAU/USD'}[x])
min_sev = st.sidebar.slider("Minimum severity (1-5)", 1, 5, 1)
show_type = st.sidebar.multiselect("Show shift types", ['scheduled','unexpected'], default=['scheduled','unexpected'])

# --- load data ---
shifts = load_csv(f'results/detection/{pair}_shifts.csv')
attr = load_csv(f'results/attribution/{pair}_attribution.csv')
rolling = load_csv(f'results/predictions/xgboost_{pair}_rolling_mae.csv')
price = load_csv(f'data/raw/price/{pair}_4h.csv')
decisions = load_decisions(pair)

# --- title ---
st.title("ShiftGuard")
st.write(f"Monitoring **{pair}** for distribution shifts. This dashboard lets you review detected regime changes, see what caused them, and decide whether to retrain the model.")

# --- quick numbers ---
if not shifts.empty:
    col1, col2, col3 = st.columns(3)
    col1.write(f"**{len(shifts)}** total shifts detected")
    n_sched = len(shifts[shifts['type']=='scheduled']) if 'type' in shifts.columns else 0
    n_unexp = len(shifts[shifts['type']=='unexpected']) if 'type' in shifts.columns else 0
    col2.write(f"**{n_sched}** around economic events")
    col3.write(f"**{n_unexp}** unexpected (no calendar match)")

# --- price chart ---
st.subheader("Price chart")
st.write("4-hour closing prices. Look for sudden jumps or crashes — those are likely regime changes.")
if not price.empty:
    price['datetime_utc'] = pd.to_datetime(price['datetime_utc'])
    recent = price[price['datetime_utc'] >= '2021-01-01']
    st.line_chart(recent.set_index('datetime_utc')['close'])

# --- model error ---
st.subheader("How well is the model doing?")
st.write("This shows the model's average error over a rolling window. When it spikes, the model is struggling — probably because the market regime changed.")
if not rolling.empty:
    rolling['datetime_utc'] = pd.to_datetime(rolling['datetime_utc'])
    st.line_chart(rolling.set_index('datetime_utc')['rolling_mae_30'].dropna())

# --- what caused the shifts ---
st.subheader("What's causing the shifts?")
st.write("We used SHAP to figure out which group of features is responsible for each detected shift. Here's the breakdown:")
if not attr.empty and 'dominant_group' in attr.columns:
    col1, col2 = st.columns(2)
    with col1:
        st.write("**How many shifts each group caused:**")
        st.bar_chart(attr['dominant_group'].value_counts())
    with col2:
        gcols = [c for c in attr.columns if c.startswith('group_')]
        if gcols:
            st.write("**Average % contribution per group:**")
            means = attr[gcols].mean().sort_values(ascending=False)
            means.index = [c.replace('group_','') for c in means.index]
            st.bar_chart(means)

# --- shift table ---
st.subheader("Detected shifts")
st.write(f"Filtered to severity >= {min_sev}/5 and types: {', '.join(show_type)}")
if not shifts.empty:
    f = shifts.copy()
    if 'type' in f.columns:
        f = f[f['type'].isin(show_type)]
    if 'severity' in f.columns:
        f = f[f['severity'] >= min_sev]
        f['severity'] = f['severity'].apply(lambda x: f"{int(x)}/5")
    f = f.sort_values('datetime_utc', ascending=False)

    # pick readable columns
    cols = [c for c in ['datetime_utc','type','severity','event_names','n_features_triggered','trigger_features'] if c in f.columns]
    st.dataframe(f[cols].head(80), use_container_width=True, height=300)
    st.caption(f"Showing top {min(80, len(f))} of {len(f)} matching shifts")

# --- review a shift ---
st.subheader("Review a shift")
st.write("Pick a shift, see the SHAP breakdown, then confirm or reject it. Confirmed shifts will be used to retrain the model.")

if not shifts.empty:
    options = shifts.sort_values('datetime_utc', ascending=False)['datetime_utc'].head(40).tolist()
    pick = st.selectbox("Choose a shift", options)

    if pick:
        row = shifts[shifts['datetime_utc']==pick].iloc[0]
        sev = int(row.get('severity',0))
        labels = {1:'Minimal',2:'Low',3:'Moderate',4:'High',5:'Extreme'}

        st.write(f"**Type:** {str(row.get('type','')).replace('_',' ').title()}")
        st.write(f"**Severity:** {sev}/5 ({labels.get(sev,'Unknown')})")
        st.write(f"**Date:** {str(pick)[:10]}")

        # show shap breakdown if available
        if not attr.empty:
            match = attr[attr['datetime_utc']==pick]
            if not match.empty:
                arow = match.iloc[0]
                gcols = [c for c in arow.index if c.startswith('group_')]
                breakdown = {c.replace('group_',''):arow[c] for c in gcols if pd.notna(arow[c]) and arow[c]>0}
                if breakdown:
                    st.write("**What drove this shift (SHAP %):**")
                    st.bar_chart(pd.Series(breakdown).sort_values(ascending=True))
                    st.write(f"Main cause: **{arow.get('dominant_group','?')}** · Top feature: **{arow.get('top_feature_1','?')}**")

        # buttons
        notes = st.text_input("Add a note (optional)", key="note")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Confirm — retrain model"):
                save_decision(pair, pick, "confirm", notes)
                st.success("Confirmed. Model will retrain on post-shift data.")
                st.rerun()
        with col2:
            if st.button("Reject — false alarm"):
                save_decision(pair, pick, "reject", notes)
                st.warning("Rejected. Logged as false positive.")
                st.rerun()
        with col3:
            if st.button("Reclassify type"):
                new = "unexpected" if row.get('type')=="scheduled" else "scheduled"
                save_decision(pair, pick, f"reclassify_to_{new}", notes)
                st.info(f"Changed to {new}")
                st.rerun()

# --- decision log ---
st.subheader("Your decisions so far")
decisions = load_decisions(pair)
if not decisions.empty:
    st.dataframe(decisions, use_container_width=True)
    nc = len(decisions[decisions['decision']=='confirm'])
    nr = len(decisions[decisions['decision']=='reject'])
    st.write(f"**{nc}** confirmed, **{nr}** rejected, **{len(decisions)}** total")
else:
    st.write("No decisions yet. Start reviewing shifts above.")

st.write("---")
st.caption("ShiftGuard · Sohan Mahesh, Anusha Ravi Kumar, Dishaben Manubhai Patel · CS 6140 Machine Learning · Northeastern University")
