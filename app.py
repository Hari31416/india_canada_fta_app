import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="India-Canada FTA Export Impact", page_icon="📈", layout="wide"
)
DATA_DIR = Path(__file__).parent / "data"


@st.cache_data(show_spinner=False)
def load_data(tariff_file=None, elasticity_file=None):
    tp = (
        tariff_file
        if tariff_file is not None
        else DATA_DIR / "Canada_Tariff_import_HS6_2023.csv"
    )
    ep = (
        elasticity_file
        if elasticity_file is not None
        else DATA_DIR / "import_demand_elasticity_canada_by_productHS6.xlsx"
    )
    tr = pd.read_csv(tp, encoding="cp1252", dtype={"HS Code": str})
    el = pd.read_excel(ep, engine="openpyxl", dtype={"HS Code": str})
    tr["HS6"] = tr["HS Code"].str.replace(r"\.0$", "", regex=True).str.zfill(6)
    el["HS6"] = el["HS Code"].str.replace(r"\.0$", "", regex=True).str.zfill(6)
    tr["Imports_USD"] = (
        pd.to_numeric(tr["Imports Value in 1000 USD"], errors="coerce").fillna(0) * 1000
    )
    tr["Tariff_pct"] = pd.to_numeric(tr["Simple Average"], errors="coerce")
    el["Import_Elasticity"] = pd.to_numeric(
        el["Import demand elasticity"], errors="coerce"
    )
    return tr, el[["HS6", "Import_Elasticity"]].drop_duplicates("HS6")


def model(tr, el, armington, export_supply, new_tariff, missing_policy):
    # One observation per supplier/product: MFN is used as baseline to avoid AHS/MFN duplication.
    mfn = tr[tr["DutyType"].astype(str).str.upper().eq("MFN")].copy()
    if mfn.empty:
        mfn = tr.drop_duplicates(["HS6", "Partner"]).copy()
    market = (
        mfn.groupby("HS6", as_index=False)["Imports_USD"]
        .sum()
        .rename(columns={"Imports_USD": "Canada_Total_Imports_USD"})
    )
    india = mfn[mfn["Partner"].eq(356)].copy()
    india = india.groupby("HS6", as_index=False).agg(
        Product_Name=("Product Name", "first"),
        Baseline_Exports_USD=("Imports_USD", "sum"),
        Initial_Tariff_pct=("Tariff_pct", "first"),
    )
    d = india.merge(market, on="HS6", how="left").merge(el, on="HS6", how="left")
    d["Elasticity_Status"] = np.where(
        d["Import_Elasticity"].isna(), "Missing", "Available"
    )
    if missing_policy == "Use median of available elasticities":
        d["Elasticity_Used"] = d["Import_Elasticity"].fillna(
            d["Import_Elasticity"].median()
        )
    else:
        d["Elasticity_Used"] = d["Import_Elasticity"]
    d["Other_Suppliers_USD"] = (
        d["Canada_Total_Imports_USD"] - d["Baseline_Exports_USD"]
    ).clip(lower=0)
    d["Initial_Tariff_pct"] = d["Initial_Tariff_pct"].fillna(0)
    d["Scenario_Tariff_pct"] = new_tariff
    t0 = d["Initial_Tariff_pct"] / 100
    t1 = d["Scenario_Tariff_pct"] / 100
    cut = ((t0 - t1) / (1 + t0)).clip(lower=0)
    eta = d["Elasticity_Used"].abs()
    # Trade creation: baseline Indian imports x absolute demand elasticity x proportional landed-price reduction.
    d["Trade_Creation_USD"] = d["Baseline_Exports_USD"] * eta * cut
    # Trade diversion: CES/Armington reallocation of the post-creation market, holding non-India landed prices fixed.
    total = d["Canada_Total_Imports_USD"].replace(0, np.nan)
    s = (d["Baseline_Exports_USD"] / total).fillna(0).clip(0, 1)
    rel = ((1 + t1) / (1 + t0)).clip(lower=1e-9)
    shock = np.power(rel, 1 - armington)
    new_s = (s * shock) / (s * shock + (1 - s))
    post_market = d["Canada_Total_Imports_USD"] + d["Trade_Creation_USD"]
    d["Trade_Diversion_USD"] = ((new_s - s) * post_market).clip(lower=0)
    # 99 approximates perfectly elastic export supply; price effect is intentionally zero.
    d["Price_Effect_USD"] = (
        0.0
        if export_supply >= 99
        else (d["Trade_Creation_USD"] + d["Trade_Diversion_USD"]) / (1 + export_supply)
    )
    d["Total_Export_Gain_USD"] = (
        d["Trade_Creation_USD"] + d["Trade_Diversion_USD"] + d["Price_Effect_USD"]
    )
    d["Projected_Exports_USD"] = d["Baseline_Exports_USD"] + d["Total_Export_Gain_USD"]
    d["Uplift_pct"] = np.where(
        d["Baseline_Exports_USD"] > 0,
        d["Total_Export_Gain_USD"] / d["Baseline_Exports_USD"] * 100,
        np.nan,
    )
    return d


st.title("India-Canada FTA Export Impact Simulator")
st.caption(
    "Deterministic HS6-level partial-equilibrium prototype inspired by the WITS SMART framework"
)
with st.sidebar:
    st.header("Scenario controls")
    arm = st.number_input("Armington substitution elasticity", 0.1, 20.0, 1.5, 0.1)
    supply = st.number_input(
        "India export supply elasticity",
        0.1,
        999.0,
        99.0,
        0.5,
        help="99 represents price-taker treatment and produces zero price effect.",
    )
    new_t = st.number_input("Preferential tariff after FTA (%)", 0.0, 100.0, 0.0, 0.25)
    miss = st.selectbox(
        "Missing import elasticity",
        ["Exclude from quantified total", "Use median of available elasticities"],
    )
    st.divider()
    st.caption("Optional: replace bundled source files")
    tf = st.file_uploader("Tariff/import CSV", type="csv")
    ef = st.file_uploader("Import elasticity Excel", type=["xlsx"])

tr, el = load_data(tf, ef)
d = model(tr, el, arm, supply, new_t, miss)
chapters = sorted(d["HS6"].str[:2].unique())
sel_ch = st.multiselect("HS chapters", chapters, default=[])
q = st.text_input("Search HS6 or product")
f = d.copy()
if sel_ch:
    f = f[f["HS6"].str[:2].isin(sel_ch)]
if q:
    f = f[
        f["HS6"].str.contains(q, case=False, na=False)
        | f["Product_Name"].str.contains(q, case=False, na=False)
    ]
quant = f.dropna(subset=["Elasticity_Used"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Baseline exports", f"US$ {quant['Baseline_Exports_USD'].sum()/1e6:,.1f}m")
c2.metric("Trade creation", f"US$ {quant['Trade_Creation_USD'].sum()/1e6:,.1f}m")
c3.metric("Trade diversion", f"US$ {quant['Trade_Diversion_USD'].sum()/1e6:,.1f}m")
c4.metric("Total export gain", f"US$ {quant['Total_Export_Gain_USD'].sum()/1e6:,.1f}m")

st.subheader("Top HS6 opportunities")
show = quant.sort_values("Total_Export_Gain_USD", ascending=False).copy()
show["HS6"] = show["HS6"].astype(str)
cols = [
    "HS6",
    "Product_Name",
    "Initial_Tariff_pct",
    "Import_Elasticity",
    "Baseline_Exports_USD",
    "Trade_Creation_USD",
    "Trade_Diversion_USD",
    "Total_Export_Gain_USD",
    "Uplift_pct",
]
st.dataframe(
    show[cols],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Initial_Tariff_pct": st.column_config.NumberColumn(
            "MFN tariff", format="%.2f%%"
        ),
        "Import_Elasticity": st.column_config.NumberColumn(
            "Import elasticity", format="%.3f"
        ),
        "Baseline_Exports_USD": st.column_config.NumberColumn(
            "Baseline exports", format="$%,.0f"
        ),
        "Trade_Creation_USD": st.column_config.NumberColumn(
            "Trade creation", format="$%,.0f"
        ),
        "Trade_Diversion_USD": st.column_config.NumberColumn(
            "Trade diversion", format="$%,.0f"
        ),
        "Total_Export_Gain_USD": st.column_config.NumberColumn(
            "Total gain", format="$%,.0f"
        ),
        "Uplift_pct": st.column_config.NumberColumn("Uplift", format="%.1f%%"),
    },
)
st.bar_chart(
    show.head(20).set_index("HS6")[["Trade_Creation_USD", "Trade_Diversion_USD"]]
)

out = f.copy()
out["Armington_Elasticity"] = arm
out["Export_Supply_Elasticity"] = supply
st.download_button(
    "Download detailed results (CSV)",
    out.to_csv(index=False).encode("utf-8"),
    "India_Canada_FTA_results.csv",
    "text/csv",
)
with st.expander("Methodology, assumptions and guardrails"):
    st.markdown(
        f"""**Orientation.** Canada is the importing market and India is the beneficiary exporter. 2023 MFN rows are used as the baseline; duplicate AHS rows are not double-counted.

**Trade creation.** `India baseline exports × |import demand elasticity| × (initial tariff − scenario tariff)/(1 + initial tariff)`.

**Trade diversion.** A transparent CES/Armington market-share reallocation is applied using substitution elasticity **{arm:.2f}** and all non-India suppliers as the competing supply pool.

**Price effect.** Export supply elasticity **{supply:.1f}**. At 99, India is treated as a price taker and price effect is zero.

**Total effect.** Trade creation + trade diversion + price effect. With elasticity 99, total effect equals trade creation + trade diversion.

**Important guardrail.** This is an auditable prototype, not a certified replication of the proprietary/live WITS SMART calculation. Benchmark selected HS6 outputs against WITS before using results in an official negotiation brief. Missing elasticities are explicitly flagged."""
    )
if f["Elasticity_Used"].isna().any():
    st.warning(
        f"{f['Elasticity_Used'].isna().sum():,} India HS6 lines are excluded because import demand elasticity is missing."
    )
