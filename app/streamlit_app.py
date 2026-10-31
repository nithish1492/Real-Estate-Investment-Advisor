"""Real Estate Investment Advisor: Streamlit app.

Run from the project root:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.predict import load_artifacts, predict

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")

TEAL, AMBER, GREEN, RED = "#1f6f8b", "#c9822b", "#2e8b57", "#c0392b"
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
.verdict {border-radius: 14px; padding: 22px 26px; color: white; margin-bottom: 8px;}
.verdict h2 {margin: 0; color: white;} .verdict p {margin: 4px 0 0 0; opacity: .92;}
.kpi {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 14px 18px;}
.kpi small {opacity: .7;} .kpi h3 {margin: 2px 0 0 0;}
</style>""", unsafe_allow_html=True)

NUM_COLS = ["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
            "Price_per_SqFt_INR", "Age_of_Property", "Furnished_Status", "Facing", "Owner_Type",
            "Availability_Status", "Public_Transport_Accessibility", "Parking_Space", "Security",
            "Nearby_Schools", "Nearby_Hospitals", "Infrastructure_Score", "Amenity_Count",
            "Growth_Rate", "Future_Price_5Y_Lakhs", "Good_Investment", "Investment_Score"]


@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(C.CLEAN_CSV, usecols=NUM_COLS)
    for c in ["State", "City", "Locality", "Property_Type", "Furnished_Status", "Facing", "Owner_Type",
              "Availability_Status", "Public_Transport_Accessibility"]:
        df[c] = df[c].astype("category")
    return df


@st.cache_resource
def artifacts():
    return load_artifacts()


for f in [C.CLEAN_CSV, C.MODEL_DIR / "classification_model.joblib"]:
    if not f.exists():
        st.error(f"Missing `{f.name}`. Run `python -m src.train` first (see README).")
        st.stop()

df = load_data()
_, _, ref, metrics = artifacts()
city_state = df.drop_duplicates("City").set_index("City")["State"].astype(str).to_dict()

PAGES = ["🏠 Home", "💡 Investment Advisor", "🔎 Property Explorer", "📊 Market Insights",
         "🧪 Model Performance", "ℹ️ About"]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("Real Estate Investment Advisor\n\nPredicting profitability & 5-year value")


def kpi(label, value, sub=""):
    st.markdown(f'<div class="kpi"><small>{label}</small><h3>{value}</h3><small>{sub}</small></div>',
                unsafe_allow_html=True)


# ============================================================================ HOME
if page == PAGES[0]:
    st.title("🏠 Real Estate Investment Advisor")
    st.write("Answer two questions about any property: **is it a good investment**, and "
             "**what will it be worth in 5 years**?")
    c = st.columns(4)
    with c[0]: kpi("Properties analysed", f"{len(df):,}")
    with c[1]: kpi("Cities / states", f"{df.City.nunique()} / {df.State.nunique()}")
    with c[2]: kpi("Median price", f"₹{df.Price_in_Lakhs.median():,.0f} L")
    with c[3]: kpi("Good-investment share", f"{df.Good_Investment.mean():.0%}")
    st.markdown("")
    l, r = st.columns(2)
    with l:
        g = df.groupby("City", observed=True)["Good_Investment"].mean().sort_values(ascending=False).head(10)
        fig = px.bar(g[::-1] * 100, orientation="h", color_discrete_sequence=[TEAL],
                     labels={"value": "% rated good investment", "City": ""},
                     title="Top 10 cities by share of good investments")
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, width="stretch")
    with r:
        fig = px.histogram(df.sample(30000, random_state=1), x="Future_Price_5Y_Lakhs", nbins=50,
                           color_discrete_sequence=[AMBER], title="Projected 5-year price distribution (₹ lakhs)")
        fig.update_layout(height=380, yaxis_title="properties")
        st.plotly_chart(fig, width="stretch")
    st.info("Use **Investment Advisor** to evaluate a specific property, **Property Explorer** to filter "
            "listings, and **Market Insights** for location heatmaps and trends.")

# ============================================================================ ADVISOR
elif page == PAGES[1]:
    st.title("💡 Investment Advisor")
    st.caption("Describe a property; the models return a verdict with confidence and a 5-year price forecast.")
    cities = sorted(city_state)
    with st.form("prop"):
        a, b, c3 = st.columns(3)
        with a:
            city = st.selectbox("City", cities, index=cities.index("Bangalore") if "Bangalore" in cities else 0)
            ptype = st.selectbox("Property type", sorted(df.Property_Type.unique()))
            bhk = st.slider("BHK", 1, 5, 3)
            size = st.number_input("Size (sq ft)", 300, 8000, 1500, step=50)
            price = st.number_input("Asking price (₹ lakhs)", 5.0, 1000.0, 150.0, step=5.0)
        with b:
            age = st.slider("Age of property (years)", 0, 50, 10)
            floor = st.number_input("Floor number", 0, 60, 5)
            tfloors = st.number_input("Total floors in building", 1, 60, 12)
            furn = st.selectbox("Furnishing", sorted(df.Furnished_Status.unique()))
            facing = st.selectbox("Facing", sorted(df.Facing.unique()))
        with c3:
            schools = st.slider("Nearby schools", 1, 10, 6)
            hosp = st.slider("Nearby hospitals", 1, 10, 5)
            transport = st.select_slider("Public transport access", ["Low", "Medium", "High"], value="Medium")
            status = st.selectbox("Availability", sorted(df.Availability_Status.unique()))
            owner = st.selectbox("Seller type", sorted(df.Owner_Type.unique()))
        d1, d2, d3 = st.columns([1, 1, 3])
        parking = d1.checkbox("Parking", True)
        security = d2.checkbox("Security", True)
        amen = d3.multiselect("Amenities", ["Playground", "Gym", "Garden", "Pool", "Clubhouse"],
                              default=["Gym", "Garden"])
        go_btn = st.form_submit_button("Analyse property", type="primary", width="stretch")

    if go_btn:
        res = predict(dict(
            State=city_state[city], City=city, Property_Type=ptype, BHK=bhk, Size_in_SqFt=size,
            Price_in_Lakhs=price, Age_of_Property=age, Furnished_Status=furn, Floor_No=floor,
            Total_Floors=tfloors, Nearby_Schools=schools, Nearby_Hospitals=hosp,
            Public_Transport_Accessibility=transport, Parking_Space=parking, Security=security,
            Amenities=amen or ["None"], Facing=facing, Owner_Type=owner, Availability_Status=status))
        col = GREEN if res["is_good"] else RED
        verdict = "Good investment" if res["is_good"] else "Not a good investment"
        st.markdown(f'<div class="verdict" style="background:{col}"><h2>{"✅" if res["is_good"] else "⚠️"} {verdict}</h2>'
                    f'<p>Calibrated model estimate · P(good) = {res["good_probability"]:.0%}</p></div>',
                    unsafe_allow_html=True)
        st.caption("This probability is calibrated against the project's rule-generated labels; it is not a verified real-world chance of investment success.")
        k = st.columns(4)
        with k[0]: kpi("Price today", f"₹{price:,.1f} L")
        with k[1]: kpi("Estimated price in 5 yrs", f"₹{res['future_price']:,.1f} L", f"+₹{res['gain']:,.1f} L")
        with k[2]: kpi("Implied annual growth", f"{res['cagr']:.2%}", f"vs fixed 8%: ₹{res['fixed_8pct']:,.0f} L")
        with k[3]: kpi("Price / sq ft", f"₹{res['ppsf']:,.0f}",
                       f"{(res['ppsf_ratio'] - 1):+.0%} vs {city} {ptype.lower()} median")
        l, r = st.columns(2)
        with l:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=res["good_probability"] * 100,
                number={"suffix": "%"}, title={"text": "Probability of being a good investment"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": col},
                       "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}))
            fig.update_layout(height=320, margin=dict(t=60, b=10))
            st.plotly_chart(fig, width="stretch")
        with r:
            yrs = list(range(0, 6))
            fig = go.Figure()
            fig.add_scatter(x=yrs, y=res["trajectory"], mode="lines+markers", name="Model forecast",
                            line=dict(color=TEAL, width=3))
            fig.add_scatter(x=yrs, y=[price * 1.08 ** t for t in yrs], mode="lines", name="Fixed 8% baseline",
                            line=dict(color="gray", dash="dash"))
            fig.update_layout(title="Projected value over 5 years", xaxis_title="Years from now",
                              yaxis_title="₹ lakhs", height=320, margin=dict(t=60, b=10))
            st.plotly_chart(fig, width="stretch")
        fi = pd.read_csv(C.MODEL_DIR / "classification_feature_importance.csv", index_col=0).head(10)
        st.subheader("What drives the verdict (global feature importance)")
        fig = px.bar(fi[::-1], orientation="h", color_discrete_sequence=[AMBER], labels={"value": "importance", "index": ""})
        fig.update_layout(showlegend=False, height=340)
        st.plotly_chart(fig, width="stretch")
        st.caption("Forecasts come from a model trained on a rule-based growth assumption (see About). "
                   "Treat them as scenario estimates, not financial advice.")

# ============================================================================ EXPLORER
elif page == PAGES[2]:
    st.title("🔎 Property Explorer")
    with st.sidebar:
        st.subheader("Filters")
        states = st.multiselect("State", sorted(df.State.unique()))
        pool = df[df.State.isin(states)] if states else df
        cities = st.multiselect("City", sorted(pool.City.unique()))
        types = st.multiselect("Property type", sorted(df.Property_Type.unique()))
        bhk = st.slider("BHK", 1, 5, (1, 5))
        price = st.slider("Price (₹ lakhs)", 10, 500, (10, 500))
        size = st.slider("Size (sq ft)", 500, 5000, (500, 5000), step=50)
        status = st.multiselect("Availability", sorted(df.Availability_Status.unique()))
        only_good = st.checkbox("Only good investments")
    m = df.BHK.between(*bhk) & df.Price_in_Lakhs.between(*price) & df.Size_in_SqFt.between(*size)
    if states: m &= df.State.isin(states)
    if cities: m &= df.City.isin(cities)
    if types: m &= df.Property_Type.isin(types)
    if status: m &= df.Availability_Status.isin(status)
    if only_good: m &= df.Good_Investment == 1
    f = df[m]
    c = st.columns(4)
    with c[0]: kpi("Matching properties", f"{len(f):,}")
    with c[1]: kpi("Median price", f"₹{f.Price_in_Lakhs.median():,.0f} L" if len(f) else "–")
    with c[2]: kpi("Median ₹/sq ft", f"{f.Price_per_SqFt_INR.median():,.0f}" if len(f) else "–")
    with c[3]: kpi("Good investment", f"{f.Good_Investment.mean():.0%}" if len(f) else "–")
    if len(f):
        l, r = st.columns(2)
        s = f.sample(min(5000, len(f)), random_state=1)
        with l:
            fig = px.scatter(s, x="Size_in_SqFt", y="Price_in_Lakhs", color=s.Good_Investment.map({1: "Good", 0: "Not good"}),
                             color_discrete_map={"Good": GREEN, "Not good": RED}, opacity=.5,
                             title="Size vs price", labels={"color": ""})
            st.plotly_chart(fig, width="stretch")
        with r:
            fig = px.histogram(f, x="Price_per_SqFt_INR", nbins=40, color_discrete_sequence=[TEAL],
                               title="Price per sq ft distribution")
            st.plotly_chart(fig, width="stretch")
        show = ["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
                "Price_per_SqFt_INR", "Availability_Status", "Furnished_Status", "Growth_Rate",
                "Future_Price_5Y_Lakhs", "Good_Investment"]
        top = f.sort_values("Investment_Score", ascending=False).head(500)[show]
        st.subheader("Top 500 matches by investment score")
        st.dataframe(top, width="stretch", hide_index=True)
        st.download_button("Download matches (CSV)", f.head(50000)[show].to_csv(index=False), "matches.csv")
    else:
        st.warning("No properties match these filters.")

# ============================================================================ INSIGHTS
elif page == PAGES[3]:
    st.title("📊 Market Insights")
    t1, t2, t3, t4 = st.tabs(["Heatmap", "City ranking", "Appreciation", "EDA gallery"])
    with t1:
        metric = st.selectbox("Metric", ["Price_per_SqFt_INR", "Price_in_Lakhs", "Growth_Rate", "Good_Investment"])
        pv = df.pivot_table(index="State", columns="Property_Type", values=metric, aggfunc="mean", observed=True)
        fig = px.imshow(pv, color_continuous_scale="Teal", aspect="auto", text_auto=".3g",
                        title=f"Mean {metric} by state × property type")
        fig.update_layout(height=620)
        st.plotly_chart(fig, width="stretch")
        pv2 = df.pivot_table(index="City", columns="BHK", values=metric, aggfunc="mean", observed=True)
        fig = px.imshow(pv2, color_continuous_scale="Teal", aspect="auto", title=f"Mean {metric} by city × BHK")
        fig.update_layout(height=900)
        st.plotly_chart(fig, width="stretch")
    with t2:
        g = df.groupby("City", observed=True).agg(avg_price=("Price_in_Lakhs", "mean"),
            ppsf=("Price_per_SqFt_INR", "mean"), good=("Good_Investment", "mean"),
            growth=("Growth_Rate", "mean")).reset_index()
        g["good"] *= 100; g["growth"] *= 100
        fig = px.scatter(g, x="ppsf", y="growth", size="avg_price", color="good", hover_name="City",
                         color_continuous_scale="RdYlGn", title="Cities: price/sq ft vs projected annual growth",
                         labels={"ppsf": "Avg ₹/sq ft", "growth": "Avg growth % p.a.", "good": "% good"})
        fig.update_layout(height=520)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(g.sort_values("good", ascending=False).round(2), hide_index=True, width="stretch")
    with t3:
        g = df.groupby("Year_Built", observed=True) if "Year_Built" in df else None
        a = df.groupby(["City"], observed=True)[["Price_in_Lakhs", "Future_Price_5Y_Lakhs"]].mean().reset_index()
        a["Gain %"] = (a.Future_Price_5Y_Lakhs / a.Price_in_Lakhs - 1) * 100
        fig = px.bar(a.sort_values("Gain %", ascending=False), x="City", y="Gain %",
                     color_discrete_sequence=[TEAL], title="Projected 5-year price gain by city (%)")
        st.plotly_chart(fig, width="stretch")
    with t4:
        pics = sorted(C.EDA_DIR.glob("q*.png"))
        names = [p.stem for p in pics]
        pick = st.selectbox("EDA question", names)
        st.image(str(C.EDA_DIR / f"{pick}.png"))
        fj = C.EDA_DIR / "eda_findings.json"
        if fj.exists():
            key = pick.split("_")[0].upper().replace("Q0", "Q")
            st.info(json.load(open(fj)).get(key, ""))

# ============================================================================ PERFORMANCE
elif page == PAGES[4]:
    st.title("🧪 Model Performance")
    cm, rm = metrics["classification"], metrics["regression"]
    meta = metrics["meta"]
    st.caption(f"Held-out test set: {meta['test_rows']:,} properties · champions: "
               f"**{cm['_champion']}** (classifier), **{rm['_champion']}** (regressor)")
    st.subheader("Classification: Good Investment")
    t = pd.DataFrame({k: v for k, v in cm.items() if k != "_champion"}).T.drop(columns="confusion_matrix")
    st.dataframe(t.round(4), width="stretch")
    champ = cm["_champion"]
    c1, c2 = st.columns(2)
    with c1:
        z = cm[champ]["confusion_matrix"]
        fig = px.imshow(z, text_auto=True, x=["Pred: Not good", "Pred: Good"], y=["Actual: Not good", "Actual: Good"],
                        color_continuous_scale="Blues", title=f"{champ} confusion matrix")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fi = pd.read_csv(C.MODEL_DIR / "classification_feature_importance.csv", index_col=0).head(12)
        fig = px.bar(fi[::-1], orientation="h", color_discrete_sequence=[AMBER], title="Classifier feature importance")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    st.subheader("Regression: price in 5 years")
    t = pd.DataFrame({k: v for k, v in rm.items() if k != "_champion"}).T
    st.dataframe(t.round(4), width="stretch")
    fi = pd.read_csv(C.MODEL_DIR / "regression_feature_importance.csv", index_col=0).head(12)
    fig = px.bar(fi[::-1], orientation="h", color_discrete_sequence=[TEAL], title="Regressor feature importance")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.warning("Both targets are derived from rules applied to the dataset's own features, so near-perfect scores "
               "show the models recover those rules; they are not evidence of real-world predictive power. "
               "See About → Limitations.")

# ============================================================================ ABOUT
else:
    st.title("ℹ️ About & Methodology")
    md = ROOT / "docs" / "METHODOLOGY.md"
    st.markdown(md.read_text(encoding="utf-8") if md.exists() else "Methodology document not found.")
