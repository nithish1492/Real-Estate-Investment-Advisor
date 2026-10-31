"""Exploratory Data Analysis: answers the 20 questions in the project brief.

Run:  python -m src.eda
Outputs: eda_outputs/q01_*.png ... q20_*.png and eda_outputs/eda_findings.json
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

plt.rcParams.update({"figure.figsize": (8, 4.5), "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25,
                     "font.size": 10})
COL = "#1f6f8b"
OUT = C.EDA_DIR
findings: dict = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=110)
    plt.close(fig)


def bar(series, name, title, xlabel="", ylabel="", horizontal=False, fmt="{:.1f}"):
    fig, ax = plt.subplots()
    if horizontal:
        series.sort_values().plot.barh(ax=ax, color=COL)
    else:
        series.plot.bar(ax=ax, color=COL, rot=45)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    save(fig, name)


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(C.CLEAN_CSV)
    ppsf, price, size = "Price_per_SqFt_INR", "Price_in_Lakhs", "Size_in_SqFt"

    # ---------- 1-5 Price & size ----------
    fig, ax = plt.subplots(); ax.hist(df[price], bins=50, color=COL)
    ax.axvline(df[price].median(), color="crimson", ls="--", label=f"median {df[price].median():.0f}L")
    ax.set_title("Q1. Distribution of property prices"); ax.set_xlabel("Price (Lakhs)"); ax.legend()
    save(fig, "q01_price_distribution")
    findings["Q1"] = (f"Prices range {df[price].min():.0f}-{df[price].max():.0f} lakhs with a near-uniform "
                      f"spread (median {df[price].median():.0f}L, skew {df[price].skew():.2f}); "
                      "no long right tail typical of real housing markets.")

    fig, ax = plt.subplots(); ax.hist(df[size], bins=50, color="#5b8e55")
    ax.set_title("Q2. Distribution of property sizes"); ax.set_xlabel("Size (sq ft)")
    save(fig, "q02_size_distribution")
    findings["Q2"] = (f"Sizes span {df[size].min():.0f}-{df[size].max():.0f} sq ft, roughly uniform "
                      f"(median {df[size].median():.0f}).")

    fig, ax = plt.subplots(); df.boxplot(column=ppsf, by="Property_Type", ax=ax, grid=False)
    ax.set_title("Q3. Price per sq ft by property type"); plt.suptitle(""); ax.set_ylabel("INR / sq ft")
    save(fig, "q03_ppsf_by_type")
    g = df.groupby("Property_Type")[ppsf].median().round(0)
    findings["Q3"] = f"Median INR/sqft by type: {g.to_dict()} - type has almost no effect."

    samp = df.sample(8000, random_state=1)
    fig, ax = plt.subplots(); ax.scatter(samp[size], samp[price], s=4, alpha=.3, color=COL)
    r = df[size].corr(df[price])
    ax.set_title(f"Q4. Size vs price (r = {r:.3f})"); ax.set_xlabel("Size (sq ft)"); ax.set_ylabel("Price (L)")
    save(fig, "q04_size_vs_price")
    findings["Q4"] = (f"Correlation between size and price is {r:.3f}, i.e. essentially none. In real markets "
                      "this would be strongly positive, which signals the dataset is synthetic.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].boxplot(df[ppsf], vert=False); axes[0].set_title("Price per sq ft (INR)")
    axes[1].boxplot(df[size], vert=False); axes[1].set_title("Size (sq ft)")
    fig.suptitle("Q5. Outlier check (IQR rule)")
    save(fig, "q05_outliers")
    findings["Q5"] = (f"IQR outliers: {int(df[f'{ppsf}_Outlier'].sum()):,} rows "
                      f"({df[f'{ppsf}_Outlier'].mean():.1%}) in price/sqft, "
                      f"{int(df[f'{size}_Outlier'].sum())} in size. They arise from small flats with high "
                      "price, flagged but retained.")

    # ---------- 6-10 Location ----------
    s = df.groupby("State")[ppsf].mean().sort_values()
    bar(s, "q06_ppsf_by_state", "Q6. Average price per sq ft by state", ylabel="INR / sq ft", horizontal=True)
    findings["Q6"] = f"Highest: {s.idxmax()} ({s.max():,.0f}); lowest: {s.idxmin()} ({s.min():,.0f}) INR/sqft."

    s = df.groupby("City")[price].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 9)); s.plot.barh(ax=ax, color=COL)
    ax.set_title("Q7. Average property price by city (Lakhs)")
    ax.set_xlim(s.min() - 5, s.max() + 2)
    save(fig, "q07_price_by_city")
    findings["Q7"] = (f"City averages only span {s.min():.0f}-{s.max():.0f} lakhs "
                      f"(top {s.idxmax()}, bottom {s.idxmin()}): geography barely moves price here.")

    loc_age = df.groupby("Locality")["Age_of_Property"].median()
    fig, ax = plt.subplots(); ax.hist(loc_age, bins=20, color="#c9822b")
    ax.set_title("Q8. Median property age by locality (500 localities)"); ax.set_xlabel("Median age (years)")
    save(fig, "q08_median_age_locality")
    findings["Q8"] = (f"Locality median age ranges {loc_age.min():.0f}-{loc_age.max():.0f} yrs; overall median "
                      f"{df.Age_of_Property.median():.0f}. Oldest: {loc_age.idxmax()}.")

    ct = pd.crosstab(df["City"], df["BHK"], normalize="index")
    fig, ax = plt.subplots(figsize=(8, 9)); im = ax.imshow(ct.values, aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(ct))); ax.set_yticklabels(ct.index, fontsize=7)
    ax.set_xticks(range(len(ct.columns))); ax.set_xticklabels(ct.columns); ax.set_xlabel("BHK")
    ax.set_title("Q9. BHK mix by city (share of listings)"); fig.colorbar(im); ax.grid(False)
    save(fig, "q09_bhk_by_city")
    findings["Q9"] = "BHK 1-5 are about evenly split (~20% each) in every city."

    top_loc = df.groupby("Locality")[price].mean().nlargest(5).index
    t = df[df.Locality.isin(top_loc)].groupby(["Year_Built", "Locality"])[ppsf].mean().unstack()
    fig, ax = plt.subplots(); t.rolling(5, min_periods=1).mean().plot(ax=ax)
    ax.set_title("Q10. Price/sqft by construction year: 5 most expensive localities (5-yr rolling)")
    ax.set_ylabel("INR / sq ft")
    save(fig, "q10_top5_locality_trend")
    findings["Q10"] = (f"Top-5 localities by mean price: {list(top_loc)}. The data has no transaction "
                       "dates, so trend is shown by Year_Built, and it is noisy with no clear upward drift.")

    # ---------- 11-15 Correlations ----------
    num = ["BHK", size, price, ppsf, "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
           "Floor_No", "Total_Floors", "Infrastructure_Score", "Amenity_Count"]
    corr = df[num].corr()
    fig, ax = plt.subplots(figsize=(9, 7)); im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num))); ax.set_xticklabels(num, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(num))); ax.set_yticklabels(num, fontsize=8); ax.grid(False)
    for i in range(len(num)):
        for j in range(len(num)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im); ax.set_title("Q11. Correlation matrix")
    save(fig, "q11_correlation_heatmap")
    findings["Q11"] = (f"Only structural relationship: price/sqft vs size ({corr.loc[ppsf, size]:.2f}) and vs "
                       f"price ({corr.loc[ppsf, price]:.2f}), both by construction (ratio). Other sizeable pairs are also engineered: Floor_No vs Total_Floors (0.60, after the floor fix) and Infrastructure_Score vs schools/hospitals (0.58/0.48). Every pair of raw attributes is uncorrelated (|r|<0.01).")

    for q, col, nm in [("12", "Nearby_Schools", "schools"), ("13", "Nearby_Hospitals", "hospitals")]:
        g = df.groupby(col)[ppsf].mean()
        fig, ax = plt.subplots(); ax.plot(g.index, g.values, "o-", color=COL)
        ax.set_title(f"Q{q}. Nearby {nm} vs mean price/sqft (r = {df[col].corr(df[ppsf]):.3f})")
        ax.set_xlabel(col); ax.set_ylabel("INR / sq ft")
        save(fig, f"q{q}_{nm}_vs_ppsf")
        findings[f"Q{q}"] = (f"r = {df[col].corr(df[ppsf]):.3f}; mean price/sqft is flat across 1-10 {nm}.")

    g = df.groupby("Furnished_Status")[[price, ppsf]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    g[price].plot.bar(ax=axes[0], color=COL, rot=0, title="Mean price (L)")
    g[ppsf].plot.bar(ax=axes[1], color="#c9822b", rot=0, title="Mean price/sqft (INR)")
    fig.suptitle("Q14. Price by furnished status"); save(fig, "q14_furnished_status")
    findings["Q14"] = f"Mean price by furnishing: {g[price].round(1).to_dict()} lakhs - differences are negligible."

    g = df.groupby("Facing")[ppsf].mean()
    bar(g, "q15_facing_ppsf", "Q15. Price per sq ft by facing direction", ylabel="INR / sq ft")
    findings["Q15"] = f"Mean INR/sqft by facing: {g.round(0).to_dict()} - no direction premium."

    # ---------- 16-20 Investment / amenities / ownership ----------
    g = df["Owner_Type"].value_counts()
    bar(g, "q16_owner_type", "Q16. Properties per owner type", ylabel="count")
    findings["Q16"] = f"Owner types: {g.to_dict()}."
    g = df["Availability_Status"].value_counts()
    bar(g, "q17_availability", "Q17. Properties per availability status", ylabel="count")
    findings["Q17"] = f"Availability: {g.to_dict()}."

    g = df.groupby("Parking_Space")[[price, ppsf]].mean()
    g.index = g.index.map({0: "No parking", 1: "Parking"})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    g[price].plot.bar(ax=axes[0], color=COL, rot=0, title="Mean price (L)")
    g[ppsf].plot.bar(ax=axes[1], color="#c9822b", rot=0, title="Mean price/sqft (INR)")
    fig.suptitle("Q18. Effect of parking on price"); save(fig, "q18_parking")
    findings["Q18"] = f"Parking vs none: {g[price].round(1).to_dict()} lakhs - no meaningful effect."

    rows = []
    for a in ["Playground", "Gym", "Garden", "Pool", "Clubhouse"]:
        rows.append((a, df[df[f"Has_{a}"] == 1][ppsf].mean(), df[df[f"Has_{a}"] == 0][ppsf].mean()))
    am = pd.DataFrame(rows, columns=["Amenity", "With", "Without"]).set_index("Amenity")
    fig, ax = plt.subplots(); am.plot.bar(ax=ax, rot=0, color=[COL, "#c9a227"])
    ax.set_title("Q19. Price per sq ft with / without each amenity"); ax.set_ylabel("INR / sq ft")
    save(fig, "q19_amenities")
    findings["Q19"] = ("Largest with-vs-without gap: "
                       f"{(am.With - am.Without).abs().idxmax()} ({(am.With - am.Without).abs().max():.0f} INR/sqft, "
                       "<1%): amenities do not drive price in this data.")

    g = df.groupby("Public_Transport_Accessibility").agg(ppsf=(ppsf, "mean"), good=("Good_Investment", "mean"))
    g = g.loc[["Low", "Medium", "High"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    g.ppsf.plot.bar(ax=axes[0], color=COL, rot=0, title="Mean price/sqft (INR)")
    (g.good * 100).plot.bar(ax=axes[1], color="#5b8e55", rot=0, title="% Good Investment")
    fig.suptitle("Q20. Public transport accessibility"); save(fig, "q20_transport")
    findings["Q20"] = (f"Price/sqft is flat across transport levels, but Good-Investment share rises from "
                       f"{g.good['Low']:.1%} (Low) to {g.good['High']:.1%} (High), because transport feeds "
                       "the investment score by design.")

    # ---------- Extra: crime-rate request in brief is not answerable ----------
    findings["Note_crime"] = ("The brief mentions crime rate, but the dataset has no crime column; "
                              "Security and Infrastructure_Score are used as proxies.")
    gi = df.groupby("Security")["Good_Investment"].mean()
    findings["Security_vs_GoodInvestment"] = {"No": round(gi[0], 4), "Yes": round(gi[1], 4)}
    c2 = df.groupby("City")["Good_Investment"].mean().sort_values()
    bar(c2 * 100, "extra_good_investment_by_city", "Share of 'Good Investment' by city (%)",
        ylabel="%", horizontal=False)
    corr_ret = df[[size, "BHK", "Infrastructure_Score", "Growth_Rate"]].corr()["Growth_Rate"].round(3).to_dict()
    findings["Extra_corr_with_growth"] = corr_ret

    (OUT / "eda_findings.json").write_text(json.dumps(findings, indent=2))
    print(f"EDA done: {len(list(OUT.glob('*.png')))} charts, findings saved.")


if __name__ == "__main__":
    main()
