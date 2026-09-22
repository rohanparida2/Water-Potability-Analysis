"""
RohanParida_WaterQualityAnalysis.py
Water Quality Analysis & Potability Assessment — Streamlit Dashboard
Run: streamlit run RohanParida_WaterQualityAnalysis.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – required for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Water Quality Analysis — Rohan Parida",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 110,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})
sns.set_palette("Set2")

COLORS = {0: "#e74c3c", 1: "#2ecc71"}
LABELS = {0: "Non-Potable", 1: "Potable"}

PARAM_UNITS = {
    "ph": "pH units", "Hardness": "mg/L", "Solids": "mg/L",
    "Chloramines": "ppm", "Sulfate": "mg/L", "Conductivity": "μS/cm",
    "Organic_carbon": "ppm", "Trihalomethanes": "μg/L", "Turbidity": "NTU",
}
PARAM_LABELS = {
    "ph": "pH", "Hardness": "Hardness", "Solids": "Solids (TDS)",
    "Chloramines": "Chloramines", "Sulfate": "Sulfate",
    "Conductivity": "Conductivity", "Organic_carbon": "Organic Carbon",
    "Trihalomethanes": "Trihalomethanes", "Turbidity": "Turbidity",
}
FEATURES = list(PARAM_LABELS.keys())

os.makedirs("report_images", exist_ok=True)

# ── Data loading & cleaning (cached) ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("water_potability.csv")
    df_clean = df.copy()
    for col in ["ph", "Sulfate", "Trihalomethanes"]:
        group_medians = df_clean.groupby("Potability")[col].median()
        for pv in [0, 1]:
            mask = (df_clean["Potability"] == pv) & (df_clean[col].isnull())
            df_clean.loc[mask, col] = group_medians[pv]
    return df, df_clean

df_raw, df = load_data()

potable     = df[df["Potability"] == 1]
non_potable = df[df["Potability"] == 0]

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/water.png", width=60)
st.sidebar.title("💧 Water Quality Analysis")
st.sidebar.markdown("**Rohan Parida**")
st.sidebar.markdown("---")

SECTIONS = [
    "🏠 Overview",
    "📋 Dataset Inspection",
    "🧹 Data Cleaning",
    "📊 Descriptive Statistics",
    "🔵 Potability Analysis",
    "📈 Parameter Distributions",
    "📦 Boxplots & Violin Plots",
    "🔍 Outlier Analysis",
    "🔗 Correlation Analysis",
    "📐 Statistical Testing",
    "📋 Comparison Table",
    "🔎 EDA Scatter & Pair Plots",
    "📝 Key Findings & Conclusion",
]
section = st.sidebar.radio("Navigate to", SECTIONS)
st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: `water_potability.csv`  \n{df.shape[0]:,} rows × {df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════════════════════
# Helper: close every figure after displaying to avoid memory leaks
# ══════════════════════════════════════════════════════════════════════════════
def show(fig):
    st.pyplot(fig)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION: Overview
# ══════════════════════════════════════════════════════════════════════════════
if section == "🏠 Overview":
    st.title("💧 Water Quality Analysis & Potability Assessment")
    st.markdown("### Rohan Parida — Data Analytics Project")
    st.markdown("---")
    st.markdown("""
This interactive dashboard presents a **complete Exploratory Data Analysis (EDA)**
of the Water Potability dataset containing **3,276 water samples** across
**nine physicochemical parameters**.

**Analysis pipeline:**
1. Dataset loading & inspection
2. Data cleaning & missing-value treatment
3. Descriptive statistics
4. Potability analysis
5. Parameter-wise distribution analysis
6. Boxplots & violin plots (potable vs non-potable)
7. Outlier detection (IQR method)
8. Correlation analysis
9. Statistical significance testing (Mann-Whitney U)
10. Comparison table & key findings

Use the **sidebar** to navigate between sections.
    """)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", f"{df.shape[0]:,}")
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Potable", f"{(df['Potability']==1).sum():,} ({(df['Potability']==1).mean()*100:.1f}%)")
    col4.metric("Non-Potable", f"{(df['Potability']==0).sum():,} ({(df['Potability']==0).mean()*100:.1f}%)")
    st.info("📌 Navigate using the sidebar on the left.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION: Dataset Inspection
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📋 Dataset Inspection":
    st.title("📋 Dataset Inspection")
    st.markdown("---")

    st.subheader("First 10 Rows")
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.subheader("Shape & Data Types")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Rows:** {df_raw.shape[0]:,}  \n**Columns:** {df_raw.shape[1]}")
        dtype_df = df_raw.dtypes.reset_index()
        dtype_df.columns = ["Column", "Dtype"]
        st.dataframe(dtype_df, use_container_width=True)
    with col2:
        st.subheader("Missing Values")
        missing = df_raw.isnull().sum()
        missing_pct = (missing / len(df_raw) * 100).round(2)
        mv_df = pd.DataFrame({"Missing Count": missing, "Missing (%)": missing_pct})
        mv_df = mv_df[mv_df["Missing Count"] > 0]
        st.dataframe(mv_df.style.background_gradient(cmap="Reds"), use_container_width=True)
        st.markdown(f"**Duplicate rows:** {df_raw.duplicated().sum()}")

    st.subheader("Missing Value Visualisation")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    mc = missing[missing > 0]
    axes[0].bar(mc.index, mc.values, color=["#e74c3c","#e67e22","#3498db"],
                edgecolor="black", linewidth=0.8)
    for i, (col, val) in enumerate(mc.items()):
        axes[0].text(i, val + 8, f"{val}\n({val/len(df_raw)*100:.1f}%)",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].set_title("Missing Values per Column", fontweight="bold")
    axes[0].set_ylabel("Count"); axes[0].set_xlabel("Column")
    sns.heatmap(df_raw[["ph","Sulfate","Trihalomethanes"]].isnull(),
                ax=axes[1], cbar=False, cmap="Reds", yticklabels=False)
    axes[1].set_title("Missingness Heatmap", fontweight="bold")
    axes[1].set_xlabel("Column"); axes[1].set_ylabel("Row Index")
    plt.tight_layout()
    show(fig)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION: Data Cleaning
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🧹 Data Cleaning":
    st.title("🧹 Data Cleaning")
    st.markdown("---")
    st.markdown("""
**Strategy: Grouped Median Imputation**

Three columns contain missing values: `ph` (491), `Sulfate` (781), `Trihalomethanes` (162).

Each missing value is replaced by the **median of its respective Potability group** (Potable / Non-Potable).
This preserves inter-group distributional differences better than global median imputation.
    """)

    imputation_info = []
    for col in ["ph", "Sulfate", "Trihalomethanes"]:
        gm = df_raw.groupby("Potability")[col].median()
        n_miss = df_raw[col].isnull().sum()
        imputation_info.append({
            "Column": col, "Missing": n_miss,
            "Missing %": f"{n_miss/len(df_raw)*100:.2f}%",
            "Potable Median": round(gm[1], 4),
            "Non-Potable Median": round(gm[0], 4),
        })
    st.dataframe(pd.DataFrame(imputation_info).set_index("Column"), use_container_width=True)

    st.success("✅ After imputation: **0 missing values** remain. No duplicate rows were found.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before imputation — missing counts:**")
        st.dataframe(df_raw.isnull().sum().rename("Missing"), use_container_width=True)
    with col2:
        st.markdown("**After imputation — missing counts:**")
        st.dataframe(df.isnull().sum().rename("Missing"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION: Descriptive Statistics
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📊 Descriptive Statistics":
    st.title("📊 Descriptive Statistics")
    st.markdown("---")
    stats_df = df[FEATURES].describe().T
    stats_df["median"]   = df[FEATURES].median()
    stats_df["skewness"] = df[FEATURES].skew()
    stats_df["kurtosis"] = df[FEATURES].kurtosis()
    stats_df = stats_df.rename(columns={"25%": "Q1", "75%": "Q3"})
    cols_show = ["mean", "median", "std", "min", "max", "Q1", "Q3", "skewness", "kurtosis"]
    stats_df.index = [PARAM_LABELS[c] for c in FEATURES]
    st.dataframe(
        stats_df[cols_show].round(4)
        .style.background_gradient(subset=["mean","std"], cmap="Blues")
        .background_gradient(subset=["skewness"], cmap="RdYlGn_r"),
        use_container_width=True
    )
    st.caption("Skewness > 1 or < -1 indicates a significantly skewed distribution.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION: Potability Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🔵 Potability Analysis":
    st.title("🔵 Potability Analysis")
    st.markdown("---")

    pot_counts = df["Potability"].value_counts()
    pot_pct    = df["Potability"].value_counts(normalize=True) * 100

    col1, col2 = st.columns(2)
    col1.metric("Non-Potable (0)", f"{pot_counts[0]:,}", f"{pot_pct[0]:.2f}%")
    col2.metric("Potable (1)",     f"{pot_counts[1]:,}", f"{pot_pct[1]:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bars = axes[0].bar(["Non-Potable","Potable"], [pot_counts[0], pot_counts[1]],
                       color=[COLORS[0], COLORS[1]], edgecolor="black", linewidth=0.8, width=0.5)
    for bar, cnt, pct in zip(bars, [pot_counts[0], pot_counts[1]], [pot_pct[0], pot_pct[1]]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                     f"{cnt:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[0].set_title("Potability Count Distribution", fontweight="bold")
    axes[0].set_ylabel("Number of Samples"); axes[0].set_ylim(0, 2400)

    wedges, texts, autotexts = axes[1].pie(
        [pot_counts[0], pot_counts[1]], labels=["Non-Potable","Potable"],
        colors=[COLORS[0], COLORS[1]], autopct="%1.1f%%", startangle=140,
        explode=(0.05, 0.05), wedgeprops={"edgecolor":"white","linewidth":2})
    for at in autotexts:
        at.set_fontsize(12); at.set_fontweight("bold")
    axes[1].set_title("Potability Proportion", fontweight="bold")
    plt.suptitle("Potability Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    show(fig)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION: Parameter Distributions
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📈 Parameter Distributions":
    st.title("📈 Parameter Distributions")
    st.markdown("---")
    st.markdown("Overlapping histograms + KDE curves split by Potability class.")

    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    axes = axes.flatten()
    for i, col in enumerate(FEATURES):
        ax = axes[i]
        for pot, grp in df.groupby("Potability"):
            grp[col].plot.hist(ax=ax, bins=40, alpha=0.5, color=COLORS[pot],
                               label=LABELS[pot], density=True, edgecolor="none")
            grp[col].plot.kde(ax=ax, color=COLORS[pot], linewidth=2)
        ax.set_title(PARAM_LABELS[col], fontweight="bold")
        ax.set_xlabel(f"{PARAM_LABELS[col]} ({PARAM_UNITS[col]})")
        ax.set_ylabel("Density"); ax.legend(fontsize=8)
    plt.suptitle("Parameter Distributions — Potable vs Non-Potable",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    show(fig)

    st.markdown("---")
    st.subheader("Individual Parameter Explorer")
    sel_col = st.selectbox("Select a parameter", FEATURES,
                           format_func=lambda c: PARAM_LABELS[c])
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    for pot, grp in df.groupby("Potability"):
        grp[sel_col].plot.hist(ax=ax2, bins=50, alpha=0.55, color=COLORS[pot],
                               label=LABELS[pot], density=True, edgecolor="none")
        grp[sel_col].plot.kde(ax=ax2, color=COLORS[pot], linewidth=2.5)
    ax2.set_title(f"{PARAM_LABELS[sel_col]} — Distribution by Potability", fontweight="bold")
    ax2.set_xlabel(f"{PARAM_LABELS[sel_col]} ({PARAM_UNITS[sel_col]})")
    ax2.set_ylabel("Density"); ax2.legend()
    plt.tight_layout()
    show(fig2)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION: Boxplots & Violin Plots
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📦 Boxplots & Violin Plots":
    st.title("📦 Boxplots & Violin Plots")
    st.markdown("---")

    tab1, tab2 = st.tabs(["Notched Boxplots", "Violin Plots"])

    with tab1:
        fig, axes = plt.subplots(3, 3, figsize=(17, 13))
        axes = axes.flatten()
        for i, col in enumerate(FEATURES):
            ax = axes[i]
            data_plot = [non_potable[col].values, potable[col].values]
            bp = ax.boxplot(data_plot, patch_artist=True, notch=True,
                            medianprops=dict(color="black", linewidth=2),
                            flierprops=dict(marker="o", markersize=2, alpha=0.4))
            for patch, color in zip(bp["boxes"], [COLORS[0], COLORS[1]]):
                patch.set_facecolor(color); patch.set_alpha(0.8)
            ax.set_xticks([1, 2]); ax.set_xticklabels(["Non-Potable","Potable"])
            ax.set_title(PARAM_LABELS[col], fontweight="bold")
            ax.set_ylabel(PARAM_UNITS[col])
        plt.suptitle("Boxplots — Potable vs Non-Potable", fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()
        show(fig)

    with tab2:
        fig, axes = plt.subplots(3, 3, figsize=(17, 13))
        axes = axes.flatten()
        for i, col in enumerate(FEATURES):
            ax = axes[i]
            vp = ax.violinplot([non_potable[col].values, potable[col].values],
                               positions=[1, 2], showmedians=True, showextrema=True)
            for pc, color in zip(vp["bodies"], [COLORS[0], COLORS[1]]):
                pc.set_facecolor(color); pc.set_alpha(0.7)
            ax.set_xticks([1, 2]); ax.set_xticklabels(["Non-Potable","Potable"])
            ax.set_title(PARAM_LABELS[col], fontweight="bold")
            ax.set_ylabel(PARAM_UNITS[col])
        plt.suptitle("Violin Plots — Distribution Shape by Potability",
                     fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()
        show(fig)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION: Outlier Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🔍 Outlier Analysis":
    st.title("🔍 Outlier Analysis (IQR Method)")
    st.markdown("---")
    st.markdown("Fences: **Q1 − 1.5×IQR** and **Q3 + 1.5×IQR**. Points outside are flagged as outliers.")

    outlier_rows = []
    for col in FEATURES:
        Q1 = df[col].quantile(0.25); Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR; upper = Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_rows.append({
            "Parameter": PARAM_LABELS[col], "Q1": round(Q1, 4), "Q3": round(Q3, 4),
            "IQR": round(IQR, 4), "Lower Fence": round(lower, 4), "Upper Fence": round(upper, 4),
            "Outliers": int(n_out), "Outlier %": round(n_out / len(df) * 100, 2),
        })
    out_df = pd.DataFrame(outlier_rows).set_index("Parameter")
    st.dataframe(out_df.style.background_gradient(subset=["Outliers","Outlier %"], cmap="YlOrRd"),
                 use_container_width=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    names = [r["Parameter"] for r in outlier_rows]
    vals  = [r["Outliers"]  for r in outlier_rows]
    pcts  = [r["Outlier %"] for r in outlier_rows]
    bars = ax.bar(names, vals, color="#3498db", edgecolor="black", linewidth=0.8)
    for bar, val, pct in zip(bars, vals, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title("Number of Outliers per Parameter (IQR Method)", fontweight="bold")
    ax.set_ylabel("Outlier Count"); ax.set_xlabel("Parameter")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    show(fig)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION: Correlation Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🔗 Correlation Analysis":
    st.title("🔗 Correlation Analysis")
    st.markdown("---")

    corr = df[FEATURES + ["Potability"]].corr()

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size": 9},
                square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson Correlation Matrix (all features + Potability)",
                 fontweight="bold", fontsize=12, pad=15)
    plt.xticks(rotation=40, ha="right"); plt.tight_layout()
    show(fig)

    st.subheader("Correlation with Potability")
    pot_corr = corr["Potability"].drop("Potability").sort_values(ascending=False).round(4)
    pot_corr_df = pot_corr.reset_index()
    pot_corr_df.columns = ["Feature","Correlation with Potability"]
    pot_corr_df["Feature"] = pot_corr_df["Feature"].map(PARAM_LABELS)
    st.dataframe(
        pot_corr_df.style.background_gradient(subset=["Correlation with Potability"],
                                               cmap="RdBu_r", vmin=-0.1, vmax=0.1),
        use_container_width=True)
    st.caption("All |r| < 0.10 — no single parameter is a strong linear predictor of potability.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION: Statistical Testing
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📐 Statistical Testing":
    st.title("📐 Statistical Significance Testing")
    st.markdown("---")
    st.markdown("""
**Test: Mann-Whitney U (non-parametric, two-sided)**  
Chosen because several features are not normally distributed.  
**Null hypothesis:** the distribution of the parameter is identical in both groups.  
**Significance threshold: α = 0.05**

> ⚠️ Statistical significance ≠ causation.
    """)

    alpha = 0.05
    test_rows = []
    for col in FEATURES:
        u, p = stats.mannwhitneyu(potable[col].dropna(), non_potable[col].dropna(), alternative="two-sided")
        test_rows.append({
            "Parameter": PARAM_LABELS[col], "U-Statistic": round(u, 1),
            "p-value": round(p, 6), "Significant (α=0.05)": "✅ YES" if p < alpha else "❌ NO",
        })

    test_df = pd.DataFrame(test_rows).set_index("Parameter")

    def highlight_sig(row):
        color = "#c6efce" if "YES" in row["Significant (α=0.05)"] else "#ffcccc"
        return [f"background-color: {color}"] * len(row)

    st.dataframe(test_df.style.apply(highlight_sig, axis=1), use_container_width=True)

    # significance bar chart
    log_p = [-np.log10(r["p-value"]) for r in test_rows]
    sig_flags = ["YES" in r["Significant (α=0.05)"] for r in test_rows]
    col_bars = ["#2ecc71" if s else "#e74c3c" for s in sig_flags]
    names = [r["Parameter"] for r in test_rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, log_p, color=col_bars, edgecolor="black", linewidth=0.8)
    ax.axhline(-np.log10(0.05), color="navy", linestyle="--", linewidth=1.5)
    p_green = mpatches.Patch(color="#2ecc71", label="Significant (p < 0.05)")
    p_red   = mpatches.Patch(color="#e74c3c", label="Not significant (p ≥ 0.05)")
    p_line  = plt.Line2D([0],[0], color="navy", linestyle="--", label="α = 0.05 line")
    ax.legend(handles=[p_green, p_red, p_line], fontsize=10)
    ax.set_title("Mann-Whitney U — Statistical Significance per Parameter", fontweight="bold")
    ax.set_ylabel("−log₁₀(p-value)  [higher = more significant]")
    ax.set_xlabel("Parameter")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    show(fig)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION: Comparison Table
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📋 Comparison Table":
    st.title("📋 Potable vs Non-Potable Comparison Table")
    st.markdown("---")

    stat_results = {}
    for col in FEATURES:
        u, p = stats.mannwhitneyu(potable[col].dropna(), non_potable[col].dropna(), alternative="two-sided")
        stat_results[col] = {"p": p, "sig": p < 0.05}

    rows = []
    for col in FEATURES:
        pm   = potable[col].mean(); npm  = non_potable[col].mean()
        pmed = potable[col].median(); npmed = non_potable[col].median()
        rows.append({
            "Parameter":            PARAM_LABELS[col],
            "Potable Mean":         round(pm, 4),
            "Non-Potable Mean":     round(npm, 4),
            "Mean Difference":      round(pm - npm, 4),
            "Potable Median":       round(pmed, 4),
            "Non-Potable Median":   round(npmed, 4),
            "p-value":              round(stat_results[col]["p"], 6),
            "Significant":          "✅" if stat_results[col]["sig"] else "❌",
        })
    comp_df = pd.DataFrame(rows).set_index("Parameter")

    def color_sig(row):
        if row["Significant"] == "✅":
            return ["background-color: #c6efce"] * len(row)
        return [""] * len(row)

    st.dataframe(comp_df.style.apply(color_sig, axis=1), use_container_width=True)

    # relative diff bar chart
    rel_diff = [(r["Potable Mean"] - r["Non-Potable Mean"]) / r["Non-Potable Mean"] * 100
                for r in rows]
    bar_colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in rel_diff]
    param_names = [r["Parameter"] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(param_names, rel_diff, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=1)
    for bar, val in zip(bars, rel_diff):
        ypos = bar.get_height() + 0.03 if val >= 0 else bar.get_height() - 0.25
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{val:+.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title("Relative Mean Difference (Potable − Non-Potable) per Parameter", fontweight="bold")
    ax.set_ylabel("Relative Difference (%)"); ax.set_xlabel("Parameter")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    show(fig)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION: EDA Scatter & Pair Plots
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🔎 EDA Scatter & Pair Plots":
    st.title("🔎 EDA — Scatter & Pair Plots")
    st.markdown("---")

    st.subheader("Scatter Plots")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for pot, grp in df.groupby("Potability"):
        axes[0].scatter(grp["ph"], grp["Turbidity"],
                        c=COLORS[pot], label=LABELS[pot], alpha=0.3, s=12, edgecolors="none")
        axes[1].scatter(grp["Chloramines"], grp["Solids"],
                        c=COLORS[pot], label=LABELS[pot], alpha=0.3, s=12, edgecolors="none")
    axes[0].set_xlabel("pH (pH units)"); axes[0].set_ylabel("Turbidity (NTU)")
    axes[0].set_title("pH vs Turbidity", fontweight="bold"); axes[0].legend()
    axes[1].set_xlabel("Chloramines (ppm)"); axes[1].set_ylabel("Solids / TDS (mg/L)")
    axes[1].set_title("Chloramines vs Solids (TDS)", fontweight="bold"); axes[1].legend()
    plt.suptitle("Scatter Plots — Key Parameter Relationships", fontsize=13, fontweight="bold")
    plt.tight_layout()
    show(fig)

    st.subheader("Pair Plot (selected features)")
    st.info("⏳ Pair plot may take a few seconds to render...")
    selected = ["ph","Hardness","Chloramines","Sulfate","Turbidity","Potability"]
    pair_df = df[selected].copy()
    pair_df["Potability_label"] = pair_df["Potability"].map({0:"Non-Potable",1:"Potable"})
    g = sns.pairplot(pair_df.drop(columns="Potability"),
                     hue="Potability_label",
                     palette={"Non-Potable": COLORS[0], "Potable": COLORS[1]},
                     plot_kws={"alpha": 0.25, "s": 10},
                     diag_kind="kde", height=2.0)
    g.fig.suptitle("Pair Plot — pH, Hardness, Chloramines, Sulfate, Turbidity",
                   y=1.02, fontsize=12, fontweight="bold")
    st.pyplot(g.fig)
    plt.close(g.fig)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION: Key Findings & Conclusion
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📝 Key Findings & Conclusion":
    st.title("📝 Key Findings & Conclusion")
    st.markdown("---")

    pot_pct = df["Potability"].value_counts(normalize=True) * 100
    missing = df_raw.isnull().sum()

    stat_results = {}
    corr = df[FEATURES + ["Potability"]].corr()
    for col in FEATURES:
        u, p = stats.mannwhitneyu(potable[col].dropna(), non_potable[col].dropna(), alternative="two-sided")
        stat_results[col] = {"p": p, "sig": p < 0.05}

    st.subheader("1 · Dataset Overview")
    st.markdown(f"""
- **3,276** water samples with **9 physicochemical features** + 1 binary label.
- Class imbalance: **{pot_pct[0]:.1f}% Non-Potable**, **{pot_pct[1]:.1f}% Potable**.
- Missing values: `ph` ({missing['ph']} rows, {missing['ph']/len(df_raw)*100:.1f}%), `Sulfate` ({missing['Sulfate']} rows, {missing['Sulfate']/len(df_raw)*100:.1f}%), `Trihalomethanes` ({missing['Trihalomethanes']} rows, {missing['Trihalomethanes']/len(df_raw)*100:.1f}%).
- **Zero duplicate rows.**
    """)

    st.subheader("2 · Imputation")
    st.markdown("Grouped median imputation (by Potability) applied to `ph`, `Sulfate`, `Trihalomethanes` to preserve inter-group distributions.")

    st.subheader("3 · Descriptive Statistics")
    st.markdown(f"""
- **pH:** Mean = {df['ph'].mean():.3f}, approximately normally distributed.
- **Solids (TDS):** Highly variable (std = {df['Solids'].std():.0f} mg/L), right-skewed.
- **Conductivity:** Mean {df['Conductivity'].mean():.1f} μS/cm.
- **Turbidity:** Near-normal, mean {df['Turbidity'].mean():.3f} NTU.
    """)

    st.subheader("4 · Parameter Differences (Potable vs Non-Potable)")
    st.markdown(f"""
Potable water has:
- Higher mean **Chloramines**: {potable['Chloramines'].mean():.4f} vs {non_potable['Chloramines'].mean():.4f} ppm
- Higher mean **Solids**: {potable['Solids'].mean():.1f} vs {non_potable['Solids'].mean():.1f} mg/L
- Lower mean **Organic Carbon**: {potable['Organic_carbon'].mean():.4f} vs {non_potable['Organic_carbon'].mean():.4f} ppm
- Nearly identical **pH**: {potable['ph'].mean():.4f} vs {non_potable['ph'].mean():.4f}
    """)

    st.subheader("5 · Statistical Significance (Mann-Whitney U, α = 0.05)")
    for col in FEATURES:
        sig_label = "🟢 SIGNIFICANT" if stat_results[col]["sig"] else "🔴 not significant"
        st.markdown(f"- **{PARAM_LABELS[col]}**: p = `{stat_results[col]['p']:.6f}` → {sig_label}")

    st.subheader("6 · Correlation")
    st.markdown(f"""
- All feature–Potability correlations: **|r| < 0.10** — no strong linear predictor.
- Solids & Conductivity have the strongest inter-feature correlation (r ≈ {corr.loc['Solids','Conductivity']:.3f}).
    """)

    st.subheader("7 · Outliers")
    st.markdown("""
- **Solids (TDS)** contains the most outliers (right-skewed, extreme high-TDS samples).
- **pH** has outliers near 0 and 14 (extreme acidic/alkaline readings).
    """)

    st.subheader("⚠️ Important Caveat")
    st.warning("Statistical differences between groups do **not** imply causation. Multiple parameters must be considered together for potability assessment.")

    st.markdown("---")
    st.subheader("Conclusion")
    st.markdown("""
This analysis examined **3,276 water samples** across nine physicochemical parameters.

- Distributions of most parameters **overlap substantially** between Potable and Non-Potable classes — individual parameters are weak discriminators.
- Statistically significant but **small-effect** differences were found for **Chloramines, Solids (TDS), and Organic Carbon**.
- No single parameter reliably predicts potability on its own.
- A **multivariate ML approach** (Random Forest, XGBoost) leveraging all nine features simultaneously is the recommended next step.

**Limitations:**
- High missing-value proportion in Sulfate (~24%) and pH (~15%) may introduce imputation bias.
- Dataset source and geographic context are undocumented — generalisability is limited.
- The class imbalance should be addressed (SMOTE, class-weight tuning) before modelling.
    """)
