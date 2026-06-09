
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import joblib
import pickle

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="H1B Employer Risk Intelligence",
    page_icon="🇺🇸",
    layout="wide"
)

# ── Load model and data ───────────────────────────────────────


@st.cache_resource
def load_model():
    import mlflow
    model = mlflow.sklearn.load_model("mlruns/models/h1b_risk_model")
    return model


@st.cache_data
def load_data():
    df = pd.read_parquet("data/df_model.parquet")
    return df


# ── Header ────────────────────────────────────────────────────
st.title("🇺🇸 H1B Employer Risk Intelligence Platform")
st.markdown("""
Analyze H1B visa sponsorship risk for US employers.  
Built on **187,000+ employer records** from DOL LCA filings and USCIS petition data (2017–2026).
""")

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Employer Lookup",
    "📊 Market Insights",
    "🤖 Model Info"
])

# ══════════════════════════════════════════════════════════════
# TAB 1: EMPLOYER LOOKUP
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Search Employer Risk Profile")

    try:
        df = load_data()

        employer_input = st.text_input(
            "Enter employer name",
            placeholder="e.g. Infosys, Google, Tata Consultancy"
        )

        if employer_input:
            # Search for matching employers
            mask = df["EMPLOYER_NAME"].str.contains(
                employer_input, case=False, na=False
            )
            matches = df[mask]["EMPLOYER_NAME"].unique()

            if len(matches) == 0:
                st.warning("No employer found. Try a different name.")

            else:
                selected = st.selectbox("Select employer:", matches)
                emp_data = df[df["EMPLOYER_NAME"] == selected]

                # Show latest year data
                latest = emp_data.sort_values(
                    "FISCAL_YEAR", ascending=False
                ).iloc[0]

                # Risk metrics
                col1, col2, col3, col4 = st.columns(4)

                denial_rate = latest["DENIAL_RATE"]
                risk_flag = denial_rate >= 0.10

                col1.metric(
                    "Denial Rate",
                    f"{denial_rate:.1%}",
                    delta=None
                )
                col2.metric(
                    "Risk Level",
                    "🔴 High Risk" if risk_flag else "🟢 Low Risk"
                )
                col3.metric(
                    "Total Petitions",
                    f"{int(latest['TOTAL_PETITIONS']):,}"
                )
                col4.metric(
                    "Avg Wage",
                    f"${int(latest['LCA_AVG_WAGE']):,}"
                )

                # Wage vs prevailing
                st.markdown("---")
                col5, col6 = st.columns(2)

                with col5:
                    st.markdown("**Wage Compliance**")
                    wage_ratio = latest["LCA_AVG_WAGE_RATIO"]
                    if wage_ratio >= 1.10:
                        st.success(
                            f"Pays {wage_ratio:.1%} of prevailing wage ✓")
                    elif wage_ratio >= 1.0:
                        st.info(f"Pays {wage_ratio:.1%} of prevailing wage")
                    else:
                        st.error(
                            f"Pays only {wage_ratio:.1%} of prevailing wage ⚠️")

                with col6:
                    st.markdown("**Filing History**")
                    st.write(
                        f"LCA Certification Rate: {latest['LCA_CERT_RATE']:.1%}")
                    st.write(
                        f"Full Time Positions: {latest['LCA_FULL_TIME_PCT']:.1%}")
                    st.write(
                        f"Unique Job Titles Filed: {int(latest['LCA_UNIQUE_JOB_TITLES'])}")

                # Year over year trend for this employer
                if len(emp_data) > 1:
                    st.markdown("---")
                    st.markdown("**Denial Rate Trend**")
                    trend = emp_data.sort_values("FISCAL_YEAR")

                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(trend["FISCAL_YEAR"],
                            trend["DENIAL_RATE"],
                            marker="o", color="crimson")
                    ax.yaxis.set_major_formatter(
                        mtick.PercentFormatter(xmax=1)
                    )
                    ax.set_xlabel("Fiscal Year")
                    ax.set_ylabel("Denial Rate")
                    ax.set_title(f"Denial Rate Trend — {selected}")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

    except Exception as e:
        st.error(f"Data not loaded: {e}")
        st.info("Make sure df_model.parquet is in the data/ folder")

# ══════════════════════════════════════════════════════════════
# TAB 2: MARKET INSIGHTS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("H1B Market Intelligence")

    try:
        df = load_data()

        col1, col2 = st.columns(2)

        # Chart 1: Industry risk
        with col1:
            st.markdown("**Denial Rate by Industry**")
            industry_risk = (
                df[df["NAICS_CODE"] != "UNKNOWN"]
                .groupby("NAICS_CODE")
                .agg(
                    avg_denial_rate=("DENIAL_RATE", "mean"),
                    total_employers=("EMPLOYER_NAME", "count")
                )
                .reset_index()
                .sort_values("total_employers", ascending=False)
                .head(10)
                .sort_values("avg_denial_rate", ascending=True)
            )

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(industry_risk["NAICS_CODE"],
                    industry_risk["avg_denial_rate"],
                    color="steelblue")
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
            ax.set_title("By Industry")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Chart 2: Yearly trend
        with col2:
            st.markdown("**Denial Rate by Year**")
            yearly = (df
                      .groupby("FISCAL_YEAR")
                      .agg(avg_denial_rate=("DENIAL_RATE", "mean"))
                      .reset_index()
                      .sort_values("FISCAL_YEAR")
                      )

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(yearly["FISCAL_YEAR"],
                    yearly["avg_denial_rate"],
                    marker="o", color="crimson", linewidth=2)
            ax.axvspan(2017, 2021, alpha=0.1, color="red",
                       label="High Scrutiny Era")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
            ax.set_xticks(yearly["FISCAL_YEAR"].tolist())
            ax.set_xticklabels(
                yearly["FISCAL_YEAR"].tolist(), rotation=45
            )
            ax.legend(fontsize=8)
            ax.set_title("By Fiscal Year")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Chart 3: Wage risk
        st.markdown("**Denial Rate by Wage Compliance**")
        df["wage_ratio_bucket"] = pd.cut(
            df["LCA_AVG_WAGE_RATIO"],
            bins=[0, 0.9, 1.0, 1.1, 1.25, 1.5, 99],
            labels=["<90%", "90-100%", "100-110%",
                    "110-125%", "125-150%", ">150%"]
        )
        wage_risk = (df
                     .groupby("wage_ratio_bucket", observed=True)
                     .agg(avg_denial_rate=("DENIAL_RATE", "mean"))
                     .reset_index()
                     )

        fig, ax = plt.subplots(figsize=(10, 3))
        bars = ax.bar(wage_risk["wage_ratio_bucket"],
                      wage_risk["avg_denial_rate"],
                      color="steelblue")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.set_xlabel("Wage vs Prevailing Wage")
        ax.set_ylabel("Avg Denial Rate")
        ax.set_title("Wage Compliance vs Denial Risk")
        for bar, val in zip(bars, wage_risk["avg_denial_rate"]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + 0.001,
                    f"{val:.1%}", ha="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"Error loading charts: {e}")

# ══════════════════════════════════════════════════════════════
# TAB 3: MODEL INFO
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model Performance**")
        st.metric("ROC-AUC", "0.9351")
        st.metric("PR-AUC", "0.7298")
        st.metric("High Risk Recall", "88%")
        st.metric("High Risk Precision", "47%")

    with col2:
        st.markdown("**Dataset**")
        st.metric("Total Records", "187,454")
        st.metric("Training Size", "149,963")
        st.metric("Features Used", "19")
        st.metric("Risk Threshold", "10% denial rate")

    st.markdown("---")
    st.markdown("**Key Findings**")
    st.markdown("""
    - 📌 Professional & Technical Services has the highest denial rate at **9.1%**
    - 📌 2018 peak denial rate of **13.7%** — driven by USCIS policy changes
    - 📌 Employers paying below 90% of prevailing wage face **24.5% denial rate**
    - 📌 Paying above 150% of prevailing wage reduces denial risk to **3.5%**
    - 📌 Model catches **88% of high-risk employers** (recall)
    """)

    st.markdown("**Data Sources**")
    st.markdown("""
    - US Department of Labor — Labor Condition Application (LCA) data
    - USCIS — H1B petition approval/denial data
    - Years covered: 2017–2026
    """)
