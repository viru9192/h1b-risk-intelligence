
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="H1B Risk Intelligence",
    page_icon="🇺🇸",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 16px;
    border-left: 4px solid #1f77b4;
    margin: 8px 0;
}
.risk-high {
    background: #fff5f5;
    border-left: 4px solid #e53e3e;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
}
.risk-low {
    background: #f0fff4;
    border-left: 4px solid #38a169;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Load Data ──────────────────────────────────────────────────


@st.cache_data
def load_data():
    df = pd.read_parquet("data/df_model.parquet")
    df["FISCAL_YEAR"] = pd.to_numeric(df["FISCAL_YEAR"],
                                      errors="coerce")
    df = df.dropna(subset=["FISCAL_YEAR"])
    df["FISCAL_YEAR"] = df["FISCAL_YEAR"].astype(int)
    return df


df = load_data()

# ── Sidebar Filters ────────────────────────────────────────────
st.sidebar.title("🔧 Global Filters")
st.sidebar.markdown("Filters apply to all tabs")

all_years = sorted(df["FISCAL_YEAR"].unique())
selected_years = st.sidebar.multiselect(
    "Fiscal Year",
    options=all_years,
    default=all_years
)

all_states = sorted(df["EMPLOYER_STATE"].dropna().unique())
selected_states = st.sidebar.multiselect(
    "Employer State",
    options=all_states,
    default=[]
)

all_industries = sorted(df["NAICS_CODE"].dropna().unique())
selected_industries = st.sidebar.multiselect(
    "Industry (NAICS)",
    options=all_industries,
    default=[]
)

# Apply filters
df_filtered = df[df["FISCAL_YEAR"].isin(selected_years)]
if selected_states:
    df_filtered = df_filtered[
        df_filtered["EMPLOYER_STATE"].isin(selected_states)
    ]
if selected_industries:
    df_filtered = df_filtered[
        df_filtered["NAICS_CODE"].isin(selected_industries)
    ]

# ── Header ─────────────────────────────────────────────────────
st.title("🇺🇸 H1B Employer Risk Intelligence Platform")
st.markdown(
    f"Analyzing **{len(df_filtered):,} employer records** "
    f"from DOL LCA filings and USCIS petition data "
    f"({min(selected_years) if selected_years else ''} – "
    f"{max(selected_years) if selected_years else ''})"
)

# ── KPI Row ────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Employers",
          f"{df_filtered['EMPLOYER_NAME'].nunique():,}")
k2.metric("Avg Denial Rate",
          f"{df_filtered['DENIAL_RATE'].mean():.1%}")
k3.metric("High Risk Employers",
          f"{(df_filtered['is_high_risk'] == 1).sum():,}")
k4.metric("Avg H1B Wage",
          f"${df_filtered['LCA_AVG_WAGE'].mean():,.0f}")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Employer Lookup",
    "⚖️ Employer Comparison",
    "📊 Market Intelligence",
    "🏆 Top Safe Employers",
    "💰 Salary Benchmarking",
    "🤖 Model Info"
])

# ══════════════════════════════════════════════════════════════
# TAB 1: EMPLOYER LOOKUP
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Employer Risk Profile")
    st.markdown("Search any employer to see their H1B risk profile, "
                "wage compliance, and filing history.")

    employer_input = st.text_input(
        "Enter employer name",
        placeholder="e.g. Infosys, Google, Tata Consultancy"
    )

    if employer_input:
        mask = df["EMPLOYER_NAME"].str.contains(
            employer_input, case=False, na=False
        )
        matches = df[mask]["EMPLOYER_NAME"].unique()

        if len(matches) == 0:
            st.warning("No employer found. Try a partial name.")
        else:
            selected = st.selectbox("Select employer:", matches)
            emp_data = df[df["EMPLOYER_NAME"] == selected]

            # ── All metrics aggregated across all years ────────
            denial_rate = emp_data["DENIAL_RATE"].mean()
            total_petitions = int(emp_data["TOTAL_PETITIONS"].sum())
            avg_wage = int(emp_data["LCA_AVG_WAGE"].mean())
            avg_wage_ratio = emp_data["LCA_AVG_WAGE_RATIO"].mean()
            avg_cert_rate = emp_data["LCA_CERT_RATE"].mean()
            risk_flag = denial_rate >= 0.10

            # ── Risk banner ────────────────────────────────────
            if risk_flag:
                st.markdown(
                    f"<div class='risk-high'>"
                    f"<b>🔴 HIGH RISK EMPLOYER</b> "
                    f"Avg denial rate {denial_rate:.1%} exceeds "
                    f"10% threshold across all years"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='risk-low'>"
                    f"<b>🟢 LOW RISK EMPLOYER</b> "
                    f"Avg denial rate {denial_rate:.1%} is within "
                    f"safe range across all years"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # ── Metrics row ────────────────────────────────────
            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric(
                "Avg Denial Rate",
                f"{denial_rate:.1%}"
            )
            c2.metric(
                "Total Petitions (All Years)",
                f"{total_petitions:,}"
            )
            c3.metric(
                "Avg Wage",
                f"${avg_wage:,}"
            )
            c4.metric(
                "Avg Wage vs Market",
                f"{avg_wage_ratio:.1%}"
            )
            c5.metric(
                "Avg LCA Cert Rate",
                f"{avg_cert_rate:.1%}"
            )

            # ── Wage compliance signal ─────────────────────────
            st.markdown("**Wage Compliance**")
            if avg_wage_ratio >= 1.10:
                st.success(
                    f"Pays {avg_wage_ratio:.1%} of prevailing wage "
                    f"on average. Strong compliance signal."
                )
            elif avg_wage_ratio >= 1.0:
                st.info(
                    f"Pays {avg_wage_ratio:.1%} of prevailing wage "
                    f"on average. At market rate."
                )
            else:
                st.error(
                    f"Pays only {avg_wage_ratio:.1%} of prevailing "
                    f"wage on average. Below market rate. "
                    f"Higher denial risk."
                )

            # ── Year over year trend ───────────────────────────
            if len(emp_data) > 1:
                st.markdown("**Year-over-Year Denial Rate Trend**")
                trend = emp_data.sort_values("FISCAL_YEAR")

                col_chart, col_table = st.columns([2, 1])

                with col_chart:
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(
                        trend["FISCAL_YEAR"],
                        trend["DENIAL_RATE"],
                        marker="o", color="crimson",
                        linewidth=2.5
                    )
                    ax.fill_between(
                        trend["FISCAL_YEAR"],
                        trend["DENIAL_RATE"],
                        alpha=0.1, color="crimson"
                    )
                    ax.axhline(
                        y=0.10, color="orange",
                        linestyle="--", linewidth=1.5,
                        label="Risk threshold (10%)"
                    )
                    ax.yaxis.set_major_formatter(
                        mtick.PercentFormatter(xmax=1)
                    )
                    ax.set_xlabel("Fiscal Year")
                    ax.set_ylabel("Denial Rate")
                    ax.set_title(
                        f"Denial Rate Trend — {selected[:40]}"
                    )
                    ax.set_xticks(trend["FISCAL_YEAR"].tolist())
                    ax.set_xticklabels(
                        trend["FISCAL_YEAR"].tolist(),
                        rotation=45
                    )
                    ax.legend(fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                with col_table:
                    st.markdown("**Filing History**")
                    display_cols = [
                        "FISCAL_YEAR",
                        "TOTAL_PETITIONS",
                        "DENIAL_RATE",
                        "LCA_AVG_WAGE"
                    ]
                    table = trend[display_cols].copy()
                    table["DENIAL_RATE"] = (
                        table["DENIAL_RATE"]
                        .apply(lambda x: f"{x:.1%}")
                    )
                    table["LCA_AVG_WAGE"] = (
                        table["LCA_AVG_WAGE"]
                        .apply(lambda x: f"${x:,.0f}")
                    )
                    table.columns = [
                        "Year", "Petitions",
                        "Denial Rate", "Avg Wage"
                    ]
                    st.dataframe(
                        table,
                        hide_index=True,
                        use_container_width=True
                    )

                # ── Summary stats below chart ──────────────────
                st.markdown("**Summary Across All Years**")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric(
                    "Years Active",
                    f"{emp_data['FISCAL_YEAR'].nunique()}"
                )
                s2.metric(
                    "Best Year Denial Rate",
                    f"{emp_data['DENIAL_RATE'].min():.1%}"
                )
                s3.metric(
                    "Worst Year Denial Rate",
                    f"{emp_data['DENIAL_RATE'].max():.1%}"
                )
                s4.metric(
                    "Avg Petitions Per Year",
                    f"{int(emp_data['TOTAL_PETITIONS'].mean()):,}"
                )

# ══════════════════════════════════════════════════════════════
# TAB 2: EMPLOYER COMPARISON
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Compare Employers Side by Side")
    st.markdown("Compare up to 3 employers across key H1B metrics.")

    col1, col2, col3 = st.columns(3)

    with col1:
        emp1 = st.text_input("Employer 1",
                             placeholder="e.g. Infosys")
    with col2:
        emp2 = st.text_input("Employer 2",
                             placeholder="e.g. Google")
    with col3:
        emp3 = st.text_input("Employer 3 (optional)",
                             placeholder="e.g. Tata")

    employers_to_compare = [e for e in [emp1, emp2, emp3] if e]

    if len(employers_to_compare) >= 2:
        comparison_data = []

        for emp_name in employers_to_compare:
            mask = df["EMPLOYER_NAME"].str.contains(
                emp_name, case=False, na=False
            )
            matches = df[mask]

            if len(matches) > 0:
                # Aggregate across all years
                agg = matches.groupby("EMPLOYER_NAME").agg(
                    avg_denial_rate=("DENIAL_RATE", "mean"),
                    avg_wage=("LCA_AVG_WAGE", "mean"),
                    avg_wage_ratio=("LCA_AVG_WAGE_RATIO", "mean"),
                    total_petitions=("TOTAL_PETITIONS", "sum"),
                    lca_cert_rate=("LCA_CERT_RATE", "mean"),
                    years_active=("FISCAL_YEAR", "nunique")
                ).reset_index()

                # Take top match by petition volume
                top_match = agg.sort_values(
                    "total_petitions", ascending=False
                ).iloc[0]
                comparison_data.append(top_match)

        if len(comparison_data) >= 2:
            comp_df = pd.DataFrame(comparison_data)

            # Comparison metrics table
            st.markdown("---")
            st.markdown("**Head-to-Head Comparison**")

            display = comp_df[[
                "EMPLOYER_NAME", "avg_denial_rate",
                "avg_wage", "avg_wage_ratio",
                "total_petitions", "lca_cert_rate"
            ]].copy()

            display.columns = [
                "Employer", "Avg Denial Rate",
                "Avg Wage ($)", "Wage vs Market",
                "Total Petitions", "LCA Cert Rate"
            ]
            display["Avg Denial Rate"] = display[
                "Avg Denial Rate"
            ].apply(lambda x: f"{x:.1%}")
            display["Avg Wage ($)"] = display[
                "Avg Wage ($)"
            ].apply(lambda x: f"${x:,.0f}")
            display["Wage vs Market"] = display[
                "Wage vs Market"
            ].apply(lambda x: f"{x:.1%}")
            display["LCA Cert Rate"] = display[
                "LCA Cert Rate"
            ].apply(lambda x: f"{x:.1%}")

            st.dataframe(display, hide_index=True,
                         use_container_width=True)

            # Visual comparison
            st.markdown("---")
            st.markdown("**Visual Comparison**")

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            names = [n[:20] for n in comp_df["EMPLOYER_NAME"]]
            colors = ["#e53e3e", "#3182ce", "#38a169"][:len(comp_df)]

            # Chart A: Denial Rate
            axes[0].bar(names, comp_df["avg_denial_rate"],
                        color=colors)
            axes[0].yaxis.set_major_formatter(
                mtick.PercentFormatter(xmax=1)
            )
            axes[0].axhline(y=0.10, color="orange",
                            linestyle="--", linewidth=1.5,
                            label="Risk threshold")
            axes[0].set_title("Avg Denial Rate")
            axes[0].legend(fontsize=7)
            axes[0].tick_params(axis="x", rotation=15)

            # Chart B: Average Wage
            axes[1].bar(names, comp_df["avg_wage"],
                        color=colors)
            axes[1].yaxis.set_major_formatter(
                mtick.FuncFormatter(
                    lambda x, p: f"${x:,.0f}"
                )
            )
            axes[1].set_title("Avg H1B Wage ($)")
            axes[1].tick_params(axis="x", rotation=15)

            # Chart C: Wage vs Market
            axes[2].bar(names, comp_df["avg_wage_ratio"],
                        color=colors)
            axes[2].yaxis.set_major_formatter(
                mtick.PercentFormatter(xmax=1)
            )
            axes[2].axhline(y=1.0, color="gray",
                            linestyle="--", linewidth=1.5,
                            label="Prevailing wage")
            axes[2].set_title("Wage vs Prevailing Market")
            axes[2].legend(fontsize=7)
            axes[2].tick_params(axis="x", rotation=15)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Winner summary
            st.markdown("---")
            st.markdown("**Summary**")

            safest = comp_df.loc[
                comp_df["avg_denial_rate"].idxmin(),
                "EMPLOYER_NAME"
            ]
            highest_wage = comp_df.loc[
                comp_df["avg_wage"].idxmax(),
                "EMPLOYER_NAME"
            ]

            s1, s2 = st.columns(2)
            s1.success(f"✅ Safest Sponsor: **{safest[:30]}**")
            s2.info(f"💰 Highest Payer: **{highest_wage[:30]}**")

        else:
            st.warning(
                "Could not find enough matching employers. "
                "Try different names."
            )

# ══════════════════════════════════════════════════════════════
# TAB 3: MARKET INTELLIGENCE
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("H1B Market Intelligence")
    st.markdown("Industry trends, policy impact, and "
                "wage compliance analysis.")

    # Chart 1 + 2 side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Denial Rate by Industry**")
        industry_data = (
            df_filtered[df_filtered["NAICS_CODE"] != "UNKNOWN"]
            .groupby("NAICS_CODE")
            .agg(
                avg_denial_rate=("DENIAL_RATE", "mean"),
                total=("EMPLOYER_NAME", "count")
            )
            .reset_index()
            .sort_values("total", ascending=False)
            .head(10)
            .sort_values("avg_denial_rate", ascending=True)
        )

        fig, ax = plt.subplots(figsize=(7, 5))
        colors_bar = [
            "#e53e3e" if x > 0.08 else
            "#ed8936" if x > 0.05 else
            "#38a169"
            for x in industry_data["avg_denial_rate"]
        ]
        bars = ax.barh(industry_data["NAICS_CODE"],
                       industry_data["avg_denial_rate"],
                       color=colors_bar)
        ax.xaxis.set_major_formatter(
            mtick.PercentFormatter(xmax=1)
        )
        for bar, val in zip(
            bars, industry_data["avg_denial_rate"]
        ):
            ax.text(val + 0.001,
                    bar.get_y() + bar.get_height()/2,
                    f"{val:.1%}", va="center", fontsize=8)
        ax.set_title("Top 10 Industries by Filing Volume")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Denial Rate by Fiscal Year**")
        yearly = (df_filtered
                  .groupby("FISCAL_YEAR")
                  .agg(avg_denial_rate=("DENIAL_RATE", "mean"))
                  .reset_index()
                  .sort_values("FISCAL_YEAR")
                  )

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(yearly["FISCAL_YEAR"],
                yearly["avg_denial_rate"],
                marker="o", color="crimson", linewidth=2.5)
        ax.fill_between(yearly["FISCAL_YEAR"],
                        yearly["avg_denial_rate"],
                        alpha=0.1, color="crimson")
        ax.axvspan(2017, 2021, alpha=0.1, color="red",
                   label="High Scrutiny Era (2017-2021)")
        ax.yaxis.set_major_formatter(
            mtick.PercentFormatter(xmax=1)
        )
        ax.set_xticks(yearly["FISCAL_YEAR"].tolist())
        ax.set_xticklabels(
            yearly["FISCAL_YEAR"].tolist(), rotation=45
        )
        ax.legend(fontsize=8)
        ax.set_title("Denial Rate Over Time")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Chart 3: Wage compliance
    st.markdown("**Denial Rate by Wage Compliance Level**")
    df_filtered["wage_bucket"] = pd.cut(
        df_filtered["LCA_AVG_WAGE_RATIO"],
        bins=[0, 0.9, 1.0, 1.1, 1.25, 1.5, 99],
        labels=["<90%", "90-100%", "100-110%",
                "110-125%", "125-150%", ">150%"]
    )
    wage_data = (df_filtered
                 .groupby("wage_bucket", observed=True)
                 .agg(
                     avg_denial=("DENIAL_RATE", "mean"),
                     count=("EMPLOYER_NAME", "count")
                 )
                 .reset_index()
                 )

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    bars = ax1.bar(wage_data["wage_bucket"],
                   wage_data["avg_denial"],
                   color="steelblue", alpha=0.8,
                   label="Denial Rate")
    ax2.plot(wage_data["wage_bucket"],
             wage_data["count"],
             color="orange", marker="o",
             linewidth=2, label="Employer Count")

    ax1.yaxis.set_major_formatter(
        mtick.PercentFormatter(xmax=1)
    )
    ax1.set_ylabel("Average Denial Rate", color="steelblue")
    ax2.set_ylabel("Number of Employers", color="orange")
    ax1.set_xlabel("Wage Offered vs Prevailing Wage")
    ax1.set_title(
        "Wage Compliance vs Denial Risk "
        "(bar=denial rate, line=employer count)"
    )

    for bar, val in zip(bars, wage_data["avg_denial"]):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 val + 0.002,
                 f"{val:.1%}", ha="center", fontsize=9,
                 fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=9, loc="upper right")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # State level analysis
    st.markdown("---")
    st.markdown("**Denial Rate by State (Top 15)**")
    state_data = (
        df_filtered[df_filtered["EMPLOYER_STATE"].notna()]
        .groupby("EMPLOYER_STATE")
        .agg(
            avg_denial=("DENIAL_RATE", "mean"),
            total=("EMPLOYER_NAME", "count")
        )
        .reset_index()
        .sort_values("total", ascending=False)
        .head(15)
        .sort_values("avg_denial", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_state = [
        "#e53e3e" if x > 0.08 else
        "#ed8936" if x > 0.05 else
        "#38a169"
        for x in state_data["avg_denial"]
    ]
    bars = ax.barh(state_data["EMPLOYER_STATE"],
                   state_data["avg_denial"],
                   color=colors_state)
    ax.xaxis.set_major_formatter(
        mtick.PercentFormatter(xmax=1)
    )
    for bar, val in zip(bars, state_data["avg_denial"]):
        ax.text(val + 0.001,
                bar.get_y() + bar.get_height()/2,
                f"{val:.1%}", va="center", fontsize=8)
    ax.set_title(
        "Top 15 States by Filing Volume — "
        "Color: Red>8%, Orange>5%, Green≤5%"
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 4: TOP SAFE EMPLOYERS
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Top Safe H1B Employers")
    st.markdown(
        "Employers with lowest denial rates among "
        "high-volume sponsors."
    )

    # Filters
    f1, f2, f3 = st.columns(3)

    with f1:
        min_petitions = st.slider(
            "Min total petitions",
            min_value=5,
            max_value=500,
            value=50,
            step=5
        )
    with f2:
        industry_filter = st.selectbox(
            "Filter by Industry",
            options=["All Industries"] + sorted(
                df_filtered["NAICS_CODE"]
                .dropna().unique().tolist()
            )
        )
    with f3:
        state_filter = st.selectbox(
            "Filter by State",
            options=["All States"] + sorted(
                df_filtered["EMPLOYER_STATE"]
                .dropna().unique().tolist()
            )
        )

    # Build safe employer list
    safe_df = (df_filtered
               .groupby("EMPLOYER_NAME")
               .agg(
                   avg_denial_rate=("DENIAL_RATE", "mean"),
                   avg_wage=("LCA_AVG_WAGE", "mean"),
                   avg_wage_ratio=("LCA_AVG_WAGE_RATIO", "mean"),
                   total_petitions=("TOTAL_PETITIONS", "sum"),
                   lca_cert_rate=("LCA_CERT_RATE", "mean"),
                   industry=("NAICS_CODE", "first"),
                   state=("EMPLOYER_STATE", "first"),
                   years=("FISCAL_YEAR", "nunique")
               )
               .reset_index()
               )

    # Apply filters
    safe_df = safe_df[
        safe_df["total_petitions"] >= min_petitions
    ]
    if industry_filter != "All Industries":
        safe_df = safe_df[
            safe_df["industry"] == industry_filter
        ]
    if state_filter != "All States":
        safe_df = safe_df[
            safe_df["state"] == state_filter
        ]

    safe_df = safe_df.sort_values(
        "avg_denial_rate", ascending=True
    ).head(25)

    # Display
    st.markdown(
        f"Showing top **{len(safe_df)}** safest employers "
        f"(min {min_petitions} petitions)"
    )

    display_safe = safe_df[[
        "EMPLOYER_NAME", "avg_denial_rate",
        "avg_wage", "avg_wage_ratio",
        "total_petitions", "lca_cert_rate",
        "industry", "state"
    ]].copy()

    display_safe.columns = [
        "Employer", "Denial Rate", "Avg Wage ($)",
        "Wage vs Market", "Total Petitions",
        "LCA Cert Rate", "Industry", "State"
    ]
    display_safe["Denial Rate"] = display_safe[
        "Denial Rate"
    ].apply(lambda x: f"{x:.1%}")
    display_safe["Avg Wage ($)"] = display_safe[
        "Avg Wage ($)"
    ].apply(lambda x: f"${x:,.0f}")
    display_safe["Wage vs Market"] = display_safe[
        "Wage vs Market"
    ].apply(lambda x: f"{x:.1%}")
    display_safe["LCA Cert Rate"] = display_safe[
        "LCA Cert Rate"
    ].apply(lambda x: f"{x:.1%}")
    display_safe["Total Petitions"] = display_safe[
        "Total Petitions"
    ].apply(lambda x: f"{int(x):,}")

    st.dataframe(display_safe, hide_index=True,
                 use_container_width=True, height=600)

    # Download button
    csv = safe_df.to_csv(index=False)
    st.download_button(
        label="📥 Download This List as CSV",
        data=csv,
        file_name="safe_h1b_employers.csv",
        mime="text/csv"
    )

# ══════════════════════════════════════════════════════════════
# TAB 5: SALARY BENCHMARKING
# ══════════════════════════════════════════════════════════════
with tab5:
    st.subheader("H1B Salary Benchmarking")
    st.markdown(
        "Analyze wage patterns across industries, "
        "states, and employer types."
    )

    # Wage distribution by industry
    st.markdown("**Average H1B Wage by Industry**")

    wage_industry = (
        df_filtered[df_filtered["NAICS_CODE"] != "UNKNOWN"]
        .groupby("NAICS_CODE")
        .agg(
            avg_wage=("LCA_AVG_WAGE", "mean"),
            avg_prevailing=("LCA_AVG_PREVAILING", "mean"),
            count=("EMPLOYER_NAME", "count")
        )
        .reset_index()
        .sort_values("count", ascending=False)
        .head(10)
        .sort_values("avg_wage", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    y = range(len(wage_industry))
    ax.barh(y, wage_industry["avg_wage"],
            color="steelblue", alpha=0.8,
            label="Avg H1B Wage")
    ax.barh(y, wage_industry["avg_prevailing"],
            color="orange", alpha=0.6,
            label="Avg Prevailing Wage")
    ax.set_yticks(list(y))
    ax.set_yticklabels(wage_industry["NAICS_CODE"].tolist())
    ax.xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, p: f"${x:,.0f}")
    )
    ax.set_title(
        "H1B Wage vs Prevailing Wage by Industry"
    )
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Wage by state
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Average H1B Wage by State (Top 15)**")
        wage_state = (
            df_filtered[df_filtered["EMPLOYER_STATE"].notna()]
            .groupby("EMPLOYER_STATE")
            .agg(
                avg_wage=("LCA_AVG_WAGE", "mean"),
                count=("EMPLOYER_NAME", "count")
            )
            .reset_index()
            .sort_values("count", ascending=False)
            .head(15)
            .sort_values("avg_wage", ascending=True)
        )

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.barh(wage_state["EMPLOYER_STATE"],
                wage_state["avg_wage"],
                color="steelblue")
        ax.xaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, p: f"${x:,.0f}")
        )
        ax.set_title("Avg H1B Wage by State")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Wage Premium Distribution**")
        df_filtered["wage_premium_k"] = (
            (df_filtered["LCA_AVG_WAGE"] -
             df_filtered["LCA_AVG_PREVAILING"]) / 1000
        )

        fig, ax = plt.subplots(figsize=(6, 6))
        premium_data = df_filtered[
            df_filtered["wage_premium_k"].between(-50, 100)
        ]["wage_premium_k"]

        ax.hist(premium_data, bins=40,
                color="steelblue", edgecolor="white",
                alpha=0.8)
        ax.axvline(x=0, color="red", linestyle="--",
                   linewidth=2, label="Prevailing wage line")
        ax.axvline(x=premium_data.median(),
                   color="green", linestyle="--",
                   linewidth=2,
                   label=f"Median: ${premium_data.median():.1f}K")
        ax.set_xlabel("Wage Premium Above Prevailing ($ thousands)")
        ax.set_ylabel("Number of Employers")
        ax.set_title("Distribution of Wage Premium")
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Wage vs denial scatter insight
    st.markdown("---")
    st.markdown("**Wage Compliance Summary Table**")

    wage_summary = (
        df_filtered
        .groupby("NAICS_CODE")
        .agg(
            avg_wage=("LCA_AVG_WAGE", "mean"),
            avg_prevailing=("LCA_AVG_PREVAILING", "mean"),
            avg_ratio=("LCA_AVG_WAGE_RATIO", "mean"),
            avg_denial=("DENIAL_RATE", "mean"),
            count=("EMPLOYER_NAME", "count")
        )
        .reset_index()
        .sort_values("avg_wage", ascending=False)
        .head(15)
    )

    wage_summary.columns = [
        "Industry", "Avg H1B Wage", "Avg Prevailing",
        "Wage Ratio", "Denial Rate", "Employers"
    ]
    wage_summary["Avg H1B Wage"] = wage_summary[
        "Avg H1B Wage"
    ].apply(lambda x: f"${x:,.0f}")
    wage_summary["Avg Prevailing"] = wage_summary[
        "Avg Prevailing"
    ].apply(lambda x: f"${x:,.0f}")
    wage_summary["Wage Ratio"] = wage_summary[
        "Wage Ratio"
    ].apply(lambda x: f"{x:.1%}")
    wage_summary["Denial Rate"] = wage_summary[
        "Denial Rate"
    ].apply(lambda x: f"{x:.1%}")

    st.dataframe(wage_summary, hide_index=True,
                 use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 6: MODEL INFO
# ══════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model Performance**")
        st.metric("ROC-AUC", "0.9351")
        st.metric("PR-AUC", "0.7298")
        st.metric("High Risk Recall", "88%")
        st.metric("High Risk Precision", "47%")
        st.metric("Training Size", "149,963 records")

    with col2:
        st.markdown("**What Each Metric Means**")
        st.markdown("""
        - **ROC-AUC 0.93** — Model correctly ranks 93% of
          employer pairs by risk level
        - **Recall 88%** — Catches 88 out of 100 genuinely
          risky employers
        - **Precision 47%** — Of flagged employers, 47% are
          truly high risk (conservative flagging intentional)
        - **Threshold 10%** — Employers with >10% historical
          denial rate classified as high risk
        """)

    st.markdown("---")
    st.markdown("**Key Findings From Analysis**")
    st.markdown("""
    | Finding | Detail |
    |---------|--------|
    | Riskiest Industry | Professional & Technical Services: 9.1% denial rate |
    | Policy Impact | 2018 peak: 13.7% denial rate — 6x higher than post-2021 |
    | Wage Effect | Below 90% prevailing wage → 24.5% denial rate |
    | Wage Safety | Above 150% prevailing wage → only 3.5% denial rate |
    | Safest Industry | Healthcare: 3.6% denial rate |
    """)

    st.markdown("**Data Sources**")
    st.markdown("""
    - US Department of Labor — LCA filings (2017–2026)
    - USCIS — H1B petition approval/denial data
    - 187,454 employer-year records after cleaning
    """)

    st.markdown("**Tech Stack**")
    st.markdown("""
    `Python` `XGBoost` `SHAP` `Streamlit`
    `scikit-learn` `MLflow` `pandas` `matplotlib`
    """)
