"""Credit default risk scoring: interactive app.

Run with:  streamlit run app/streamlit_app.py
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import BLUE, ORANGE, bar, lines, load, predict, show, style, vector

st.set_page_config(page_title="Credit default risk", layout="wide")
model, s = load("model.json"), load("summary.json")
base = s["base_rate"]
chosen = next(m for m in s["metrics"] if "chosen" in m["model"])

STATUS = {"No balance to pay": -2, "Paid in full": -1, "Paid the minimum": 0, "1 month behind": 1, "2 months behind": 2, "3 months behind": 3,
          "4 months behind": 4, "5 months behind": 5, "6 months behind": 6, "7 months behind": 7, "8 or more months behind": 8}
NAMES = {"status_1": "Latest repayment status", "status_2": "Status 2 months ago", "status_3": "Status 3 months ago",
         "max_delay": "Worst delay in six months", "months_late": "Months with a late payment", "limit": "Credit limit",
         "util_mean": "Average utilisation", "util_latest": "Latest utilisation", "last_payment": "Last payment amount",
         "pay_ratio": "Share of bills paid", "bill_trend": "Change in balance", "last_pay_ratio": "Last payment as share of bill"}


def clip(value, low, high):
    return max(low, min(high, value))


def to_score(p):
    p = clip(p, 1e-4, 1 - 1e-4)
    return 600 + 50 * math.log2((1 - p) / p / 10)


st.title("Who should a lender approve?")
st.caption("A probability of default model and credit score built from six months of real repayment behaviour. "
           "Research demonstration only. Not a lending tool.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers analysed", f"{s['customers']:,}")
c2.metric("Default rate", f"{base:.1%}")
c3.metric("Model AUROC", f"{chosen['AUROC']:.3f}", f"Gini {chosen['Gini']:.2f}, KS {chosen['KS']:.2f}", delta_color="off")
c4.metric("Demographic inputs", "0")

loss = st.sidebar.slider("A default costs as much as this many good accounts", 1, 15, 5)
cutoff = 1 / (1 + loss)
st.sidebar.write(f"Break-even risk: **{cutoff:.1%}**. Customers below it are approved.")

tab_score, tab_cut, tab_explore, tab_model, tab_fair = st.tabs(["Score a customer", "Approval cutoff", "What separates defaulters", "Model performance", "Fairness audit"])

with tab_score:
    st.subheader("Six months of account history")
    st.caption("Amounts are in New Taiwan dollars, as in the source data.")
    a, b, c = st.columns(3)
    limit = a.number_input("Credit limit", 10000, 1000000, 140000, 10000)
    bill1 = b.number_input("Latest statement balance", 0, 1000000, 45000, 1000)
    bill2 = c.number_input("Previous statement balance", 0, 1000000, 44000, 1000)
    bill6 = a.number_input("Balance six months ago", 0, 1000000, 40000, 1000)
    bill_avg = b.number_input("Average balance, six months", 0, 1000000, 42000, 1000)
    pay1 = c.number_input("Last payment made", 0, 1000000, 2000, 500)
    pay_total = a.number_input("Total paid, six months", 0, 3000000, 12000, 1000)
    st.markdown("**Repayment status by month** (month 1 is the most recent)")
    status_columns = st.columns(6)
    statuses = [STATUS[col.selectbox(f"Month {i + 1}", list(STATUS), index=2, key=f"status{i}")] for i, col in enumerate(status_columns)]

    values = {
        "limit": limit, "max_delay": max(statuses), "months_late": sum(v >= 1 for v in statuses),
        "util_latest": clip(bill1 / limit, -0.5, 2), "util_mean": clip(bill_avg / limit, -0.5, 2),
        "bill_trend": clip((bill1 - bill6) / limit, -2, 2), "pay_ratio": clip(pay_total / max(bill_avg * 6, 1), 0, 2),
        "last_pay_ratio": clip(pay1 / max(bill2, 1), 0, 2), "last_payment": pay1,
    }
    values.update({f"status_{i + 1}": v for i, v in enumerate(statuses)})
    pd_value = predict(model, vector(model, values))
    st.divider()
    left, middle, right = st.columns(3)
    left.metric("Probability of default next month", f"{pd_value:.1%}")
    middle.metric("Credit score", f"{to_score(pd_value):.0f}")
    right.metric("Decision", "Approve" if pd_value < cutoff else "Decline")
    st.write(f"This customer is at {pd_value / base:.1f} times the portfolio average of {base:.1%}. "
             f"Utilisation {values['util_latest']:.0%}, {values['months_late']} late month(s) in six.")

with tab_cut:
    st.caption("A good account earns one unit. A default costs the number of units set in the sidebar. "
               "The curve is total profit on held-out test customers if the lender approves the lowest-risk share shown.")
    p_sorted, y_sorted = np.array(s["test_sorted"]["p"]), np.array(s["test_sorted"]["y"])
    profit = np.cumsum(np.where(y_sorted == 1, -loss, 1))
    approved = np.arange(1, len(y_sorted) + 1)
    at_cut = int(np.searchsorted(p_sorted, cutoff))
    figure = lines([("Total profit", approved / len(approved) * 100, profit, BLUE, "solid")], "Share of customers approved, lowest risk first (%)", "Total profit (units)")
    if at_cut > 0:
        figure.add_scatter(x=[at_cut / len(approved) * 100], y=[profit[at_cut - 1]], mode="markers", name="Break-even cutoff", marker=dict(color=ORANGE, size=11))
    show(figure)
    if at_cut > 0:
        st.write(f"Approving customers below {cutoff:.1%} risk approves {at_cut / len(approved):.0%} of the {len(approved):,} test customers, "
                 f"with a default rate of {y_sorted[:at_cut].mean():.1%} among them, for a profit of {profit[at_cut - 1]:,} units. "
                 f"Approving everyone gives {profit[-1]:,} units. The best cutoff in hindsight gives {profit.max():,}.")

with tab_explore:
    left, right = st.columns(2)
    with left:
        st.markdown("**Default rate by latest repayment status**")
        rows = s["rate_by_status"]
        show(bar([str(r["PAY_1"]) for r in rows], [r["rate"] * 100 for r in rows], "Months behind on latest statement", "Default rate (%)", reference=base * 100))
    with right:
        st.markdown("**Default rate by credit limit**")
        rows = s["rate_by_limit"]
        show(bar([r["band"] for r in rows], [r["rate"] * 100 for r in rows], "Credit limit (NT$)", "Default rate (%)", reference=base * 100))

with tab_model:
    left, right = st.columns(2)
    cal = s["calibration"]
    with left:
        st.markdown("**Predicted against observed default**")
        show(lines([("Model", [v * 100 for v in cal["predicted"]], [v * 100 for v in cal["observed"]], BLUE, "solid")],
                   "Predicted probability of default (%)", "Observed default rate (%)", diagonal=max(cal["predicted"] + cal["observed"]) * 108))
    with right:
        st.markdown("**Credit score distribution**")
        hist = s["score_hist"]
        figure = go.Figure([go.Bar(x=hist["bins"], y=hist["repaid"], name="Repaid", marker_color=BLUE),
                            go.Bar(x=hist["bins"], y=hist["defaulted"], name="Defaulted", marker_color=ORANGE)])
        figure.update_xaxes(title="Credit score")
        figure.update_yaxes(title="Test customers")
        show(style(figure, barmode="stack"))
    left, right = st.columns(2)
    with left:
        st.markdown("**What drives the predictions**")
        importance = s["importance"][::-1]
        show(bar([NAMES.get(r["feature"], r["feature"]) for r in importance], [r["value"] for r in importance], "Mean absolute SHAP value", "", horizontal=True))
    with right:
        st.markdown("**Model comparison**")
        st.dataframe(pd.DataFrame(s["metrics"]).round(4), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame(s["score_bands"]).round(3), hide_index=True, width="stretch")

with tab_fair:
    st.caption("The model was not given sex, age, education or marital status. Outcomes are checked anyway, at a cutoff where a default costs five good accounts. "
               "The four-fifths rule asks that every group's approval rate is at least 80 percent of the highest group's.")
    fairness = pd.DataFrame(s["fairness"])[["attribute", "group", "customers", "approval_rate", "approval_ratio_to_highest", "default_rate_approved", "observed_default", "predicted_default"]]
    percent = {c: "{:.1%}" for c in ["approval_rate", "default_rate_approved", "observed_default", "predicted_default"]}
    st.dataframe(fairness.style.format({**percent, "approval_ratio_to_highest": "{:.2f}", "customers": "{:,}"}), hide_index=True, width="stretch")
