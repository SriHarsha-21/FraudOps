"""FraudOps review app. Run with:  streamlit run app/streamlit_app.py

Shows escalated cases, the evidence the agent gathered and its recommendation. A human
reviewer approves or blocks; that is the only way a human verdict enters the decision
record. Local demo only: the reviewer name is not an identity check, and the optional
PIN (environment variable FRAUDOPS_REVIEW_PIN) is a shared secret, not user accounts.
ASCII only.
"""
import json
import os
import sys

import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from fraudops.pipeline import fit_pipeline, load_frames  # noqa: E402
from fraudops.review import ReviewService, check_pin  # noqa: E402

st.set_page_config(page_title="FraudOps review", layout="wide")


@st.cache_resource(show_spinner="Training the detector and building the queue...")
def load_service(csv_path, max_cards, workdir):
    with open(os.path.join(HERE, "..", "config.json")) as fh:
        cfg = json.load(fh)
    train_df, test_df = load_frames(csv_path or None, None, int(max_cards), cfg["seed"])
    pipe = fit_pipeline(train_df, test_df, cfg)
    return ReviewService(pipe, cfg, workdir=workdir or None)


with st.sidebar:
    st.header("Session")
    reviewer = st.text_input("Reviewer name", key="reviewer")
    pin = st.text_input("PIN", type="password", key="pin") if os.environ.get(
        "FRAUDOPS_REVIEW_PIN") else ""
    st.header("Data")
    csv_path = st.text_input("Path to fraudTrain.csv (blank = synthetic demo data)",
                             value="", key="csv_path")
    max_cards = st.number_input("Cards to keep (0 = all)", min_value=0, value=0,
                                step=50, key="max_cards")
    workdir = st.text_input("State folder (keeps pending reviews across restarts)",
                            value="review_state", key="workdir")

st.title("FraudOps review queue")

if not check_pin(pin):
    st.warning("Enter the PIN in the sidebar to continue.")
    st.stop()

svc = load_service(csv_path.strip(), max_cards, workdir.strip())
if not csv_path.strip():
    st.info("Showing synthetic demo data. Results here say nothing about real fraud.")

tab_queue, tab_log, tab_summary = st.tabs(["Review queue", "Decision log", "Summary"])

with tab_queue:
    queue = svc.queue(limit=50)
    left, right = st.columns([3, 1])
    with right:
        st.metric("Waiting for review", len(svc.queue(limit=10 ** 9)))
        n_sim = st.number_input("Cases to simulate", min_value=1, value=10, step=5,
                                key="n_sim")
        if st.button("Simulate a reviewer (demo)", key="simulate"):
            done = svc.simulate_reviewer(int(n_sim))
            st.session_state["notice"] = "A simulated reviewer decided %d cases." % done
            st.rerun()
    with left:
        if "notice" in st.session_state:
            st.success(st.session_state.pop("notice"))
        if not queue:
            st.success("Nothing is waiting. Every escalated case has a decision.")
        else:
            tid = queue[0]
            opened = svc.open_case(tid)
            d = svc.details(tid)
            st.subheader("Case %s" % tid)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Amount", "%.2f" % d["amount"])
            c2.metric("Model fraud probability", "%.3f" % opened["payload"]["p_fraud"])
            c3.metric("Gate", opened["payload"]["gate_action"])
            c4.metric("Agent recommendation", opened["payload"]["recommendation"])
            st.write("Category: **%s** | Merchant: %s | Time: %s | Distance from "
                     "customer: %.0f km | Card ref: %s"
                     % (d["category"], d["merchant"], d["time"], d["distance_km"],
                        d["card_ref"]))
            if d["text_flag"]:
                st.warning("A text field looked like an instruction aimed at a model. "
                           "It was cleaned and never given to any model.")
            st.markdown("**Why this case needs a person**")
            for reason in opened["payload"]["reasons"] or ["No reason recorded."]:
                st.write("- " + reason)
            st.markdown("**Evidence the agent gathered**")
            ev = opened["payload"]["evidence"]
            if ev:
                st.dataframe(pd.DataFrame(ev), hide_index=True, use_container_width=True)
            else:
                st.write("No tools were run for this case.")
            b1, b2, _ = st.columns([1, 1, 4])
            ready = bool(reviewer.strip())
            if not ready:
                st.caption("Enter a reviewer name in the sidebar to enable the buttons.")
            if b1.button("Approve", key="approve", disabled=not ready):
                svc.decide(tid, "approve", reviewer)
                st.session_state["notice"] = "Case %s approved." % tid
                st.rerun()
            if b2.button("Block", key="block", disabled=not ready, type="primary"):
                svc.decide(tid, "block", reviewer)
                st.session_state["notice"] = "Case %s blocked." % tid
                st.rerun()

with tab_log:
    recs = svc.records()
    if svc.store.verify_chain():
        st.success("Decision chain verified: no record has been changed or removed.")
    else:
        st.error("Decision chain FAILED verification. A record was changed or removed.")
    if recs:
        show_truth = st.checkbox("Show simulated ground truth", key="show_truth")
        rows = []
        for r in recs:
            row = {"case": r["case_id"], "verdict": r["verdict"],
                   "decided by": r["decided_by"], "reviewer": r.get("reviewer"),
                   "agent recommended": r.get("recommendation"),
                   "model probability": round(r["p_fraud"], 3)}
            if show_truth:
                label = svc.true_label(r["case_id"])
                row["true label"] = "fraud" if label == 1 else "legitimate"
                row["correct"] = (r["verdict"] == "block") == (label == 1)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.write("No decisions yet.")

with tab_summary:
    s = svc.summary()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Decided", s["decided"])
    m2.metric("By a human", s["by_human"])
    m3.metric("Still in queue", s["remaining_in_queue"])
    m4.metric("Chain intact", "yes" if s["chain_ok"] else "NO")
    if s["agent_agreement"] is not None:
        st.write("Where the agent made a recommendation, the human agreed in %.0f%% of %d "
                 "cases." % (100 * s["agent_agreement"], s["agent_comparable"]))
    st.caption("Reviewers in this demo are simulated or local. Agreement here says "
               "nothing about how well evidence helps real analysts.")
