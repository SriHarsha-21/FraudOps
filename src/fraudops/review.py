"""Review service behind the Streamlit app: queue, case details and decisions.

All logic lives here so it can be tested without a browser. The app only draws it.
Reviews go through the LangGraph workflow, so a decision is the only way a human
verdict enters the hash-chained store. Card numbers are never shown. ASCII only.
"""
import hashlib
import hmac
import os

import numpy as np

from .agents.graph import build_graph, make_checkpointer, resume_case, start_case
from .agents.llm import make_explainer
from .agents.planner import rule_planner
from .agents.store import DecisionStore
from .agents.tools import CaseContext
from .backtest import _oracle


def check_pin(supplied):
    """True if no PIN is configured, or the supplied PIN matches FRAUDOPS_REVIEW_PIN."""
    expected = os.environ.get("FRAUDOPS_REVIEW_PIN", "")
    if not expected:
        return True
    return hmac.compare_digest(str(supplied).encode("utf-8"), expected.encode("utf-8"))


class ReviewService:
    def __init__(self, pipe, cfg, workdir=None, planner=None, explainer=None):
        self.pipe, self.cfg = pipe, cfg
        self.explainer = explainer or make_explainer(None)
        self._explained = {}
        self.ctx = CaseContext(pipe, cfg)
        if workdir:
            os.makedirs(workdir, exist_ok=True)
            db_path = os.path.join(workdir, "decisions.db")
            ck_path = os.path.join(workdir, "checkpoints.db")
        else:
            db_path = ck_path = ":memory:"
        self.store = DecisionStore(db_path)
        self.app = build_graph(self.ctx, self.store, make_checkpointer(ck_path),
                               planner=planner or rule_planner)

    def _cfg(self, txn_id):
        return {"configurable": {"thread_id": txn_id}}

    def queue(self, limit=50):
        """Undecided escalated test-period cases, highest amount times fraud probability first."""
        frame = self.ctx.frame
        rows = np.flatnonzero(self.pipe["masks"]["test"])
        rows = [i for i in rows if self.ctx.gate_actions[i] == "escalate"]
        amt = frame["amt"].to_numpy(float)
        rows.sort(key=lambda i: -amt[i] * float(self.ctx.p[i]))
        ids = [frame["txn_id"].iloc[i] for i in rows]
        return [t for t in ids if self.store.get(t) is None][:limit]

    def total_escalated(self):
        rows = np.flatnonzero(self.pipe["masks"]["test"])
        return int(sum(self.ctx.gate_actions[i] == "escalate" for i in rows))

    def details(self, txn_id):
        """Fields a reviewer sees. The card number is replaced by a short reference."""
        i = self.ctx.row(txn_id)
        r = self.ctx.frame.iloc[i]
        return {
            "amount": float(r["amt"]), "category": str(r["category"]),
            "merchant": str(r["merchant"]), "time": str(r["ts"]),
            "distance_km": float(self.ctx.X["dist_km"].iloc[i]),
            "card_ref": hashlib.sha256(str(r["cc_num"]).encode("utf-8")).hexdigest()[:6],
            "text_flag": bool(r["text_flag"]),
        }

    def explain(self, txn_id):
        """Short text for the reviewer. Cached per case. Never changes a verdict."""
        if txn_id not in self._explained:
            payload = self.pending_payload(txn_id)
            if payload is None:
                return None
            self._explained[txn_id] = self.explainer(payload)
        return self._explained[txn_id]

    def pending_payload(self, txn_id):
        """The paused review payload for a case, or None if it is not waiting."""
        snap = self.app.get_state(self._cfg(txn_id))
        if snap.next:
            for task in snap.tasks:
                if task.interrupts:
                    return task.interrupts[0].value
        return None

    def open_case(self, txn_id):
        """Return the review payload. A case already waiting is not run a second time."""
        existing = self.store.get(txn_id)
        if existing is not None:
            return {"status": "done", "verdict": existing["verdict"],
                    "decided_by": existing["decided_by"]}
        payload = self.pending_payload(txn_id)
        if payload is not None:
            return {"status": "paused", "payload": payload}
        return start_case(self.app, txn_id, self.cfg)

    def decide(self, txn_id, decision, reviewer):
        if decision not in ("approve", "block"):
            raise ValueError("decision must be 'approve' or 'block'")
        if not str(reviewer).strip():
            raise ValueError("a reviewer name is required")
        existing = self.store.get(txn_id)
        if existing is not None:  # decisions are final; a second click changes nothing
            return {"status": "done", "verdict": existing["verdict"],
                    "decided_by": existing["decided_by"]}
        if self.pending_payload(txn_id) is None:
            start_case(self.app, txn_id, self.cfg)
        return resume_case(self.app, txn_id,
                           {"decision": decision, "reviewer": str(reviewer).strip()}, self.cfg)

    def simulate_reviewer(self, n, seed=0):
        """Demo helper: a simulated reviewer decides the next n cases, erring at the set rate."""
        rng = np.random.default_rng(seed)
        y = self.ctx.y
        done = 0
        for tid in self.queue(limit=n):
            self.open_case(tid)
            truth = int(y[self.ctx.row(tid)])
            decision = _oracle(truth, rng, self.cfg["reviewer"]["error_rate"])
            self.decide(tid, decision, "simulated reviewer")
            done += 1
        return done

    def true_label(self, txn_id):
        """Simulated ground truth, for demo feedback only. Never shown before a decision."""
        if self.store.get(txn_id) is None:
            return None
        return int(self.ctx.y[self.ctx.row(txn_id)])

    def records(self):
        return self.store.all_records()

    def summary(self):
        recs = self.records()
        human = [r for r in recs if r["decided_by"] == "human"]
        comparable = [r for r in human if r.get("recommendation") in ("approve", "block")]
        agree = sum(r["recommendation"] == r["verdict"] for r in comparable)
        return {
            "decided": len(recs), "by_human": len(human),
            "by_gate": len(recs) - len(human),
            "remaining_in_queue": len(self.queue(limit=10 ** 9)),
            "agent_comparable": len(comparable),
            "agent_agreement": (agree / len(comparable)) if comparable else None,
            "chain_ok": self.store.verify_chain(),
        }
