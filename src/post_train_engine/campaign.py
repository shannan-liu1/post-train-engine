"""Local transactional campaign control plane."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Iterator, Literal


@dataclass(frozen=True)
class ExperimentProposal:
    campaign_id: str
    hypothesis_id: str
    method: str
    task_name: str
    parent_candidate_id: str
    comparison_group_id: str
    comparison_axis: str
    parameters: dict[str, Any]
    mechanism_parent_ids: tuple[str, ...] = ()
    estimated_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "campaign_id",
            "hypothesis_id",
            "method",
            "task_name",
            "parent_candidate_id",
            "comparison_group_id",
            "comparison_axis",
        ):
            if not getattr(self, name):
                raise ValueError(f"proposal {name} must be non-empty")
        if self.comparison_axis not in self.parameters:
            raise ValueError(
                "proposal parameters must include the declared comparison_axis"
            )
        if (
            type(self.estimated_cost_usd) is bool
            or not isinstance(self.estimated_cost_usd, int | float)
            or not isfinite(float(self.estimated_cost_usd))
            or self.estimated_cost_usd < 0
        ):
            raise ValueError("proposal estimated_cost_usd must be finite and non-negative")

    @property
    def proposal_hash(self) -> str:
        body = asdict(self)
        body["mechanism_parent_ids"] = sorted(self.mechanism_parent_ids)
        payload = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProposalOutcome:
    run_id: str
    status: Literal["failed", "rejected", "promoted"]
    decision: Literal["reject", "promote", "failed"]
    failure_category: str | None
    cost_usd: float | None
    primary_delta: float | None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("outcome run_id must be non-empty")
        expected_decision = {
            "failed": "failed",
            "rejected": "reject",
            "promoted": "promote",
        }[self.status]
        if self.decision != expected_decision:
            raise ValueError("outcome status and decision must agree")
        for name, value in (
            ("cost_usd", self.cost_usd),
            ("primary_delta", self.primary_delta),
        ):
            if value is not None and (
                type(value) is bool
                or not isinstance(value, int | float)
                or not isfinite(float(value))
            ):
                raise ValueError(f"outcome {name} must be finite when provided")
        if self.cost_usd is not None and self.cost_usd < 0:
            raise ValueError("outcome cost_usd must be non-negative")


@dataclass(frozen=True)
class WorkLease:
    proposal_id: str
    worker_id: str
    expires_at: float


@dataclass(frozen=True)
class ExperimentRecommendation:
    category: Literal[
        "evaluation_repair",
        "data_repair",
        "sft",
        "preference_training",
        "rlvr",
        "distillation",
        "protected_replay",
        "runtime_work",
        "task_expansion",
    ]
    rationale: str

    def to_dict(self) -> dict[str, str]:
        return {"category": self.category, "rationale": self.rationale}


def recommend_from_run(
    decision: dict[str, Any],
    *,
    evidence_signal: str | None = None,
    training_eligible: bool | None = None,
) -> ExperimentRecommendation:
    signal_mapping: dict[str, ExperimentRecommendation] = {
        "verifier_disagreement": ExperimentRecommendation(
            "evaluation_repair", "Repair verifier validity before optimizing its reward."
        ),
        "data_quality": ExperimentRecommendation(
            "data_repair", "Repair invalid or low-signal evidence before training."
        ),
        "parse_regression": ExperimentRecommendation(
            "sft", "Use targeted supervised format repair on eligible Traces."
        ),
        "preference_signal": ExperimentRecommendation(
            "preference_training", "Use paired preference evidence with clear separation."
        ),
        "sparse_verifiable_reward": ExperimentRecommendation(
            "rlvr", "Use RLVR where executable verifier signal is trusted."
        ),
        "teacher_advantage": ExperimentRecommendation(
            "distillation", "Distill measured teacher advantage under OOD guards."
        ),
        "protected_regression": ExperimentRecommendation(
            "protected_replay", "Mix eligible protected replay before another candidate."
        ),
        "runtime_bottleneck": ExperimentRecommendation(
            "runtime_work", "Optimize the measured bottleneck under output equivalence."
        ),
        "task_saturation": ExperimentRecommendation(
            "task_expansion", "Add a new executable-verifier task after saturation."
        ),
    }
    if evidence_signal in signal_mapping:
        return signal_mapping[evidence_signal]
    if training_eligible is False:
        return ExperimentRecommendation(
            "data_repair",
            "Collect discriminative evidence because the parent-policy probe found no learnable frontier.",
        )
    reasons = " ".join(str(item) for item in decision.get("rejection_reasons", ())).lower()
    if "underpowered_eval" in reasons:
        return ExperimentRecommendation(
            "evaluation_repair", "Increase independent promotion evidence before training again."
        )
    if "verifier" in reasons:
        return signal_mapping["verifier_disagreement"]
    if "parse_regression" in reasons:
        return ExperimentRecommendation(
            "sft", "Use targeted supervised format repair on eligible Traces."
        )
    if "easy_regression" in reasons or "severity" in reasons or "protected" in reasons:
        return signal_mapping["protected_regression"]
    if "token" in reasons or "cost" in reasons:
        return signal_mapping["runtime_bottleneck"]
    if decision.get("decision") == "promote":
        return ExperimentRecommendation(
            "task_expansion",
            "Test a controlled composition or harder executable-verifier slice against the new Incumbent.",
        )
    return ExperimentRecommendation(
        "data_repair",
        "Collect more discriminative eligible evidence before choosing another trainer.",
    )


class CampaignStore:
    """SQLite index over immutable campaign evidence and coordination state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create_campaign(
        self,
        campaign_id: str,
        *,
        quality_budget: int,
        cost_budget_usd: float,
    ) -> None:
        if quality_budget <= 0 or cost_budget_usd < 0:
            raise ValueError("campaign budgets must be positive and non-negative")
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO campaigns VALUES (?, ?, ?, 0, 0.0)",
                (campaign_id, quality_budget, cost_budget_usd),
            )

    def add_hypothesis(self, campaign_id: str, hypothesis_id: str, statement: str) -> None:
        if not statement:
            raise ValueError("hypothesis statement must be non-empty")
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO hypotheses VALUES (?, ?, ?)",
                (hypothesis_id, campaign_id, statement),
            )

    def submit_proposal(self, proposal: ExperimentProposal) -> str:
        proposal_id = "proposal-" + proposal.proposal_hash.removeprefix("sha256:")[:20]
        with self._transaction() as connection:
            duplicate = connection.execute(
                "SELECT proposal_id FROM proposals WHERE proposal_hash = ?",
                (proposal.proposal_hash,),
            ).fetchone()
            if duplicate is not None:
                raise ValueError(f"duplicate proposal: {duplicate[0]}")
            connection.execute(
                """
                INSERT INTO proposals(
                    proposal_id, campaign_id, hypothesis_id, proposal_hash,
                    method, task_name, parent_candidate_id, comparison_group_id,
                    comparison_axis, parameters_json, mechanism_parent_ids_json,
                    estimated_cost_usd, state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    proposal_id,
                    proposal.campaign_id,
                    proposal.hypothesis_id,
                    proposal.proposal_hash,
                    proposal.method,
                    proposal.task_name,
                    proposal.parent_candidate_id,
                    proposal.comparison_group_id,
                    proposal.comparison_axis,
                    _json(proposal.parameters),
                    _json(sorted(proposal.mechanism_parent_ids)),
                    proposal.estimated_cost_usd,
                ),
            )
        return proposal_id

    def claim_proposal(
        self,
        proposal_id: str,
        *,
        worker_id: str,
        ttl_seconds: int,
    ) -> WorkLease | None:
        if ttl_seconds <= 0:
            raise ValueError("lease ttl_seconds must be positive")
        now = time.time()
        expires = now + ttl_seconds
        with self._transaction() as connection:
            connection.execute("DELETE FROM leases WHERE expires_at <= ?", (now,))
            proposal = connection.execute(
                """
                SELECT state, campaign_id, estimated_cost_usd
                FROM proposals WHERE proposal_id = ?
                """,
                (proposal_id,),
            ).fetchone()
            if proposal is None:
                raise ValueError(f"unknown proposal: {proposal_id}")
            if proposal[0] != "pending":
                return None
            existing = connection.execute(
                "SELECT worker_id FROM leases WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
            if existing is not None:
                return None
            campaign_id = str(proposal[1])
            estimated_cost_usd = float(proposal[2])
            quality_budget, cost_budget_usd = connection.execute(
                """
                SELECT quality_budget, cost_budget_usd
                FROM campaigns WHERE campaign_id = ?
                """,
                (campaign_id,),
            ).fetchone()
            completed_runs, completed_cost = connection.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(COALESCE(o.cost_usd, p.estimated_cost_usd)), 0.0)
                FROM outcomes o
                JOIN proposals p ON p.proposal_id = o.proposal_id
                WHERE p.campaign_id = ?
                """,
                (campaign_id,),
            ).fetchone()
            active_runs, active_cost = connection.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(p.estimated_cost_usd), 0.0)
                FROM leases l
                JOIN proposals p ON p.proposal_id = l.proposal_id
                WHERE p.campaign_id = ?
                """,
                (campaign_id,),
            ).fetchone()
            if int(completed_runs) + int(active_runs) + 1 > int(quality_budget):
                raise ValueError("quality budget exceeded")
            accounted_cost = float(completed_cost) + float(active_cost)
            if accounted_cost + estimated_cost_usd > float(cost_budget_usd):
                raise ValueError("cost budget exceeded")
            connection.execute(
                "INSERT INTO leases VALUES (?, ?, ?)",
                (proposal_id, worker_id, expires),
            )
        return WorkLease(proposal_id, worker_id, expires)

    def get_outcome(self, proposal_id: str) -> ProposalOutcome:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT run_id, status, decision, failure_category, cost_usd, primary_delta
                FROM outcomes WHERE proposal_id = ?
                """,
                (proposal_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"proposal has no outcome: {proposal_id}")
        return ProposalOutcome(*row)

    def finalize_run(
        self,
        *,
        campaign_id: str,
        proposal_id: str,
        worker_id: str,
        expected_incumbent_candidate_id: str,
        candidate_id: str,
        outcome: ProposalOutcome,
        promotion_suite_id: str | None = None,
        promotion_suite_version: str | None = None,
        max_suite_exposures: int | None = None,
    ) -> None:
        suite_values = (
            promotion_suite_id,
            promotion_suite_version,
            max_suite_exposures,
        )
        if any(value is not None for value in suite_values) and not all(
            value is not None for value in suite_values
        ):
            raise ValueError("promotion suite binding requires id, version, and max exposures")
        if max_suite_exposures is not None and max_suite_exposures <= 0:
            raise ValueError("max_suite_exposures must be positive")
        with self._transaction() as connection:
            existing = connection.execute(
                """
                SELECT run_id, status, decision, failure_category, cost_usd, primary_delta
                FROM outcomes WHERE proposal_id = ?
                """,
                (proposal_id,),
            ).fetchone()
            if existing is not None:
                if ProposalOutcome(*existing) != outcome:
                    raise ValueError("existing proposal outcome differs from Run evidence")
                return
            proposal = connection.execute(
                """
                SELECT campaign_id, state, parent_candidate_id
                FROM proposals WHERE proposal_id = ?
                """,
                (proposal_id,),
            ).fetchone()
            if proposal is None or proposal[0] != campaign_id:
                raise ValueError("proposal does not belong to campaign")
            if proposal[1] != "pending":
                raise ValueError("proposal is not pending")
            if str(proposal[2]) != expected_incumbent_candidate_id:
                raise ValueError(
                    "proposal parent does not match the expected Incumbent"
                )
            lease = connection.execute(
                "SELECT worker_id, expires_at FROM leases WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
            if lease is None or lease[0] != worker_id or float(lease[1]) <= time.time():
                raise ValueError("active proposal lease is required to finalize a Run")
            if promotion_suite_id is not None:
                exposure_count = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM suite_exposures
                        WHERE campaign_id = ? AND suite_id = ? AND suite_version = ?
                        """,
                        (campaign_id, promotion_suite_id, promotion_suite_version),
                    ).fetchone()[0]
                )
                if exposure_count >= int(max_suite_exposures):
                    raise ValueError(
                        f"promotion suite rotation required after {exposure_count} exposures"
                    )
                connection.execute(
                    "INSERT INTO suite_exposures VALUES (?, ?, ?, ?, ?)",
                    (
                        campaign_id,
                        promotion_suite_id,
                        promotion_suite_version,
                        candidate_id,
                        time.time(),
                    ),
                )
            connection.execute(
                "INSERT INTO outcomes VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    proposal_id,
                    outcome.run_id,
                    outcome.status,
                    outcome.decision,
                    outcome.failure_category,
                    outcome.cost_usd,
                    outcome.primary_delta,
                ),
            )
            if outcome.status == "promoted":
                cursor = connection.execute(
                    """
                    UPDATE incumbents
                    SET candidate_id = ?, run_id = ?, generation = generation + 1
                    WHERE campaign_id = ? AND candidate_id = ?
                    """,
                    (
                        candidate_id,
                        outcome.run_id,
                        campaign_id,
                        expected_incumbent_candidate_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ValueError("incumbent changed before promotion transaction")
            connection.execute(
                "UPDATE proposals SET state = 'completed' WHERE proposal_id = ?",
                (proposal_id,),
            )
            connection.execute("DELETE FROM leases WHERE proposal_id = ?", (proposal_id,))

    def current_incumbent(self, campaign_id: str) -> dict[str, str | int]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT candidate_id, run_id, generation FROM incumbents WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"campaign has no Incumbent: {campaign_id}")
        return {
            "candidate_id": str(row[0]),
            "run_id": str(row[1]),
            "generation": int(row[2]),
        }

    def initialize_incumbent(
        self,
        campaign_id: str,
        *,
        candidate_id: str,
        run_id: str,
    ) -> None:
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO incumbents VALUES (?, ?, ?, 0)",
                (campaign_id, candidate_id, run_id),
            )

    def campaign_usage(self, campaign_id: str) -> dict[str, float | int]:
        with self._connect() as connection:
            campaign = connection.execute(
                "SELECT campaign_id FROM campaigns WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if campaign is None:
                raise ValueError(f"unknown campaign: {campaign_id}")
            completed_runs, completed_cost = connection.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(COALESCE(o.cost_usd, p.estimated_cost_usd)), 0.0)
                FROM outcomes o
                JOIN proposals p ON p.proposal_id = o.proposal_id
                WHERE p.campaign_id = ?
                """,
                (campaign_id,),
            ).fetchone()
            active_runs, active_cost = connection.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(p.estimated_cost_usd), 0.0)
                FROM leases l
                JOIN proposals p ON p.proposal_id = l.proposal_id
                WHERE p.campaign_id = ? AND l.expires_at > ?
                """,
                (campaign_id, time.time()),
            ).fetchone()
        return {
            "reserved_runs": int(completed_runs) + int(active_runs),
            "reserved_cost_usd": float(completed_cost) + float(active_cost),
            "completed_runs": int(completed_runs),
            "active_leases": int(active_runs),
        }

    def record_suite_exposure(
        self,
        campaign_id: str,
        *,
        suite_id: str,
        suite_version: str,
        candidate_id: str,
    ) -> None:
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO suite_exposures VALUES (?, ?, ?, ?, ?)
                """,
                (campaign_id, suite_id, suite_version, candidate_id, time.time()),
            )

    def suite_exposure(self, campaign_id: str, suite_id: str, suite_version: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) FROM suite_exposures
                WHERE campaign_id = ? AND suite_id = ? AND suite_version = ?
                """,
                (campaign_id, suite_id, suite_version),
            ).fetchone()
        return int(row[0])

    def require_suite_available(
        self,
        campaign_id: str,
        *,
        suite_id: str,
        suite_version: str,
        max_exposures: int,
    ) -> None:
        if max_exposures <= 0:
            raise ValueError("max_exposures must be positive")
        exposures = self.suite_exposure(campaign_id, suite_id, suite_version)
        if exposures >= max_exposures:
            raise ValueError(
                f"promotion suite rotation required after {exposures} exposures"
            )

    def recommend_next_experiment(
        self,
        campaign_id: str,
        *,
        failure_category: str | None = None,
    ) -> ExperimentRecommendation:
        self.campaign_usage(campaign_id)
        return recommend_from_run(
            {"decision": "reject", "rejection_reasons": []},
            evidence_signal=failure_category,
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS campaigns(
                    campaign_id TEXT PRIMARY KEY,
                    quality_budget INTEGER NOT NULL,
                    cost_budget_usd REAL NOT NULL,
                    reserved_runs INTEGER NOT NULL,
                    reserved_cost_usd REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS hypotheses(
                    hypothesis_id TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
                    statement TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS proposals(
                    proposal_id TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
                    hypothesis_id TEXT NOT NULL REFERENCES hypotheses(hypothesis_id),
                    proposal_hash TEXT NOT NULL UNIQUE,
                    method TEXT NOT NULL,
                    task_name TEXT NOT NULL,
                    parent_candidate_id TEXT NOT NULL,
                    comparison_group_id TEXT NOT NULL,
                    comparison_axis TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    mechanism_parent_ids_json TEXT NOT NULL,
                    estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
                    state TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS leases(
                    proposal_id TEXT PRIMARY KEY REFERENCES proposals(proposal_id),
                    worker_id TEXT NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS outcomes(
                    proposal_id TEXT PRIMARY KEY REFERENCES proposals(proposal_id),
                    run_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    failure_category TEXT,
                    cost_usd REAL,
                    primary_delta REAL
                );
                CREATE TABLE IF NOT EXISTS incumbents(
                    campaign_id TEXT PRIMARY KEY REFERENCES campaigns(campaign_id),
                    candidate_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    generation INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS suite_exposures(
                    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
                    suite_id TEXT NOT NULL,
                    suite_version TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    exposed_at REAL NOT NULL,
                    UNIQUE(campaign_id, suite_id, suite_version, candidate_id)
                );
                """
            )
            proposal_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(proposals)")
            }
            if "estimated_cost_usd" not in proposal_columns:
                connection.execute(
                    "ALTER TABLE proposals ADD COLUMN estimated_cost_usd REAL NOT NULL DEFAULT 0.0"
                )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


__all__ = [
    "CampaignStore",
    "ExperimentProposal",
    "ExperimentRecommendation",
    "ProposalOutcome",
    "WorkLease",
    "recommend_from_run",
]
