# models/schemas.py
#
# Pydantic request/response models. Request bodies were already fully
# validated everywhere they existed; response_model was declared on only
# 1 of ~25 endpoints (ZoneSelectResponse below), leaving /docs undocumented
# for the rest and giving FastAPI nothing to validate outgoing payloads
# against. This file adds typed response models for the endpoints most
# likely to be consumed programmatically or cited as "the API" (per the
# design review: /health, /zone/status, /eval/summary, /data/*,
# /compare/results) rather than exhaustively typing all 25 — several
# remaining endpoints (e.g. GET /config, GET /compare/results' nested
# per-controller metrics) return intentionally dynamic, frequently-extended
# dicts where a strict schema would be a maintenance trap rather than a
# genuine safety net; those keep loosely-typed `Dict[str, Any]` fields
# where the dynamism is real, while still declaring the stable outer shape.
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class ZoneSelectRequest(BaseModel):
    zone: Literal[1, 2, 3, 4]
    traffic: Literal["low", "medium", "high", "rush_hour"]


class ZoneSelectResponse(BaseModel):
    zone: int
    traffic: str
    active_signal: int
    signal_color: str
    duration_s: int
    arduino_ack: str
    action: int
    action_name: str
    reward: float
    message: str


class ZoneStatusResponse(BaseModel):
    mode: str
    active_signal: int
    emergency_zone: Optional[int] = None
    step: int
    queue_lengths: Dict[int, int]
    traffic_pattern: str
    signal_states: Dict[int, str]
    status: str


class ZoneClearResponse(BaseModel):
    message: str
    queue: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    mode: str
    current_zone: int
    arduino: bool
    rl_epsilon: float
    connected_clients: int


class RootResponse(BaseModel):
    message: str
    mode: str
    current_zone: int
    arduino: bool


class ArduinoStatusResponse(BaseModel):
    connected: bool
    simulation_mode: bool
    port: str
    baud: int
    commands_sent: int
    commands_success: int
    commands_failed: int
    success_rate: float
    reconnect_attempts: int
    last_error: Optional[str] = None
    last_ack: Optional[str] = None


class EvalSummaryPerformance(BaseModel):
    avg_reward: float
    avg_queue_length: float
    avg_wait_time: float
    clearance_rate: float
    emergency_accuracy: float
    fairness_index: float


class EvalSummaryTotals(BaseModel):
    vehicles_arrived: int
    vehicles_cleared: int
    signal_switches: int
    emergency_events: int


class EvalSummaryResponse(BaseModel):
    mode: str
    traffic_pattern: str
    total_steps: int
    uptime_seconds: float
    performance: EvalSummaryPerformance
    totals: EvalSummaryTotals


class DataStatsResponse(BaseModel):
    total_emergencies: int
    resolved_emergencies: int
    avg_response_time: float
    total_snapshots: int
    total_benchmarks: int
    total_decisions: int
    db_path: str
    db_size_kb: float


class CompareResultsResponse(BaseModel):
    """Shape of ControllerComparator.get_summary(). `controllers` maps
    controller name -> a metrics dict whose keys grow as new evaluation
    metrics are added (avg_reward, avg_ambulance_delay, composite_score,
    etc.) — kept as Dict[str, Any] deliberately rather than enumerated,
    since a strict schema here would need editing every time a metric is
    added and would provide little real safety (these are report values,
    not control-flow inputs).

    environment/environment_caveat/oracle_controllers/oracle_reference:
    added alongside the oracle-reframing and environment-provenance
    fixes — Optional/defaulted so this remains backward compatible with
    any consumer still expecting the pre-fix shape."""

    controllers: Dict[str, Dict[str, Any]] = {}
    environment: Optional[str] = None
    environment_caveat: Optional[str] = None
    oracle_controllers: List[str] = []
    oracle_reference: Dict[str, Dict[str, Any]] = {}
    best_controller: str = ""
    best_per_metric: Dict[str, Optional[str]] = {}
    composite_score_sensitivity: Dict[str, Any] = {}
    rl_improvements: Dict[str, Any] = {}
    controller_profiles: Dict[str, str] = {}
    error: Optional[str] = None
