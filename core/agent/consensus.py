"""
Consensus Arbiter — Multi-Agent Independent Voting (AWS)
==========================================================

Runs 3 independent Sonnet sessions in parallel to classify
diagnostic results with structural hallucination mitigation.

Classification:
    - Consensus (3/3): all agree → confirmed
    - Majority (2/3): majority finding → priority review
    - Minority Report (1/3): dissenting opinion → preserved as 2nd priority

IMPORTANT: Minority reports are never deleted once identified.
The minority opinion's reasoning is preserved for human review.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.recommendation.reason.llm_provider import AbstractLLMProvider

logger = logging.getLogger(__name__)

__all__ = ["ConsensusArbiter", "ConsensusResult", "AgentVote"]


# System prompt perspectives for independent voting
_PERSPECTIVES = {
    "alpha": "보수적 관점에서 판단하세요. 위험 신호를 놓치지 않는 것이 중요합니다. 의심스러우면 WARN으로 판단하세요.",
    "beta": "통계적 유의성을 중시하세요. 모수(sample size)가 충분한지, 변동이 통계적으로 유의한지 확인하세요.",
    "gamma": "비즈니스 영향을 중시하세요. 실제 고객 경험과 비즈니스 KPI에 미치는 영향을 기준으로 판단하세요.",
}


@dataclass
class AgentVote:
    """Single agent's vote."""
    agent_id: str        # "alpha", "beta", "gamma"
    perspective: str     # system prompt perspective used
    verdict: str         # "PASS", "WARN", "FAIL"
    confidence: float    # 0.0-1.0
    reasoning: str       # detailed reasoning (500-800 tokens)
    recommendation: str  # suggested action


@dataclass
class ConsensusResult:
    """Result of multi-agent consensus."""
    consensus_type: str    # "consensus", "majority", "minority"
    final_verdict: str     # the agreed/majority verdict
    votes: List[AgentVote] = field(default_factory=list)
    minority_report: Optional[Dict[str, Any]] = None  # minority opinion if exists
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "consensus_type": self.consensus_type,
            "final_verdict": self.final_verdict,
            "votes": [
                {
                    "agent_id": v.agent_id,
                    "verdict": v.verdict,
                    "confidence": v.confidence,
                    "reasoning": v.reasoning,
                    "recommendation": v.recommendation,
                }
                for v in self.votes
            ],
            "timestamp": self.timestamp,
        }
        if self.minority_report:
            result["minority_report"] = self.minority_report
        return result


class ConsensusArbiter:
    """Multi-agent consensus via independent parallel voting.

    Args:
        llm_provider: LLM provider for Bedrock Sonnet calls.
        config: Consensus configuration dict.
    """

    def __init__(
        self,
        llm_provider: "AbstractLLMProvider",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._llm = llm_provider
        cfg = config or {}
        self._num_agents = cfg.get("agents", 3)
        self._parallel = cfg.get("parallel", True)
        self._perspectives = cfg.get("perspectives", _PERSPECTIVES)

    def evaluate(
        self,
        item_description: str,
        measurements: Dict[str, Any],
        rule_engine_verdict: str = "",
        similar_cases: Optional[List[Dict]] = None,
    ) -> ConsensusResult:
        """Run multi-agent consensus evaluation.

        Args:
            item_description: Checklist item description.
            measurements: Current measurement data.
            rule_engine_verdict: What the deterministic rule engine decided.
            similar_cases: Past similar cases from DiagnosticCaseStore.

        Returns:
            ConsensusResult with classification and minority report.
        """
        # Build base prompt
        base_prompt = self._build_prompt(
            item_description, measurements, rule_engine_verdict, similar_cases
        )

        # Run independent votes
        votes = self._collect_votes(base_prompt)

        # Classify consensus
        result = self._classify(votes)

        logger.info(
            "Consensus: %s (verdicts: %s)",
            result.consensus_type,
            [v.verdict for v in votes],
        )
        return result

    def _build_prompt(
        self,
        item_description: str,
        measurements: Dict[str, Any],
        rule_engine_verdict: str,
        similar_cases: Optional[List[Dict]],
    ) -> str:
        """Build the base evaluation prompt (perspective is prepended per agent)."""
        parts = [
            "## 진단 대상",
            f"항목: {item_description}",
            f"룰 엔진 판정: {rule_engine_verdict}" if rule_engine_verdict else "",
            "",
            "## 현재 측정값",
        ]

        for key, value in measurements.items():
            parts.append(f"- {key}: {value}")

        if similar_cases:
            parts.append("")
            parts.append("## 과거 유사 케이스")
            for i, case in enumerate(similar_cases[:3], 1):
                parts.append(f"  #{i}: {case.get('finding', '')} → {case.get('resolution', '미해결')}")

        parts.append("")
        parts.append("## 지시")
        parts.append("위 정보를 분석하고 다음을 JSON으로 응답하세요:")
        parts.append('{"verdict": "PASS|WARN|FAIL", "confidence": 0.0-1.0,')
        parts.append(' "reasoning": "상세한 판단 근거 (한국어, 300자 이상)",')
        parts.append(' "recommendation": "권고 조치"}')

        return "\n".join(p for p in parts if p is not None)

    def _collect_votes(self, base_prompt: str) -> List[AgentVote]:
        """Collect votes from all agents (parallel or sequential)."""
        agent_ids = list(self._perspectives.keys())[:self._num_agents]

        if self._parallel:
            return self._collect_parallel(agent_ids, base_prompt)
        return self._collect_sequential(agent_ids, base_prompt)

    def _collect_parallel(self, agent_ids: List[str], base_prompt: str) -> List[AgentVote]:
        """Collect votes in parallel using ThreadPoolExecutor."""
        votes = []
        with ThreadPoolExecutor(max_workers=self._num_agents) as executor:
            futures = {
                executor.submit(self._single_vote, aid, base_prompt): aid
                for aid in agent_ids
            }
            for future in as_completed(futures):
                aid = futures[future]
                try:
                    vote = future.result()
                    votes.append(vote)
                except Exception as e:
                    logger.error("Agent %s voting failed: %s", aid, e)
                    votes.append(AgentVote(
                        agent_id=aid,
                        perspective=self._perspectives.get(aid, ""),
                        verdict="WARN",  # conservative default
                        confidence=0.0,
                        reasoning=f"투표 실패: {e}",
                        recommendation="수동 검토 필요",
                    ))
        return votes

    def _collect_sequential(self, agent_ids: List[str], base_prompt: str) -> List[AgentVote]:
        """Collect votes sequentially."""
        return [self._single_vote(aid, base_prompt) for aid in agent_ids]

    def _single_vote(self, agent_id: str, base_prompt: str) -> AgentVote:
        """Get a single agent's vote via LLM call."""
        perspective = self._perspectives.get(agent_id, "")
        full_prompt = f"## 관점\n{perspective}\n\n{base_prompt}"

        try:
            response = self._llm.generate(full_prompt)
            parsed = self._parse_vote(response)
            return AgentVote(
                agent_id=agent_id,
                perspective=perspective,
                verdict=parsed.get("verdict", "WARN"),
                confidence=parsed.get("confidence", 0.5),
                reasoning=parsed.get("reasoning", response[:500]),
                recommendation=parsed.get("recommendation", ""),
            )
        except Exception as e:
            logger.error("Agent %s LLM call failed: %s", agent_id, e)
            return AgentVote(
                agent_id=agent_id,
                perspective=perspective,
                verdict="WARN",
                confidence=0.0,
                reasoning=f"LLM 호출 실패: {e}",
                recommendation="수동 검토 필요",
            )

    def _parse_vote(self, response: str) -> Dict[str, Any]:
        """Parse structured vote from LLM response."""
        import json as _json
        import re

        # Try direct JSON parse
        try:
            return _json.loads(response)
        except _json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        json_match = re.search(r'\{[^{}]*"verdict"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return _json.loads(json_match.group())
            except _json.JSONDecodeError:
                pass

        # Fallback: extract verdict from text
        verdict = "WARN"
        for v in ["FAIL", "PASS", "WARN"]:
            if v in response.upper():
                verdict = v
                break

        return {
            "verdict": verdict,
            "confidence": 0.5,
            "reasoning": response[:500],
            "recommendation": "",
        }

    def _classify(self, votes: List[AgentVote]) -> ConsensusResult:
        """Classify consensus from votes.

        - 3/3 same: consensus
        - 2/3 same: majority (1 minority)
        - All different: majority = WARN (conservative), all are minorities
        """
        if not votes:
            return ConsensusResult(
                consensus_type="consensus",
                final_verdict="WARN",
            )

        # Count verdicts
        verdict_counts: Dict[str, int] = {}
        for v in votes:
            verdict_counts[v.verdict] = verdict_counts.get(v.verdict, 0) + 1

        # Rule 1: FAIL — 1명이라도 FAIL이면 무조건 FAIL 에스컬레이션
        if any(v.verdict == "FAIL" for v in votes):
            minority_votes = [v for v in votes if v.verdict != "FAIL"]
            minority_report = None
            if minority_votes:
                minority_report = {
                    "dissenting_agents": [v.agent_id for v in minority_votes],
                    "dissenting_verdict": minority_votes[0].verdict,
                    "reasoning": minority_votes[0].reasoning,
                }
            return ConsensusResult(
                consensus_type="majority",
                final_verdict="FAIL",
                votes=votes,
                minority_report=minority_report,
            )

        # Rule 2: PASS — 만장일치(3/3 PASS)만 통과
        if len(verdict_counts) == 1 and votes[0].verdict == "PASS":
            return ConsensusResult(
                consensus_type="consensus",
                final_verdict="PASS",
                votes=votes,
            )

        # Rule 3: 만장일치 WARN
        if len(verdict_counts) == 1 and votes[0].verdict == "WARN":
            return ConsensusResult(
                consensus_type="consensus",
                final_verdict="WARN",
                votes=votes,
            )

        # Rule 4: 의견 불일치 (2P+1W, 2W+1P 등) → WARN + minority_report
        # PASS는 만장일치가 아니면 통과 불가, 보수적으로 WARN 처리
        majority_verdict = max(verdict_counts, key=verdict_counts.get)
        minority_votes = [v for v in votes if v.verdict != majority_verdict]

        minority_report = {
            "dissenting_agents": [v.agent_id for v in minority_votes],
            "dissenting_verdict": minority_votes[0].verdict if minority_votes else "",
            "reasoning": minority_votes[0].reasoning if minority_votes else "",
            "note": "만장일치 미달 — 검토 필요",
        }

        return ConsensusResult(
            consensus_type="majority",
            final_verdict="WARN",
            votes=votes,
            minority_report=minority_report,
        )
