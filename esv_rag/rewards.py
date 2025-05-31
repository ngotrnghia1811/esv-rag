"""
Composite reward calculator for ESV-RAG.

The reward function evaluates four dimensions (§3.6):
  r = α₁·r_quality + α₂·r_coherence + α₃·r_verify + α₄·r_efficiency

Default weights: α₁=0.4, α₂=0.3, α₃=0.2, α₄=0.1.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .state import QuestionStatus, ReasoningSkill, State
from .config import RewardConfig

logger = logging.getLogger(__name__)


class ESVRewardCalculator:
    """
    Multi-component reward function for ESV reasoning quality assessment.

    Components
    ----------
    answer_quality   : completeness, coherence, and consistency of the answer
    reasoning_coherence : skill diversity, chain depth, question relevance
    verification_success: verification pass-rate and coverage
    efficiency       : penalises overly long reasoning paths
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        cfg = config or RewardConfig()
        self.w_quality = cfg.answer_quality
        self.w_coherence = cfg.reasoning_coherence
        self.w_verify = cfg.verification_success
        self.w_efficiency = cfg.efficiency
        self.max_expected_steps = cfg.max_expected_steps
        self.optimal_questions = cfg.optimal_questions
        self.total_skills = cfg.total_skills

    def calculate_reward(self, state: State) -> float:
        """Return the composite reward ∈ [0, 1] for the given state."""
        aq = self._score_answer_quality(state)
        rc = self._score_reasoning_coherence(state)
        vs = self._score_verification_success(state)
        ef = self._score_efficiency(state)

        total = (self.w_quality * aq + self.w_coherence * rc
                 + self.w_verify * vs + self.w_efficiency * ef)

        logger.debug("Reward — AQ=%.3f RC=%.3f VS=%.3f EF=%.3f → %.3f",
                     aq, rc, vs, ef, total)
        return round(total, 4)

    # ------------------------------------------------------------------
    # Sub-scores
    # ------------------------------------------------------------------

    def _score_answer_quality(self, state: State) -> float:
        if not state.question_list or state.main_answer is None:
            return 0.0

        completeness = self._assess_completeness(state)
        coherence    = self._assess_logical_coherence(state)
        consistency  = self._assess_factual_consistency(state)
        return (completeness + coherence + consistency) / 3.0

    def _score_reasoning_coherence(self, state: State) -> float:
        if not state.question_list:
            return 0.0

        skill_diversity  = self._assess_skill_diversity(state)
        question_quality = self._assess_question_quality(state)
        reasoning_depth  = self._assess_reasoning_depth(state)
        return (skill_diversity + question_quality + reasoning_depth) / 3.0

    def _score_verification_success(self, state: State) -> float:
        verified = state.get_verified_questions()
        if not verified:
            if state.main_answer is None:
                return 0.0
            return 0.3  # answer exists but unverified

        passed = sum(1 for q in verified
                     if q.status == QuestionStatus.VERIFIED_PASS)
        pass_rate = passed / len(verified)
        coverage  = min(1.0, len(verified) / max(1, len(state.question_list)))
        return 0.7 * pass_rate + 0.3 * coverage

    def _score_efficiency(self, state: State) -> float:
        step_penalty   = max(0.0, 1.0 - state.current_step / self.max_expected_steps)
        q_count        = len(state.question_list)
        question_score = max(0.0, 1.0 - abs(q_count - self.optimal_questions)
                             / self.optimal_questions)
        return 0.5 * step_penalty + 0.5 * question_score

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assess_completeness(self, state: State) -> float:
        answered = len(state.get_answered_questions())
        total    = max(1, len(state.question_list))
        base     = answered / total
        if state.main_answer and len(state.main_answer.strip()) > 20:
            base = min(1.0, base + 0.2)
        return base

    def _assess_logical_coherence(self, state: State) -> float:
        answered = state.get_answered_questions()
        if not answered:
            return 0.0
        avg_conf = sum(q.confidence for q in answered) / len(answered)
        return avg_conf

    def _assess_factual_consistency(self, state: State) -> float:
        answered = state.get_answered_questions()
        if not answered:
            return 0.0
        confident = sum(1 for q in answered if q.confidence >= 0.7)
        return confident / len(answered)

    def _assess_skill_diversity(self, state: State) -> float:
        unique = len(set(q.skill for q in state.question_list))
        return min(1.0, unique / self.total_skills)

    def _assess_question_quality(self, state: State) -> float:
        if not state.question_list:
            return 0.0
        with_rationale = sum(1 for q in state.question_list if q.rationale)
        return with_rationale / len(state.question_list)

    def _assess_reasoning_depth(self, state: State) -> float:
        if not state.reasoning_history:
            return 0.0
        actions = [h.get("action", "") for h in state.reasoning_history]
        unique_actions = len(set(actions))
        depth_score = min(1.0, len(state.reasoning_history) / 5.0)
        variety_score = unique_actions / 3.0
        return 0.5 * depth_score + 0.5 * variety_score

    # ------------------------------------------------------------------
    # Terminal reward (used in RL training)
    # ------------------------------------------------------------------

    @staticmethod
    def terminal_reward(correct: bool, verified: bool, timeout: bool) -> float:
        """Assign discrete terminal reward (§3.6)."""
        if timeout:
            return -0.5
        if correct and verified:
            return 2.0
        if correct:
            return 1.0
        return -1.0
