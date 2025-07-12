"""
Planners for ESV-RAG.

SimplePlanner  — fixed E→S→V→S→V sequence (baseline ablation)
MCTSPlanner    — adaptive MCTS-guided planning (full model)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import ESVConfig, MCTSConfig
from .state import State
from .actions import ExplorationAction, SolvingAction, VerificationAction
from .generator import Generator
from .retriever import RetrieverClient
from .rewards import ESVRewardCalculator
from .mcts import ESVMCTS, ESVMCTSNode

logger = logging.getLogger(__name__)


@dataclass
class PlanningResult:
    """Outcome of a planning run."""
    success: bool
    final_state: Optional[State] = None
    action_sequence: List[str] = field(default_factory=list)
    planning_time: float = 0.0
    num_rollouts: int = 0
    final_reward: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# Simple sequential planner
# ===========================================================================

class SimplePlanner:
    """
    Execute a fixed action sequence: E → S → V → S → V.

    Used as the "No MCTS" ablation baseline (Table 2, row 1).
    """

    SEQUENCE = ["E", "S", "V", "S", "V"]

    def __init__(self,
                 explore_action: ExplorationAction,
                 solve_action:   SolvingAction,
                 verify_action:  VerificationAction,
                 early_stop_confidence: float = 0.9):
        self.explore  = explore_action
        self.solve    = solve_action
        self.verify   = verify_action
        self.early_stop_confidence = early_stop_confidence

    def run(self, question: str) -> PlanningResult:
        state    = State(main_question=question)
        actions  = []
        start    = time.time()

        action_map = {"E": self.explore, "S": self.solve, "V": self.verify}

        for action_key in self.SEQUENCE:
            action = action_map[action_key]
            result = action.execute(state)
            actions.append(action_key)

            if not result.success:
                logger.warning("[SimplePlanner] %s failed: %s", action_key, result.error)
                continue

            if (state.verification_passed
                    and state.main_answer_confidence >= self.early_stop_confidence):
                logger.info("[SimplePlanner] Early stop at step %d", state.current_step)
                break

        reward = ESVRewardCalculator().calculate_reward(state)
        return PlanningResult(
            success=state.main_answer is not None,
            final_state=state,
            action_sequence=actions,
            planning_time=time.time() - start,
            final_reward=reward,
        )


# ===========================================================================
# MCTS-based planner
# ===========================================================================

class MCTSPlanner:
    """
    MCTS-guided adaptive planner.

    Runs ESVMCTS.search() on the initial state, iteratively selecting
    actions until a terminal state is reached or budget is exhausted.
    """

    def __init__(self,
                 generator: Generator,
                 retriever: Optional[RetrieverClient] = None,
                 config: Optional[ESVConfig] = None):
        self.generator = generator
        self.retriever = retriever
        self.config    = config or ESVConfig()

        action_cfg = self.config.action
        self.explore = ExplorationAction(generator, retriever, action_cfg)
        self.solve   = SolvingAction(generator, retriever, action_cfg)
        self.verify  = VerificationAction(generator, retriever, action_cfg)

        self.reward_calc = ESVRewardCalculator(self.config.reward)
        self.mcts        = ESVMCTS(self.config.mcts)

    def run(self, question: str) -> PlanningResult:
        """
        Execute MCTS planning for *question*.

        The loop re-roots the tree at the chosen child after each
        best-action selection, mimicking online MCTS replanning.
        """
        start      = time.time()
        state      = State(main_question=question)
        actions    = []
        num_rolls  = 0

        root = self._make_node(state)
        max_outer  = self.config.mcts.max_depth

        for step in range(max_outer):
            if root.is_terminal():
                logger.info("[MCTSPlanner] Terminal at step %d", step)
                break

            best = self.mcts.search(root)
            num_rolls += self.config.mcts.num_simulations

            if best is root:
                logger.warning("[MCTSPlanner] No improvement at step %d; stopping", step)
                break

            actions.append(best.parent_action or "?")
            root = self._make_node(best.state)   # re-root with fresh node

            logger.info("[MCTSPlanner] Step %d — action=%s reward=%.3f",
                        step, actions[-1], self.reward_calc.calculate_reward(best.state))

        final_state = root.state
        reward = self.reward_calc.calculate_reward(final_state)

        return PlanningResult(
            success=final_state.main_answer is not None,
            final_state=final_state,
            action_sequence=actions,
            planning_time=time.time() - start,
            num_rollouts=num_rolls,
            final_reward=reward,
            metadata={"steps": len(actions),
                      "verification_passed": final_state.verification_passed},
        )

    def _make_node(self, state: State) -> ESVMCTSNode:
        return ESVMCTSNode(
            state=state,
            explore_action=self.explore,
            solve_action=self.solve,
            verify_action=self.verify,
            reward_calculator=self.reward_calc,
            config=self.config.mcts,
        )
