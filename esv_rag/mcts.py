"""
Monte Carlo Tree Search for ESV-RAG.

Two layers:
  backbone  — generic MCTS algorithm (Node ABC + MCTS class)
  ESV layer — ESVMCTSNode (State-carrying node) + ESVMCTS (ESV-specific search)

The PUCT selection formula follows §3.2 of the paper:
  UCB1(n) = Q(n) + c_puct · √(N(parent)) / (1 + N(n))
"""

from __future__ import annotations

import copy
import logging
import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Set

from .config import MCTSConfig
from .state import State, QuestionStatus
from .rewards import ESVRewardCalculator

logger = logging.getLogger(__name__)


# ===========================================================================
# Generic MCTS backbone
# ===========================================================================

class Node(ABC):
    """Abstract MCTS node."""

    @abstractmethod
    def find_children(self) -> List["Node"]:
        """Return all successor nodes."""

    @abstractmethod
    def is_terminal(self) -> bool:
        """True if no further expansion is meaningful."""

    @abstractmethod
    def reward(self) -> float:
        """Reward at a terminal node."""

    @abstractmethod
    def __hash__(self) -> int: ...

    @abstractmethod
    def __eq__(self, other) -> bool: ...


class MCTS:
    """
    Upper-Confidence Trees (UCT / PUCT) search.

    Q[n]  = total reward accumulated through node n
    N[n]  = visit count
    """

    def __init__(self, exploration_weight: float = 1.414, verbose: bool = False):
        self.Q: Dict[Node, float] = defaultdict(float)
        self.N: Dict[Node, int]   = defaultdict(int)
        self.children: Dict[Node, List[Node]] = {}
        self.c = exploration_weight
        self.verbose = verbose

    def choose(self, node: Node) -> Node:
        """Return the best child of *node* (exploitation only)."""
        if node.is_terminal():
            raise ValueError("Cannot choose child of terminal node")
        if node not in self.children:
            return self._random_child(node)
        best = max(self.children[node], key=lambda n: self._exploitation_score(n))
        return best

    def do_rollout(self, node: Node) -> None:
        """One Selection → Expansion → Simulation → Backpropagation cycle."""
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    # ------------------------------------------------------------------
    # Core phases
    # ------------------------------------------------------------------

    def _select(self, node: Node) -> List[Node]:
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                return path
            unexplored = [c for c in self.children[node] if c not in self.children]
            if unexplored:
                chosen = random.choice(unexplored)
                path.append(chosen)
                return path
            node = self._uct_select(node)

    def _expand(self, node: Node) -> None:
        if node in self.children or node.is_terminal():
            return
        self.children[node] = node.find_children()

    def _simulate(self, node: Node) -> float:
        current = node
        steps   = 0
        while not current.is_terminal() and steps < 20:
            children = current.find_children()
            if not children:
                break
            current = random.choice(children)
            steps  += 1
        return current.reward()

    def _backpropagate(self, path: List[Node], reward: float) -> None:
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    # ------------------------------------------------------------------
    # UCT / PUCT selection
    # ------------------------------------------------------------------

    def _uct_select(self, node: Node) -> Node:
        log_n = math.log(self.N[node] + 1)
        return max(self.children[node], key=lambda n: self._uct_score(n, log_n))

    def _uct_score(self, node: Node, log_n_parent: float) -> float:
        if self.N[node] == 0:
            return float("inf")
        q = self.Q[node] / self.N[node]
        u = self.c * math.sqrt(log_n_parent / self.N[node])
        return q + u

    def _exploitation_score(self, node: Node) -> float:
        if self.N[node] == 0:
            return float("-inf")
        return self.Q[node] / self.N[node]

    @staticmethod
    def _random_child(node: Node) -> Node:
        children = node.find_children()
        if not children:
            return node
        return random.choice(children)


# ===========================================================================
# ESV MCTS node
# ===========================================================================

class ESVMCTSNode(Node):
    """
    MCTS node wrapping an ESV reasoning State.

    Children are generated by applying the three ESV actions (E, S, V)
    to the current state.  The reward is computed by ESVRewardCalculator.
    """

    ESV_ACTIONS = ["E", "S", "V"]

    def __init__(self,
                 state: State,
                 parent_action: Optional[str] = None,
                 explore_action=None,
                 solve_action=None,
                 verify_action=None,
                 reward_calculator: Optional[ESVRewardCalculator] = None,
                 config: Optional[MCTSConfig] = None):
        self.state          = state
        self.parent_action  = parent_action
        self.explore_action = explore_action
        self.solve_action   = solve_action
        self.verify_action  = verify_action
        self.reward_calc    = reward_calculator or ESVRewardCalculator()
        self.config         = config or MCTSConfig()
        self.action_history: List[str] = []
        self._hash          = hash(state.main_question + str(state.current_step)
                                   + str(len(state.question_list)))

    # ------------------------------------------------------------------
    # Node interface
    # ------------------------------------------------------------------

    def find_children(self) -> List["ESVMCTSNode"]:
        """Generate one child per legal action."""
        children = []
        for action in self._legal_actions():
            child_state = self._apply_action(action)
            child = ESVMCTSNode(
                state=child_state,
                parent_action=action,
                explore_action=self.explore_action,
                solve_action=self.solve_action,
                verify_action=self.verify_action,
                reward_calculator=self.reward_calc,
                config=self.config,
            )
            child.action_history = self.action_history + [action]
            children.append(child)
        return children

    def is_terminal(self) -> bool:
        return self.state.is_terminal(
            confidence_threshold=0.8,
            max_depth=self.config.max_depth,
        )

    def reward(self) -> float:
        return self.reward_calc.calculate_reward(self.state)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        if not isinstance(other, ESVMCTSNode):
            return False
        return (self.state.main_question == other.state.main_question
                and self.state.current_step == other.state.current_step
                and len(self.state.question_list) == len(other.state.question_list)
                and self.parent_action == other.parent_action)

    # ------------------------------------------------------------------
    # Action logic
    # ------------------------------------------------------------------

    def _legal_actions(self) -> List[str]:
        """Heuristic constraints from §3.2."""
        actions = list(self.ESV_ACTIONS)
        hist    = self.action_history

        # Enforce: Verify must follow Solve
        if "V" in actions and (not hist or hist[-1] != "S"):
            actions.remove("V")

        # Cap consecutive Explores at 3
        if hist and all(a == "E" for a in hist[-3:]):
            if "E" in actions:
                actions.remove("E")

        # Must have questions to solve
        if not self.state.question_list and "S" in actions:
            actions.remove("S")

        return actions if actions else ["E"]

    def _apply_action(self, action: str) -> State:
        """Apply action to a deep copy of the state."""
        new_state = self.state.copy()
        try:
            if action == "E" and self.explore_action is not None:
                result = self.explore_action.execute(new_state)
                if not result.success:
                    logger.warning("Explore failed: %s", result.error)
            elif action == "S" and self.solve_action is not None:
                result = self.solve_action.execute(new_state)
                if not result.success:
                    logger.warning("Solve failed: %s", result.error)
            elif action == "V" and self.verify_action is not None:
                result = self.verify_action.execute(new_state)
                if not result.success:
                    logger.warning("Verify failed: %s", result.error)
        except Exception as exc:
            logger.error("Action %s raised: %s", action, exc)

        return new_state


# ===========================================================================
# ESV-specific MCTS driver
# ===========================================================================

class ESVMCTS(MCTS):
    """
    MCTS configured for the ESV action space (E, S, V).

    Runs *num_simulations* rollouts from the root node, then returns
    the best child state.
    """

    def __init__(self, config: Optional[MCTSConfig] = None):
        cfg = config or MCTSConfig()
        super().__init__(exploration_weight=cfg.exploration_constant,
                         verbose=cfg.verbose)
        self.config = cfg

    def search(self, root: ESVMCTSNode) -> ESVMCTSNode:
        """
        Run MCTS for *num_simulations* iterations from *root*.
        Returns the best child of root.
        """
        for i in range(self.config.num_simulations):
            self.do_rollout(root)
            if self.config.verbose and (i + 1) % 10 == 0:
                logger.debug("Rollout %d/%d — Q(root)=%.3f N(root)=%d",
                             i + 1, self.config.num_simulations,
                             self.Q[root], self.N[root])

        if root not in self.children or not self.children[root]:
            return root

        best = self.choose(root)
        logger.info("MCTS selected action=%s after %d simulations",
                    best.parent_action, self.config.num_simulations)
        return best

    def _simulate(self, node: Node) -> float:
        """Rollout using ESV heuristic policy."""
        current = node
        steps   = 0
        while not current.is_terminal() and steps < self.config.max_simulation_steps:
            action  = self._default_policy(current)
            new_state = current._apply_action(action)    # noqa: SLF001
            child   = ESVMCTSNode(
                state=new_state,
                parent_action=action,
                explore_action=current.explore_action,
                solve_action=current.solve_action,
                verify_action=current.verify_action,
                reward_calculator=current.reward_calc,
                config=current.config,
            )
            child.action_history = current.action_history + [action]
            current = child
            steps  += 1
        return current.reward()

    @staticmethod
    def _default_policy(node: ESVMCTSNode) -> str:
        """Simple heuristic: E → S → V → repeat."""
        state = node.state
        if not state.question_list:
            return "E"
        if state.get_unanswered_questions():
            return "S"
        if (state.get_answered_questions()
                and not any(q.is_verified() for q in state.question_list)):
            return "V"
        return "E"
