"""
State management for ESV-RAG.

A State tracks the main question, a list of sub-questions (with answers and statuses),
the reasoning history, and the current step.  The ESV MCTS operates on State objects;
each MCTS node stores one State.
"""

from __future__ import annotations

import copy
import json
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class QuestionStatus(str, Enum):
    UNANSWERED = "unanswered"
    ANSWERED = "answered"
    VERIFIED_PASS = "verified_pass"
    VERIFIED_FAIL = "verified_fail"


class ReasoningSkill(str, Enum):
    """Six reasoning skills used during Exploration."""
    LOGICAL = "logical"
    COUNTERFACTUAL = "counterfactual"
    PROBABILISTIC = "probabilistic"
    SOCIAL = "social"
    CONTEXTUAL = "contextual"
    ANALOGICAL = "analogical"


class QuestionItem(BaseModel):
    """A single sub-question generated during Exploration."""
    question: str = Field(..., description="Sub-question text")
    rationale: str = Field(default="", description="Why this sub-question was generated")
    skill: ReasoningSkill = Field(..., description="Reasoning skill applied")
    status: QuestionStatus = Field(default=QuestionStatus.UNANSWERED)
    answer: Optional[str] = Field(default=None)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    expected_insight: Optional[str] = Field(default=None)
    verification_result: Optional[bool] = Field(default=None)

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    def is_answered(self) -> bool:
        return self.status != QuestionStatus.UNANSWERED

    def is_verified(self) -> bool:
        return self.status in (QuestionStatus.VERIFIED_PASS, QuestionStatus.VERIFIED_FAIL)


class State(BaseModel):
    """
    Reasoning state for one question-answering episode.

    Fields mirror the MDP definition from §3.1 of the paper:
        s = (q, D, h, a_prev)
    where q is the main question, D is retrieved documents, h is history
    (sub-questions + answers), and a_prev is the last action taken.
    """
    main_question: str = Field(..., description="Primary question being answered")
    question_list: List[QuestionItem] = Field(default_factory=list)
    answer_list: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_documents: List[str] = Field(default_factory=list)
    reasoning_history: List[Dict[str, Any]] = Field(default_factory=list)
    current_step: int = Field(default=0)
    previous_action: Optional[str] = Field(default=None)
    main_answer: Optional[str] = Field(default=None)
    main_answer_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    verification_passed: bool = Field(default=False)
    skills_used: List[str] = Field(default_factory=list)

    def get_unanswered_questions(self) -> List[QuestionItem]:
        return [q for q in self.question_list if q.status == QuestionStatus.UNANSWERED]

    def get_answered_questions(self) -> List[QuestionItem]:
        return [q for q in self.question_list if q.is_answered()]

    def get_verified_questions(self) -> List[QuestionItem]:
        return [q for q in self.question_list if q.is_verified()]

    def get_main_answer(self) -> Optional[str]:
        return self.main_answer

    def add_question(self, item: QuestionItem) -> None:
        self.question_list.append(item)
        skill = item.skill.value
        if skill not in self.skills_used:
            self.skills_used.append(skill)

    def update_question_answer(self, question_text: str, answer: str,
                                confidence: float, status: QuestionStatus) -> bool:
        for q in self.question_list:
            if q.question == question_text:
                q.answer = answer
                q.confidence = confidence
                q.status = status
                return True
        return False

    def set_main_answer(self, answer: str, confidence: float) -> None:
        self.main_answer = answer
        self.main_answer_confidence = confidence

    def add_history(self, action: str, details: Dict[str, Any]) -> None:
        self.reasoning_history.append({"step": self.current_step,
                                        "action": action,
                                        "details": details})

    def copy(self) -> "State":
        return copy.deepcopy(self)

    def get_state_summary(self) -> Dict[str, Any]:
        return {
            "main_question": self.main_question,
            "total_questions": len(self.question_list),
            "unanswered": len(self.get_unanswered_questions()),
            "answered": len(self.get_answered_questions()),
            "verified": len(self.get_verified_questions()),
            "has_main_answer": self.main_answer is not None,
            "verification_passed": self.verification_passed,
            "current_step": self.current_step,
            "skills_used": self.skills_used,
        }

    def to_context_string(self) -> str:
        """Serialise state as a prompt-friendly context string."""
        parts: List[str] = [f"Main question: {self.main_question}"]
        if self.question_list:
            parts.append("Sub-questions and answers:")
            for q in self.question_list:
                status_label = q.status.value
                answer_text = q.answer or "(unanswered)"
                parts.append(f"  [{q.skill.value}] Q: {q.question}")
                parts.append(f"           A: {answer_text} [{status_label}]")
        if self.main_answer:
            parts.append(f"Current answer: {self.main_answer} "
                         f"(confidence={self.main_answer_confidence:.2f})")
        if self.retrieved_documents:
            parts.append(f"Retrieved documents: {len(self.retrieved_documents)}")
        return "\n".join(parts)

    def is_terminal(self, confidence_threshold: float = 0.8, max_depth: int = 10) -> bool:
        """Return True when the episode should end."""
        if self.current_step >= max_depth:
            return True
        if (self.main_answer is not None
                and self.verification_passed
                and self.main_answer_confidence >= confidence_threshold):
            return True
        return False
