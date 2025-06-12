"""
ESV Action Framework — Explore, Solve, Verify.

Each action transforms a State by interacting with the LLM Generator
and, optionally, the RetrieverClient.  All actions share the same
ActionResult dataclass and abstract base.

Reference line counts from the paper:
  explore.py  476 lines
  solve.py    600+ lines
  verify.py   855 lines
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import ActionConfig
from .state import QuestionItem, QuestionStatus, ReasoningSkill, State

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass
class ActionResult:
    """Return value for every ESV action."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    action_type: str = ""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Action(ABC):
    """Base class for Explore, Solve, and Verify actions."""

    def __init__(self, generator, retriever=None, config: Optional[ActionConfig] = None):
        self.generator = generator
        self.retriever = retriever
        self.config    = config or ActionConfig()
        self._stats    = {"calls": 0, "successes": 0, "failures": 0, "total_time": 0.0}

    def execute(self, *args, **kwargs) -> ActionResult:
        start = time.time()
        self._stats["calls"] += 1
        try:
            data = self._execute_impl(*args, **kwargs)
            elapsed = time.time() - start
            self._stats["successes"] += 1
            self._stats["total_time"] += elapsed
            return ActionResult(success=True, data=data,
                                execution_time=elapsed,
                                action_type=self.__class__.__name__)
        except Exception as exc:
            elapsed = time.time() - start
            self._stats["failures"] += 1
            logger.error("[%s] failed: %s", self.__class__.__name__, exc)
            return ActionResult(success=False, error=str(exc),
                                execution_time=elapsed,
                                action_type=self.__class__.__name__)

    @abstractmethod
    def _execute_impl(self, *args, **kwargs) -> Dict[str, Any]:
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _retrieve_context(self, query: str, top_k: int = 5) -> List[str]:
        if self.retriever is None:
            return []
        try:
            return self.retriever.retrieve(query, top_k=top_k)
        except Exception as exc:
            logger.warning("Retrieval failed: %s", exc)
            return []

    @staticmethod
    def _extract_json(text: str) -> Any:
        """Extract the first JSON array or object from *text*."""
        # Try direct parse first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        # Locate first JSON structure
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start = text.find(start_char)
            end   = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        return None

    @staticmethod
    def _parse_confidence(text: str) -> float:
        """Heuristically extract a [0,1] confidence from LLM text."""
        text_lower = text.lower()
        if "high" in text_lower:
            return 0.85
        if "medium" in text_lower:
            return 0.60
        if "low" in text_lower:
            return 0.35
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        if m:
            return min(1.0, float(m.group(1)) / 100)
        m = re.search(r"confidence[:\s]+([01]?\.\d+)", text_lower)
        if m:
            return float(m.group(1))
        return 0.50


# ===========================================================================
# Explore Action (E)
# ===========================================================================

class ExplorationAction(Action):
    """
    Generate diverse sub-questions using six reasoning skills.

    Given the main question (and optional context), the LLM produces
    3–5 sub-questions that collectively provide comprehensive coverage
    of the problem space.  The action updates *state* in-place.
    """

    SKILL_ORDER = [s.value for s in ReasoningSkill]

    def _execute_impl(self, state: State, context: str = "") -> Dict[str, Any]:
        from prompts.explore import (
            EXPLORATION_PROMPT,
            FOCUSED_EXPLORATION_PROMPT,
            ITERATIVE_EXPLORATION_PROMPT,
        )

        main_q = state.main_question
        prev_q = [q.question for q in state.question_list]
        new_insights = self._summarise_insights(state)

        # Choose prompt variant based on state
        if not state.question_list:
            prompt = EXPLORATION_PROMPT.format(
                main_question=main_q,
                context=context or "No additional context.",
            )
        elif prev_q:
            prompt = ITERATIVE_EXPLORATION_PROMPT.format(
                main_question=main_q,
                previous_questions=json.dumps(prev_q, indent=2),
                new_insights=new_insights,
                context=context or "No additional context.",
            )
        else:
            prompt = FOCUSED_EXPLORATION_PROMPT.format(
                main_question=main_q,
                focus_area="key aspects",
                context=context or "No additional context.",
            )

        raw = self.generator.generate(
            prompt, temperature=self.config.explore_temperature
        )

        questions = self._parse_questions(raw)
        questions = self._balance_skills(questions, prev_q)
        questions = questions[: self.config.max_explore_questions]

        # Retrieve context for each sub-question if retriever available
        retrieval_results = []
        for q_item in questions:
            docs = self._retrieve_context(q_item["question"],
                                          top_k=self.config.max_explore_questions)
            retrieval_results.append(docs)
            if docs:
                state.retrieved_documents.extend(docs)

        # Materialise QuestionItem objects and add to state
        added = []
        existing = {q.question.strip().lower() for q in state.question_list}
        for q_dict, docs in zip(questions, retrieval_results):
            if q_dict["question"].strip().lower() in existing:
                continue
            skill_str = q_dict.get("skill", "logical").lower()
            try:
                skill = ReasoningSkill(skill_str)
            except ValueError:
                skill = ReasoningSkill.LOGICAL

            item = QuestionItem(
                question=q_dict["question"],
                rationale=q_dict.get("rationale", ""),
                skill=skill,
                expected_insight=q_dict.get("expected_insight"),
            )
            state.add_question(item)
            added.append(item.model_dump())

        state.current_step += 1
        state.previous_action = "E"
        state.add_history("E", {"questions_added": len(added)})

        return {
            "questions_added": added,
            "retrieval_results": retrieval_results,
            "total_questions": len(state.question_list),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_questions(self, raw: str) -> List[Dict]:
        parsed = self._extract_json(raw)
        if isinstance(parsed, list):
            return [q for q in parsed if isinstance(q, dict) and "question" in q]
        if isinstance(parsed, dict) and "questions" in parsed:
            return parsed["questions"]
        # Fallback: extract lines that look like questions
        lines = [l.strip() for l in raw.split("\n") if "?" in l and len(l.strip()) > 10]
        return [{"question": l, "skill": "logical", "rationale": ""} for l in lines[:5]]

    def _balance_skills(self, questions: List[Dict], prev_qs: List[str]) -> List[Dict]:
        """Prefer questions that introduce underused reasoning skills."""
        used_skills: List[str] = []
        for q in prev_qs:
            pass  # track from state when available
        skill_count: Dict[str, int] = {s: 0 for s in self.SKILL_ORDER}
        for q in questions:
            skill_count[q.get("skill", "logical")] = (
                skill_count.get(q.get("skill", "logical"), 0) + 1
            )
        # Sort by ascending skill frequency (diverse first)
        questions.sort(key=lambda q: skill_count.get(q.get("skill", "logical"), 0))
        return questions

    @staticmethod
    def _summarise_insights(state: State) -> str:
        answered = state.get_answered_questions()
        if not answered:
            return "No insights yet."
        return "; ".join(
            f"{q.skill.value}: {q.answer[:80]}" for q in answered if q.answer
        )[:500]


# ===========================================================================
# Solve Action (S)
# ===========================================================================

class SolvingAction(Action):
    """
    Answer sub-questions and synthesise a main answer.

    Phase 1: for each unanswered QuestionItem, generate an answer.
    Phase 2: synthesise all sub-answers into a coherent main answer.
    """

    def _execute_impl(self, state: State) -> Dict[str, Any]:
        from prompts.solve import (
            SUB_QUESTION_ANSWER_PROMPT,
            MAIN_ANSWER_SYNTHESIS_PROMPT,
            REANSWER_SUB_QUESTION_PROMPT,
        )

        unanswered = state.get_unanswered_questions()
        answers_generated = []

        # ------ Phase 1: answer individual sub-questions ------
        if self.config.enable_parallel_solve and len(unanswered) > 1:
            answers_generated = self._answer_parallel(state, unanswered,
                                                       SUB_QUESTION_ANSWER_PROMPT,
                                                       REANSWER_SUB_QUESTION_PROMPT)
        else:
            for q_item in unanswered:
                result = self._answer_single(state, q_item,
                                             SUB_QUESTION_ANSWER_PROMPT)
                answers_generated.append(result)
                state.update_question_answer(
                    q_item.question, result["answer"],
                    result["confidence"], QuestionStatus.ANSWERED
                )

        # ------ Phase 2: synthesise main answer ------
        answered = state.get_answered_questions()
        if answered:
            sub_qa = json.dumps(
                [{"question": q.question, "skill": q.skill.value,
                  "answer": q.answer, "confidence": q.confidence}
                 for q in answered],
                indent=2
            )
            context = "\n".join(state.retrieved_documents[-10:])
            prompt  = MAIN_ANSWER_SYNTHESIS_PROMPT.format(
                main_question=state.main_question,
                sub_questions_and_answers=sub_qa,
                context=context or "No additional context.",
            )
            raw = self.generator.generate(prompt, temperature=self.config.solve_temperature)
            main_answer, confidence = self._parse_synthesis(raw)
            state.set_main_answer(main_answer, confidence)

        state.current_step += 1
        state.previous_action = "S"
        state.add_history("S", {"sub_answers": len(answers_generated),
                                  "main_answer_set": state.main_answer is not None})

        return {
            "sub_answers": answers_generated,
            "main_answer": state.main_answer,
            "confidence": state.main_answer_confidence,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _answer_single(self, state: State, q_item: QuestionItem,
                       prompt_template: str) -> Dict:
        docs = self._retrieve_context(q_item.question)
        context = "\n".join(docs) if docs else "No retrieved context."
        prompt  = prompt_template.format(
            main_question=state.main_question,
            sub_question=q_item.question,
            reasoning_skill=q_item.skill.value,
            rationale=q_item.rationale or "Explore this aspect.",
            context=context,
        )
        raw = self.generator.generate(prompt, temperature=self.config.solve_temperature)
        answer, confidence = self._parse_answer(raw)
        return {"question": q_item.question, "answer": answer, "confidence": confidence}

    def _answer_parallel(self, state: State, unanswered: List[QuestionItem],
                         prompt_template: str, reanswer_template: str) -> List[Dict]:
        results = []
        with ThreadPoolExecutor(max_workers=self.config.max_solve_workers) as pool:
            futures = {
                pool.submit(self._answer_single, state, q, prompt_template): q
                for q in unanswered
            }
            for future in as_completed(futures):
                q_item = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    state.update_question_answer(
                        q_item.question, result["answer"],
                        result["confidence"], QuestionStatus.ANSWERED
                    )
                except Exception as exc:
                    logger.error("Parallel solve failed for '%s': %s", q_item.question, exc)
        return results

    @staticmethod
    def _parse_answer(raw: str) -> tuple[str, float]:
        """Extract ANSWER and confidence from LLM response."""
        answer = ""
        m = re.search(r"ANSWER:\s*(.+?)(?:\n|REASONING:|CONFIDENCE:|$)", raw,
                      re.DOTALL | re.IGNORECASE)
        if m:
            answer = m.group(1).strip()

        confidence = Action._parse_confidence(raw)  # noqa: SLF001

        if not answer:
            answer = raw.strip()[:500]

        return answer, confidence

    @staticmethod
    def _parse_synthesis(raw: str) -> tuple[str, float]:
        """Extract SYNTHESIS or CONCLUSION from synthesis response."""
        synthesis = ""
        for pattern in [r"CONCLUSION:\s*(.+?)(?:\n\n|$)",
                        r"SYNTHESIS:\s*(.+?)(?:\n\n|KEY INSIGHTS:|$)"]:
            m = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
            if m:
                synthesis = m.group(1).strip()
                break

        confidence = Action._parse_confidence(raw)  # noqa: SLF001

        if not synthesis:
            synthesis = raw.strip()[:800]

        return synthesis, confidence


# ===========================================================================
# Verify Action (V)
# ===========================================================================

class VerificationAction(Action):
    """
    Adversarial self-verification and self-correction.

    1. Generate structured yes/no verification questions.
    2. Answer each question — PASS or FAIL.
    3. If any fail → trigger self-correction and re-solve.
    4. Update state verification flags.
    """

    def _execute_impl(self, state: State) -> Dict[str, Any]:
        from prompts.verify import (
            VERIFICATION_QUESTION_PROMPT,
            VERIFICATION_ANSWER_PROMPT,
            SELF_CORRECTION_PROMPT,
        )

        if state.main_answer is None:
            return {"verification_passed": False,
                    "reason": "No main answer to verify."}

        answered = state.get_answered_questions()
        sub_qa   = json.dumps(
            [{"question": q.question, "answer": q.answer}
             for q in answered],
            indent=2
        )

        attempt  = 0
        passed   = False
        issues: List[str] = []

        while attempt <= self.config.max_correction_attempts:
            # Phase 1: generate verification questions
            prompt_vq = VERIFICATION_QUESTION_PROMPT.format(
                main_question=state.main_question,
                proposed_answer=state.main_answer,
                sub_questions_and_answers=sub_qa,
            )
            raw_vq = self.generator.generate(prompt_vq,
                                              temperature=self.config.verify_temperature)
            v_questions = self._parse_verification_questions(raw_vq)

            # Phase 2: answer each verification question
            results: List[Dict] = []
            context = "\n".join(state.retrieved_documents[-10:])
            for vq in v_questions:
                prompt_va = VERIFICATION_ANSWER_PROMPT.format(
                    main_question=state.main_question,
                    proposed_answer=state.main_answer,
                    verification_question=vq.get("verification_question", ""),
                    verification_type=vq.get("verification_type", ""),
                    rationale=vq.get("rationale", ""),
                    context=context or "No context.",
                )
                raw_va = self.generator.generate(
                    prompt_va, temperature=self.config.verify_temperature
                )
                vr = self._parse_verification_result(raw_va)
                results.append({**vq, "result": vr["passed"],
                                 "reasoning": vr["reasoning"]})

            failed = [r for r in results if not r["result"]]
            issues = [r.get("reasoning", "") for r in failed]
            passed = len(failed) == 0

            if passed or not self.config.enable_self_correction:
                break

            # Phase 3: self-correction
            attempt += 1
            if attempt > self.config.max_correction_attempts:
                break

            corrected = self._self_correct(state, issues, SELF_CORRECTION_PROMPT,
                                            sub_qa, context)
            if corrected:
                state.set_main_answer(corrected["answer"], corrected["confidence"])

        # Update question statuses
        new_status = QuestionStatus.VERIFIED_PASS if passed else QuestionStatus.VERIFIED_FAIL
        for q in state.question_list:
            if q.is_answered():
                q.status = new_status

        state.verification_passed = passed
        state.current_step += 1
        state.previous_action = "V"
        state.add_history("V", {"passed": passed, "attempts": attempt + 1,
                                  "issues": issues})

        return {
            "verification_passed": passed,
            "correction_attempts": attempt,
            "issues": issues,
            "final_answer": state.main_answer,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_verification_questions(self, raw: str) -> List[Dict]:
        parsed = self._extract_json(raw)
        if isinstance(parsed, list):
            return parsed[: self.config.max_verify_questions]
        return [{"verification_question": raw.strip()[:200],
                 "verification_type": "general",
                 "rationale": ""}]

    @staticmethod
    def _parse_verification_result(raw: str) -> Dict:
        raw_upper = raw.upper()
        # Explicit YES/NO
        if "VERIFICATION RESULT: YES" in raw_upper or raw_upper.strip().startswith("YES"):
            passed = True
        elif "VERIFICATION RESULT: NO" in raw_upper or raw_upper.strip().startswith("NO"):
            passed = False
        else:
            # Count positive vs negative indicators
            pos = sum(raw_upper.count(w) for w in ["YES", "PASS", "CORRECT", "VERIFIED"])
            neg = sum(raw_upper.count(w) for w in ["NO", "FAIL", "INCORRECT", "INVALID"])
            passed = pos >= neg

        reasoning_m = re.search(r"REASONING:\s*(.+?)(?:\n\n|SPECIFIC|EVIDENCE|$)", raw,
                                 re.DOTALL | re.IGNORECASE)
        reasoning = reasoning_m.group(1).strip() if reasoning_m else ""
        return {"passed": passed, "reasoning": reasoning}

    def _self_correct(self, state: State, issues: List[str],
                      prompt_template: str, sub_qa: str, context: str) -> Optional[Dict]:
        from prompts.solve import MAIN_ANSWER_SYNTHESIS_PROMPT
        issues_text = "\n".join(f"- {i}" for i in issues if i)
        prompt = prompt_template.format(
            main_question=state.main_question,
            original_answer=state.main_answer,
            failed_verification=issues_text,
            verification_issues=issues_text,
            context=context or "No context.",
        )
        raw = self.generator.generate(prompt, temperature=self.config.solve_temperature)

        m = re.search(r"CORRECTED ANSWER:\s*(.+?)(?:\n\n|EXPLANATION:|CONFIDENCE:|$)",
                      raw, re.DOTALL | re.IGNORECASE)
        if m:
            answer = m.group(1).strip()
        else:
            answer = raw.strip()[:800]

        confidence = self._parse_confidence(raw)
        return {"answer": answer, "confidence": confidence}
