"""
Prompt templates for the Verification (V) action.

  VERIFICATION_QUESTION_PROMPT   — generate adversarial verification questions
  VERIFICATION_ANSWER_PROMPT     — answer each verification question (YES/NO)
  SELF_CORRECTION_PROMPT         — self-correct answer after failed verification
  COMPREHENSIVE_VERIFICATION_PROMPT — full one-shot verification assessment
"""

VERIFICATION_QUESTION_PROMPT = """You are an expert verification agent tasked with generating critical verification questions to test the quality and reliability of an answer.

MAIN QUESTION: {main_question}
PROPOSED ANSWER: {proposed_answer}
SUB-QUESTIONS AND ANSWERS: {sub_questions_and_answers}

TASK: Generate 3-5 critical verification questions that will test the quality, accuracy, and completeness of the proposed answer. These questions should act as "red-teaming" to identify potential issues.

VERIFICATION APPROACH:
- LOGICAL CONSISTENCY: Test for internal contradictions or logical flaws
- FACTUAL ACCURACY: Verify specific claims and assertions
- COMPLETENESS: Check if the answer addresses all aspects of the question
- REASONING QUALITY: Assess the soundness of the reasoning process
- EVIDENCE SUPPORT: Verify that claims are well-supported

VERIFICATION QUESTION TYPES:
1. Contradiction Detection: Questions that might reveal internal inconsistencies
2. Edge Case Testing: Questions about unusual or extreme scenarios
3. Assumption Validation: Questions that test underlying assumptions
4. Evidence Verification: Questions that require specific evidence or examples
5. Alternative Explanation: Questions that explore competing explanations

OUTPUT FORMAT:
Return a JSON array of objects, each with:
- "verification_question": The critical verification question
- "verification_type": Type (contradiction, edge_case, assumption, evidence, alternative)
- "rationale": Why this verification question is important
- "expected_outcome": What a good answer should demonstrate
- "criticality": High/Medium/Low

EXAMPLE OUTPUT:
[
    {{
        "verification_question": "Does this answer contradict any of the sub-question answers?",
        "verification_type": "contradiction",
        "rationale": "Internal consistency is essential for answer reliability",
        "expected_outcome": "No contradictions found, all parts align",
        "criticality": "High"
    }},
    {{
        "verification_question": "What evidence supports the key claims in this answer?",
        "verification_type": "evidence",
        "rationale": "Claims without evidence are not reliable",
        "expected_outcome": "Specific, verifiable evidence provided",
        "criticality": "High"
    }}
]

Generate critical, high-quality verification questions that will thoroughly test the proposed answer."""


VERIFICATION_ANSWER_PROMPT = """You are an expert verification agent answering a critical verification question about an answer's quality.

MAIN QUESTION: {main_question}
PROPOSED ANSWER: {proposed_answer}
VERIFICATION QUESTION: {verification_question}
VERIFICATION TYPE: {verification_type}
RATIONALE: {rationale}

CONTEXT: {context}

TASK: Answer the verification question with a clear YES or NO, followed by your reasoning. Your answer should:
1. Be definitive (YES or NO)
2. Include clear reasoning for your assessment
3. Reference specific parts of the proposed answer
4. Be objective and evidence-based

VERIFICATION CRITERIA:
- YES: The proposed answer passes this verification test
- NO: The proposed answer fails this verification test

OUTPUT FORMAT:

VERIFICATION RESULT: [YES/NO]

REASONING: [Clear explanation of why you answered YES or NO]

SPECIFIC ISSUES: [If NO, what specific problems were found]

EVIDENCE: [What evidence supports your verification result]

Generate a clear, objective verification assessment."""


SELF_CORRECTION_PROMPT = """You are an expert reasoning agent tasked with self-correcting an answer that failed verification.

MAIN QUESTION: {main_question}
ORIGINAL ANSWER: {original_answer}
FAILED VERIFICATION: {failed_verification}
VERIFICATION ISSUES: {verification_issues}

CONTEXT: {context}

TASK: Generate a corrected version of the answer that addresses the verification failures. Your corrected answer should:
1. Fix the specific issues identified in verification
2. Maintain the strengths of the original answer
3. Provide better evidence and reasoning
4. Address any contradictions or gaps
5. Demonstrate improved quality and reliability

CORRECTION APPROACH:
1. Identify the root causes of verification failures
2. Revise the answer to address these issues
3. Strengthen weak reasoning or evidence
4. Ensure internal consistency
5. Validate that the correction resolves the verification problems

OUTPUT FORMAT:

CORRECTED ANSWER: [Your corrected answer that fixes the identified issues]

CHANGES MADE: [What was changed and why]

EXPLANATION: [How the correction addresses the verification failures]

CONFIDENCE: [Updated confidence level after correction]

Generate a corrected answer that demonstrably resolves the verification failures."""


COMPREHENSIVE_VERIFICATION_PROMPT = """You are an expert verification agent conducting a comprehensive quality assessment of an answer.

MAIN QUESTION: {main_question}
PROPOSED ANSWER: {proposed_answer}
REASONING TRACE: {reasoning_trace}

CONTEXT: {context}

TASK: Conduct a comprehensive verification of the proposed answer across four dimensions:

1. FACTUAL CONSISTENCY (weight 0.4): Does the answer align with retrieved evidence?
2. LOGICAL COHERENCE (weight 0.3): Is the reasoning chain valid and consistent?
3. QUESTION COVERAGE (weight 0.2): Does the answer address all parts of the question?
4. SELF-CONSISTENCY (weight 0.1): Does the answer agree with the sub-question answers?

For each dimension, assign a score from 0.0 to 1.0.

OUTPUT FORMAT:

FACTUAL CONSISTENCY SCORE: [0.0-1.0]
FACTUAL REASONING: [Explanation]

LOGICAL COHERENCE SCORE: [0.0-1.0]
LOGICAL REASONING: [Explanation]

QUESTION COVERAGE SCORE: [0.0-1.0]
COVERAGE REASONING: [Explanation]

SELF-CONSISTENCY SCORE: [0.0-1.0]
CONSISTENCY REASONING: [Explanation]

OVERALL VERIFICATION SCORE: [Weighted average]

VERIFICATION PASSED: [YES if overall score >= 0.75, else NO]

ISSUES FOUND: [List any specific problems]

RECOMMENDATION: [ACCEPT / CORRECT / REJECT with brief justification]

Provide a thorough, evidence-based comprehensive verification."""
