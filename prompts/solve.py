"""
Prompt templates for the Solving (S) action.

  SUB_QUESTION_ANSWER_PROMPT       — answer one sub-question
  MAIN_ANSWER_SYNTHESIS_PROMPT     — synthesise final answer from sub-answers
  REANSWER_SUB_QUESTION_PROMPT     — improve answer with additional context
  ANSWER_CONFIDENCE_ASSESSMENT_PROMPT — assess answer confidence
"""

SUB_QUESTION_ANSWER_PROMPT = """You are an expert reasoning agent tasked with answering a specific sub-question to help solve a larger problem.

MAIN QUESTION: {main_question}
SUB-QUESTION: {sub_question}
REASONING SKILL: {reasoning_skill}
RATIONALE: {rationale}

CONTEXT: {context}

TASK: Provide a clear, accurate, and well-reasoned answer to the sub-question. Your answer should:
1. Be directly relevant to the sub-question asked
2. Demonstrate the reasoning skill being applied
3. Be specific and factual
4. Include reasoning steps that show your thinking process
5. Be concise but comprehensive

REASONING APPROACH:
- LOGICAL: Focus on causal relationships and logical consistency
- COUNTERFACTUAL: Consider alternative scenarios and what-if situations
- PROBABILISTIC: Address uncertainty and statistical relationships
- SOCIAL: Consider human factors and institutional dynamics
- CONTEXTUAL: Include historical and situational context
- ANALOGICAL: Draw parallels from similar situations or domains

OUTPUT FORMAT:

ANSWER: [Your direct answer to the sub-question]

REASONING: [Step-by-step explanation of your reasoning process]

CONFIDENCE: [High/Medium/Low - based on the certainty of your answer]

EVIDENCE: [Key facts or reasoning that support your answer]

Generate a high-quality, well-reasoned answer that demonstrates the specified reasoning skill."""


MAIN_ANSWER_SYNTHESIS_PROMPT = """You are an expert reasoning agent tasked with synthesizing a comprehensive answer to the main question based on multiple sub-question answers.

MAIN QUESTION: {main_question}

SUB-QUESTIONS AND ANSWERS:
{sub_questions_and_answers}

CONTEXT: {context}

TASK: Synthesize a comprehensive, well-structured answer to the main question that:
1. Integrates insights from all sub-question answers
2. Addresses the main question directly and completely
3. Shows how the different reasoning skills contributed to understanding
4. Provides a coherent narrative that connects all the pieces
5. Acknowledges any remaining uncertainties or gaps

SYNTHESIS APPROACH:
- Identify the key insights from each sub-question
- Show how different reasoning skills revealed different aspects
- Connect the insights into a coherent understanding
- Address any contradictions or gaps between sub-answers
- Provide a final, synthesized conclusion

OUTPUT FORMAT:

SYNTHESIS: [Your comprehensive answer to the main question]

KEY INSIGHTS: [List of main insights from the sub-questions]

REASONING INTEGRATION: [How the different reasoning skills contributed]

CONCLUSION: [Final, synthesized conclusion]

CONFIDENCE: [High/Medium/Low - based on the completeness and consistency of sub-answers]

Generate a comprehensive, well-integrated answer that demonstrates how the exploration process led to understanding."""


REANSWER_SUB_QUESTION_PROMPT = """You are an expert reasoning agent re-answering a sub-question with additional context and insights.

MAIN QUESTION: {main_question}
SUB-QUESTION: {sub_question}
REASONING SKILL: {reasoning_skill}
ORIGINAL ANSWER: {original_answer}
ADDITIONAL CONTEXT: {additional_context}

TASK: Provide an improved answer to the sub-question that incorporates the additional context. Your answer should:
1. Build upon the original answer
2. Integrate the new context and insights
3. Maintain the focus on the specific reasoning skill
4. Show how the additional information changes or enhances understanding
5. Provide a more complete and accurate response

OUTPUT FORMAT:

IMPROVED ANSWER: [Your enhanced answer incorporating new context]

CHANGES MADE: [What changed from the original answer and why]

REASONING: [How the additional context influenced your reasoning]

CONFIDENCE: [Updated confidence level based on new information]

Generate an improved answer that demonstrates enhanced understanding through additional context."""


ANSWER_CONFIDENCE_ASSESSMENT_PROMPT = """You are an expert evaluator assessing the confidence level of an answer to a sub-question.

MAIN QUESTION: {main_question}
SUB-QUESTION: {sub_question}
REASONING SKILL: {reasoning_skill}
ANSWER: {answer}
CONTEXT: {context}

TASK: Assess the confidence level of this answer based on:
1. Completeness of the answer
2. Quality of reasoning provided
3. Specificity and relevance
4. Evidence and support provided
5. Alignment with the reasoning skill being applied

CONFIDENCE CRITERIA:
- HIGH: Complete, well-reasoned, specific, well-supported answer
- MEDIUM: Generally good but some gaps or uncertainties
- LOW: Incomplete, unclear reasoning, or insufficient support

OUTPUT FORMAT:

CONFIDENCE LEVEL: [High/Medium/Low]

REASONING: [Explanation of why you assigned this confidence level]

STRENGTHS: [What makes this answer strong]

AREAS FOR IMPROVEMENT: [What could make this answer better]

OVERALL ASSESSMENT: [Brief summary of your evaluation]

Provide a thorough and objective confidence assessment."""
