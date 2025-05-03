"""
Prompt templates for the Exploration (E) action.

Three variants are provided:
  EXPLORATION_PROMPT          — fresh exploration (no prior questions)
  FOCUSED_EXPLORATION_PROMPT  — targeted on a specific area
  ITERATIVE_EXPLORATION_PROMPT — builds on previous exploration rounds
"""

EXPLORATION_PROMPT = """You are an expert reasoning agent tasked with exploring a complex question by generating diverse sub-questions.

MAIN QUESTION: {main_question}

CONTEXT: {context}

TASK: Generate 3-5 diverse sub-questions that will help answer the main question. Each sub-question should:
1. Use a different reasoning skill (logical, counterfactual, probabilistic, social, contextual, or analogical)
2. Focus on a specific aspect or angle of the main question
3. Be specific enough to be answerable
4. Build upon each other to create a comprehensive understanding

REASONING SKILLS TO USE:
- LOGICAL: Causal relationships, logical consistency, deductive reasoning
- COUNTERFACTUAL: Alternative scenarios, what-if analysis, sensitivity testing
- PROBABILISTIC: Uncertainty quantification, statistical relationships, confidence levels
- SOCIAL: Human factors, institutional dynamics, behavioral considerations
- CONTEXTUAL: Historical development, situational factors, environmental context
- ANALOGICAL: Cross-domain parallels, transferable principles, pattern matching

OUTPUT FORMAT:
Return a JSON array of objects, each with:
- "question": The sub-question text
- "rationale": Why this question helps answer the main question
- "skill": The reasoning skill being applied (one of the six skills above)
- "expected_insight": What type of insight this question should provide

EXAMPLE OUTPUT:
[
    {{
        "question": "What are the direct causes that led to this situation?",
        "rationale": "Understanding root causes is essential for comprehensive analysis",
        "skill": "logical",
        "expected_insight": "Causal relationships and logical connections"
    }},
    {{
        "question": "What would happen if the key factors were different?",
        "rationale": "Exploring alternatives helps identify critical dependencies",
        "skill": "counterfactual",
        "expected_insight": "Sensitivity to key variables and alternative outcomes"
    }}
]

Generate diverse, high-quality sub-questions that will provide comprehensive coverage of the main question."""


FOCUSED_EXPLORATION_PROMPT = """You are an expert reasoning agent exploring a specific aspect of a complex question.

MAIN QUESTION: {main_question}
FOCUS AREA: {focus_area}
CONTEXT: {context}

TASK: Generate 2-3 focused sub-questions that specifically address the {focus_area} aspect of the main question.

REQUIREMENTS:
- Each sub-question should use a different reasoning skill
- Questions should be specific and answerable
- Focus on the designated area while maintaining relevance to the main question
- Provide clear rationale for how each question contributes to understanding

OUTPUT FORMAT:
Return a JSON array of objects with:
- "question": The focused sub-question
- "rationale": Why this question addresses the focus area
- "skill": The reasoning skill applied
- "expected_insight": Specific insight expected from this question

Generate focused, high-quality sub-questions for the specified area."""


ITERATIVE_EXPLORATION_PROMPT = """You are an expert reasoning agent refining exploration based on previous findings.

MAIN QUESTION: {main_question}
PREVIOUS QUESTIONS: {previous_questions}
NEW INSIGHTS: {new_insights}
CONTEXT: {context}

TASK: Generate 2-3 new sub-questions that:
1. Build upon the insights from previous questions
2. Address gaps or areas that need deeper exploration
3. Use reasoning skills not yet applied or apply them in new ways
4. Move toward a comprehensive understanding of the main question

REQUIREMENTS:
- Avoid duplicating previous questions
- Focus on unexplored angles or deeper analysis
- Use reasoning skills strategically to fill knowledge gaps
- Ensure questions are specific and answerable

OUTPUT FORMAT:
Return a JSON array of objects with:
- "question": The new sub-question
- "rationale": Why this question builds on previous insights
- "skill": The reasoning skill applied
- "expected_insight": How this question advances understanding
- "builds_on": Which previous questions or insights this builds upon

Generate strategic, gap-filling sub-questions that advance the exploration."""
