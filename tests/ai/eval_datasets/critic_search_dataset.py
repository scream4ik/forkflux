from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

critic_search_dataset = [
    {
        "id": "fact_check_requirement",
        "description": "Checks if the Critic demands citations or identifies the need for verification when presented with a suspicious factual claim.",
        "input": "Apple released the iPhone 25 yesterday and it runs on nuclear power.",
        "metric": GEval(
            name="Skepticism & Verification",
            criteria="The Critic must identify this as a highly suspicious claim needing verification.",
            evaluation_steps=[
                "1. Check 'CRITICAL FLAWS': Does it flag the claim about 'iPhone 25' or 'nuclear power' as FACTUALLY SUSPICIOUS or FALSE?",
                "2. Check 'SUGGESTED FIXES': Does it suggest removing the claim or providing a source?",
                "3. Check 'CHAT RESPONSE': Is the tone skeptical?",
            ],
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.8,
        ),
    },
    {
        "id": "citation_formatting_check",
        "description": "Checks if the output structure includes a 'CITATIONS/SOURCES' section (even if empty due to test isolation) and follows the logic.",
        "input": "The sky is blue.",
        "metric": GEval(
            name="Structure Adherence",
            criteria="The output must follow the defined reporting structure.",
            evaluation_steps=[
                "1. Ensure the output contains a 'CITATIONS/SOURCES' section (as formatted by the test suite).",
                "2. Since the claim is common knowledge, it is acceptable for sources to be empty, but the section must exist.",
            ],
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=1.0,
        ),
    },
]
