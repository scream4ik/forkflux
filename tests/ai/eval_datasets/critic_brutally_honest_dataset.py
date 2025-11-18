from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

critic_brutally_honest_dataset = [
    {
        "id": "dissect_flawed_reasoning",
        "description": "Checks if the Critic, in brutally honest mode, dissects a weak business plan, exposes flawed assumptions, and provides a direct, unfiltered action plan without any comforting language.",
        "input": """
        My business plan is to recreate the 'Yo' app. It was popular once, so it's a proven concept.
        We'll just add a nicer UI and market it on TikTok. I think we can get 1 million users in three months.
        The original failed because of bad timing, but now is the right moment.
        """,
        "metric": GEval(
            name="Brutal Honesty Critique Adherence",
            criteria="The response must apply the Critic's analytical methodology but with a brutally honest and direct tone. It must dissect the flaws in the user's plan without softening the truth and provide a strategic, unfiltered path forward.",
            evaluation_steps=[
                "1. Check 'CRITICAL FLAWS': verify it identifies the core failure (lack of value) and attacks 'proven concept'.",
                "2. Check 'CRITICAL FLAWS': verify it exposes the 'nicer UI' fallacy as a tactic, not strategy.",
                "3. Check 'CHAT RESPONSE': Verify the tone is direct, challenging, and avoids standard AI politeness (like 'That's an interesting idea').",
                "4. Check 'SUGGESTIONS': Verify it demands a pivot or killing the idea.",
            ],
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.8,
        ),
    },
]
