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
                "1. Check if the response directly attacks the core flawed assumption ('proven concept') and explains why the original 'Yo' app truly failed (e.g., no monetization, no long-term value).",
                "2. Evaluate if the response identifies and exposes the user's underestimation of effort (e.g., stating that 'a nicer UI' and 'TikTok marketing' are tactics, not a strategy).",
                "3. Verify that the tone is direct and unfiltered, completely avoiding comforting or validating phrases. It should use challenging language.",
                "4. Assess if the response provides a precise and demanding action plan, such as 'Define a viable business model or kill this idea immediately'.",
            ],
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.8,
        ),
    },
]
