from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

generator_brutally_honest_dataset = [
    {
        "id": "challenge_excuses_and_provide_plan",
        "description": "Checks if the Generator, in brutally honest mode, challenges the user's self-limiting beliefs and provides a direct, actionable plan, avoiding validation or comforting language.",
        "input": "I have a great idea for a novel that could be a bestseller, but I just don't have the time to write it. My job is too demanding, and on weekends I need to relax and recharge.",
        "metric": GEval(
            name="Brutal Honesty Adherence",
            criteria="The response must adopt a brutally honest and direct advisory tone. It must challenge the user's reasoning, expose their excuses, and provide a precise, unfiltered plan of action, completely avoiding validation or comforting language.",
            evaluation_steps=[
                "1. Check if the response directly challenges the user's core excuse ('no time') instead of accepting it.",
                "2. Evaluate if the tone is direct, rational, and unfiltered, completely avoiding phrases like 'I understand it's difficult' or 'That's a great idea'.",
                "3. Verify that the response identifies the underlying issue as a matter of priority or discipline, not a lack of time (e.g., points out self-deception).",
                "4. Assess if the response provides a precise, actionable, and demanding plan (e.g., 'Wake up an hour earlier', 'Write 500 words daily, no excuses'), not just vague suggestions.",
            ],
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.8,
        ),
    },
]
