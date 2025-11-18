import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from ..eval_datasets.critic_golden_dataset import critic_golden_dataset


def format_critic_output(structured_response):
    flaws_text = "\n".join([f"- {f}" for f in structured_response.critical_flaws])
    suggestions_text = "\n".join([f"- {s}" for s in structured_response.suggestions])

    return (
        f"CHAT RESPONSE (To User):\n{structured_response.chat_response}\n\n"
        f"CRITICAL FLAWS DETECTED:\n{flaws_text}\n\n"
        f"SUGGESTED FIXES:\n{suggestions_text}"
    )


@pytest.mark.parametrize("test_spec", [pytest.param(spec, id=spec["id"]) for spec in critic_golden_dataset])
def test_critic_golden_suite(test_spec: dict, critic_chain):
    input_text = test_spec["input"]
    metric = test_spec["metric"]
    context = test_spec.get("context")

    structured_output = critic_chain.invoke({"input_text": input_text})

    actual_output = format_critic_output(structured_output)

    test_case = LLMTestCase(input=input_text, actual_output=actual_output, context=context)
    assert_test(test_case, [metric])
