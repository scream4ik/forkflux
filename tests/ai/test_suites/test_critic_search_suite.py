import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from ..eval_datasets.critic_search_dataset import critic_search_dataset


def format_critic_output_with_sources(structured_response) -> str:
    flaws_text = "\n".join([f"- {f}" for f in structured_response.critical_flaws])
    suggestions_text = "\n".join([f"- {s}" for s in structured_response.suggestions])

    sources = getattr(structured_response, "sources", [])
    if not sources:
        sources_text = "No sources provided (Test run without internet access)."
    else:
        sources_text = "\n".join([f"- {c}" for c in sources])

    return (
        f"CHAT RESPONSE:\n{structured_response.chat_response}\n\n"
        f"CRITICAL FLAWS:\n{flaws_text}\n\n"
        f"SUGGESTIONS:\n{suggestions_text}\n\n"
        f"SOURCES:\n{sources_text}"
    )


@pytest.mark.parametrize("test_spec", [pytest.param(spec, id=spec["id"]) for spec in critic_search_dataset])
def test_critic_search_suite(test_spec: dict, critic_chain):
    input_text = test_spec["input"]
    metric = test_spec["metric"]
    context = test_spec.get("context", [])

    structured_output = critic_chain.invoke({"input_text": input_text})

    actual_output_text = format_critic_output_with_sources(structured_output)

    test_case = LLMTestCase(input=input_text, actual_output=actual_output_text, context=context)
    assert_test(test_case, [metric])
