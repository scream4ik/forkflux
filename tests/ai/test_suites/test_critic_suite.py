import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from ..eval_datasets.critic_golden_dataset import critic_golden_dataset


@pytest.mark.parametrize("test_spec", [pytest.param(spec, id=spec["id"]) for spec in critic_golden_dataset])
def test_critic_golden_suite(test_spec: dict, critic_chain):
    input_text = test_spec["input"]
    metric = test_spec["metric"]
    context = test_spec.get("context")

    actual_output = critic_chain.invoke({"input_text": input_text})

    test_case = LLMTestCase(input=input_text, actual_output=actual_output, context=context)
    assert_test(test_case, [metric])
