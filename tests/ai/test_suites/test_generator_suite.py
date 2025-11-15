import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from ..eval_datasets.generator_golden_dataset import generator_golden_dataset


@pytest.mark.parametrize("test_spec", [pytest.param(spec, id=spec["id"]) for spec in generator_golden_dataset])
def test_generator_golden_suite(test_spec: dict, generator_chain):
    model_input = test_spec["input"]
    test_case_input = test_spec.get("test_case_input", model_input)
    metric = test_spec["metric"]
    context = test_spec.get("context")

    actual_output = generator_chain.invoke({"input_text": model_input})

    test_case = LLMTestCase(input=test_case_input, actual_output=actual_output, context=context)
    assert_test(test_case, [metric])
