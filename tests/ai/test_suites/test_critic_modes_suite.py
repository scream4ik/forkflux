import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.agents import RefinementFeedback
from app.prompts import BRUTALLY_HONEST_PROMPT, CRITIC_SYSTEM_PROMPT, combine_prompts

from ..eval_datasets.critic_brutally_honest_dataset import critic_brutally_honest_dataset


@pytest.fixture(scope="module")
def brutally_honest_critic_chain():
    final_prompt = combine_prompts(CRITIC_SYSTEM_PROMPT, BRUTALLY_HONEST_PROMPT)

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
    prompt_template = ChatPromptTemplate.from_messages([("system", final_prompt), ("human", "{input_text}")])
    return prompt_template | llm.with_structured_output(RefinementFeedback)


@pytest.mark.parametrize("test_spec", [pytest.param(spec, id=spec["id"]) for spec in critic_brutally_honest_dataset])
def test_critic_brutally_honest_suite(test_spec: dict, brutally_honest_critic_chain):
    input_text = test_spec["input"]
    metric = test_spec["metric"]
    context = test_spec.get("context")

    structured_output = brutally_honest_critic_chain.invoke({"input_text": input_text})

    assert structured_output.critical_flaws, "Critic failed to find any flaws in a deliberately bad idea!"

    formatted_actual_output = (
        f"CHAT RESPONSE:\n{structured_output.chat_response}\n\n"
        f"CRITICAL FLAWS:\n" + "\n".join([f"- {f}" for f in structured_output.critical_flaws]) + "\n\n"
        f"SUGGESTIONS:\n" + "\n".join([f"- {s}" for s in structured_output.suggestions])
    )

    test_case = LLMTestCase(input=input_text, actual_output=formatted_actual_output, context=context)
    assert_test(test_case, [metric])
