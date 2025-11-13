from .orchestrator import ManualOrchestrator


task = "Develop a Go-To-Market strategy for an AI startup that is creating a plugin for Figma."
orchestrator = ManualOrchestrator(main_task=task)
orchestrator.add_agent(
    name="generator",
    system_prompt="You are a creative marketer, your task is to generate ideas and create draft documents.",
    model="gpt-4o-mini",
)
orchestrator.add_agent(
    name="critic",
    system_prompt="You are a pragmatic business analyst. Your task is to critically evaluate ideas, identify risks, and propose concrete, measurable steps.",
    model="gpt-4.1-mini",
)

prompt_for_generator = "Create the first draft of a GTM strategy. Outline the key acquisition channels."
generator_response_1 = orchestrator.talk_to(
    agent_name="generator", input_text=prompt_for_generator, thread_id="generator"
)
print(generator_response_1)

prompt_for_critic = "Analyze this strategy. Which channels are the riskiest and why? Suggest alternatives."
critic_response_1 = orchestrator.talk_to(
    agent_name="critic", input_text=prompt_for_critic, thread_id="critic", context_from="generator"
)
print(critic_response_1)

prompt_for_generator_2 = "Great observations. Please rewrite the channels section taking this feedback into account."
generator_response_2 = orchestrator.talk_to(
    agent_name="generator", input_text=prompt_for_generator_2, thread_id="generator", context_from="critic"
)
print(generator_response_2)

prompt_for_generator_3 = "Okay, now add a section on the budget."
generator_response_3 = orchestrator.talk_to(
    agent_name="generator", input_text=prompt_for_generator_3, thread_id="generator"
)
print(generator_response_3)
