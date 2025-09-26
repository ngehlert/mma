from strands import Agent, tool
from strands.models import BedrockModel
from camera_tool import camera, analyze_image

# Create a BedrockModel
bedrock_model = BedrockModel(
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    region_name="us-west-2",
    temperature=0.3,
)

@tool
def camera_agent(prompt=None):
    current_agent = Agent(
        model=bedrock_model,
        tools=[camera, analyze_image],
        system_prompt="""You are a group size detector. You are able to take screenshots with the
        camera tool and analyze them with the analyze_image tool to determine how many people are present.""",
    )

    return current_agent(prompt)

@tool
def activity_recommender(prompt=None):
    activity_agent = Agent(
        model=bedrock_model,
        tools=[],
        system_prompt="""You are a specialized activity recommendation assistant.
            Provide personalized activity suggestions based on user group size and weather."""
    )

agent = Agent(
    model=bedrock_model,
    tools=[camera_agent],
    system_prompt="""You are a general activity recommendation assistant.
            Based on the number of people and other factors, like weather and time of day, recommend suitable activities."""
)

agent("How many people are in front of the pc?")


