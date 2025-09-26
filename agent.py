from strands import Agent, tool
from strands.models import BedrockModel
from camera_tool import camera, analyze_image
from strands_tools.browser import AgentCoreBrowser

# Create a BedrockModel
bedrock_model = BedrockModel(
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    region_name="us-west-2",
    temperature=0.3,
)
agent_core_browser = AgentCoreBrowser(region="us-west-2")

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
            Provide personalized activity suggestions based on user group size.
            Assume that the weather is rainy and the time is afternoon.
            Keep the answer as short as possible and just list 3 activities."""
    )

    return activity_agent(prompt)

@tool
def book_recommender(prompt=None):
    book_agent = Agent(
        model=bedrock_model,
        tools=[agent_core_browser.browser],
        system_prompt="""You are a specialized book recommendation assistant.
            Provide personalized book suggestion. Assume that the user likes science fiction and fantasy books.
            Take suggestions from amazon.de. Keep the answer as short as possible and just list 3 books.""",
    )

    return book_agent(prompt)

@tool
def movie_recommender(prompt=None):
    movie_agent = Agent(
        model=bedrock_model,
        tools=[agent_core_browser.browser],
        system_prompt="""You are a specialized movie recommendation assistant.
        Use https://friedrichsbau-kino.de/programm-tickets/friedrichsbau/#default to find the movies for today in theaters in Freiburg, Germany.
        Provide a concise list of the movies. Add the FSK and Genre if available. Try to keep it to 1 line per movie."""
    )

    return movie_agent(prompt)

agent = Agent(
    model=bedrock_model,
    tools=[camera_agent, activity_recommender, book_recommender, movie_recommender],
    system_prompt="""You are a general activity recommendation assistant.
        When asked for activity suggestions based on group size, first use the camera_agent tool to determine the number of people present.
        Then, use the activity_recommender tool.
        If books are in the suggested activities, use the book_recommender to provide recommendations.
        If movies are in the suggested activities, use the movie_recommender to provide recommendations.
        Don't add any extra information, just provide the recommendations provided by the tools.""",
)

# agent("How many people are in front of the pc?")
agent("What can we do")


