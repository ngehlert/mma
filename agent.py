from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools.browser import AgentCoreBrowser
from strands.agent.conversation_manager import SlidingWindowConversationManager
from bedrock_agentcore.runtime import BedrockAgentCoreApp
import cv2
import os
import hashlib
import time

# Configure conversation management for production
conversation_manager = SlidingWindowConversationManager(
    window_size=100,  # Limit history size
)

# Create a BedrockModel
bedrock_model = BedrockModel(
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    region_name="us-west-2",
    temperature=0.3,
)
agent_core_browser = AgentCoreBrowser(region="us-west-2")

@tool
def camera(prompt=None):
    screenshots_dir = os.path.join(os.path.dirname(__file__), "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    # Generate a unique hash for the filename using time and randomness
    unique_str = f"{time.time()}_{os.urandom(8).hex()}"
    unique_hash = hashlib.sha256(unique_str.encode()).hexdigest()[:16]
    img_path = os.path.join(screenshots_dir, f"screenshot_{unique_hash}.jpg")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture image")

    # Enhance image quality using CLAHE and brightness/contrast adjustment
    # Convert to LAB color space for better contrast enhancement
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # Optionally, increase brightness and contrast
    alpha = 1.2  # Contrast control (1.0-3.0)
    beta = 20    # Brightness control (0-100)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)

    cv2.imwrite(img_path, enhanced)
    return img_path

@tool
def analyze_image(prompt=None):
    """
    Analyze an image given its path. Returns basic info and tries to detect if there are no persons, a single person, or multiple persons.
    Usage: analyze_image('screenshots/screenshot_xxx.jpg')
    """
    if not prompt:
        return "Please provide the image path to analyze."
    img_path = prompt.strip()
    if not os.path.exists(img_path):
        return f"Image not found: {img_path}"
    img = cv2.imread(img_path)
    if img is None:
        return f"Failed to load image: {img_path}"
    height, width, channels = img.shape
    mean_color = img.mean(axis=(0, 1)).tolist()  # BGR mean

    # Person detection using Haar Cascade (frontal face)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        person_result = "Cascade file not found. Person detection unavailable."
        num_persons = None
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        num_persons = len(faces)
        if num_persons == 0:
            person_result = "No persons detected."
        elif num_persons == 1:
            person_result = "Single person detected."
        else:
            person_result = f"Multiple persons detected: {num_persons}"

    return {
        "path": img_path,
        "width": width,
        "height": height,
        "channels": channels,
        "mean_color_bgr": mean_color,
        "person_detection": person_result,
        "person_count": num_persons
    }


@tool
def camera_agent(prompt=None):
    current_agent = Agent(
        model=bedrock_model,
        tools=[camera],
        conversation_manager=conversation_manager,
        system_prompt="""You are a camera. You are able to take screenshots with the camera tool.""",
    )

    return current_agent(prompt)

@tool
def analyse_image_agent(prompt=None):
    current_agent = Agent(
        model=bedrock_model,
        tools=[analyze_image],
        conversation_manager=conversation_manager,
        system_prompt="""You are a group size detector. You are able to take analyze the provided image with the analyze_image 
        tool to determine how many people are present.""",
    )

    return current_agent(prompt)

@tool
def activity_recommender(prompt=None):
    activity_agent = Agent(
        model=bedrock_model,
        tools=[],
        conversation_manager=conversation_manager,
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
        conversation_manager=conversation_manager,
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
        conversation_manager=conversation_manager,
        system_prompt="""You are a specialized movie recommendation assistant.
        Use https://friedrichsbau-kino.de/programm-tickets/friedrichsbau/#default to find the movies for today in theaters in Freiburg, Germany.
        Provide a concise list of the movies. Add the FSK and Genre if available. Try to keep it to 1 line per movie."""
    )

    return movie_agent(prompt)

agent = Agent(
    model=bedrock_model,
    tools=[camera_agent, analyse_image_agent, activity_recommender, book_recommender, movie_recommender],
    conversation_manager=conversation_manager,
    system_prompt="""You are a general activity recommendation assistant.
        When asked for activity suggestions based on group size try to use the camera_agent tool to take a picture.
        Then, take the image path and use the analyse_image_agent tool to determine how many people are present.
        With the number of people present use the activity_recommender tool to get suggestions for activities.
        If books are in the suggested activities, use the book_recommender to provide recommendations.
        If movies are in the suggested activities, use the movie_recommender to provide recommendations.
        Don't add any extra information, just provide the recommendations provided by the tools.""",
)

# agent("How many people are in front of the pc?")
# agent("What can we do?")
app = BedrockAgentCoreApp()

@app.entrypoint
async def agent_invocation(payload):
    """Handler for agent invocation"""
    user_message = payload.get(
        "prompt", "No prompt found in input, please guide customer to create a json payload with prompt key"
    )
    stream = agent.stream_async(user_message)
    async for event in stream:
        print(event)
        yield (event)

if __name__ == "__main__":
    app.run()


