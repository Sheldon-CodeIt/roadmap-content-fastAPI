from fastapi import FastAPI
from pydantic import BaseModel
# environment libraries
from dotenv import load_dotenv
import os
import requests
# Langchain libs
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
# haystack ai libs
from haystack.utils import Secret
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder
from haystack import Pipeline, component
# utilities
from typing import Dict, Any, List, Union
import json
import json_repair

from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

# Define CORS Configuration

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
    "https://career-roadmap-five.vercel.app",

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

# load environment variable for groq
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


class Topic(BaseModel):
    title: str

class Step(BaseModel):
    topic: str
    step: str




# Start Of APIS
@app.get("/")
def index():
    return {"hello": "world"}


@app.post("/roadmap/")
def create_roadmap(topic: Topic):
    response = generate_quiz(topic.title)
    return response

@app.post("/step-description/")
def get_step_description(step: Step):
    response = step_description(step.topic, step.step)
    return response


@app.post("/recommend-course/")
def recommend(topic: Topic):
    response = recommend_courses(topic.title)
    return response


@app.post("/recommend-projects/")
def recommend(topic: Topic):
    response = recommend_projects(topic.title)
    return response


@app.post("/quiz/")
def generateQuiz(topic: Topic):
    response = generate_quizzes(topic.title)
    return response


# End of APIS




# Roadmap Creation Logic
roadmap_template = """Given the following skill/topic - "{{topic}}", create a learning roadmap in JSON format.
The roadmap will be like a tree which will be having branches. Each branch corresponds to a step (skill/task).
The step should also come with a description.
The steps and it's description should be unambiguous.
The description should also briefly mention the general topic of the text so that it can be understood by the student on how to learn that skill.
Minimum of 6 steps. Maximum of 8 steps.
respond with JSON only, no markdown or descriptions.

example JSON format you should absolutely follow:
{
  "topic": "a brief explanation of the topic and it's demand in the market",
  "steps": [
    {
      "step": "Step title",
      "description": "Explanation of what needs to be done in this step"
    },
    {
      "step": "Step title",
      "description": "Explanation of what needs to be done in this step"
    }, ...
    
  ]
}

"""

@component
class QuizParser:
    @component.output_types(quiz=Dict)
    def run(self, replies: List[Union[str, Dict[str, str]]]):
        # Combine all replies into a single string if they are not already
        combined_reply = ""
        for reply in replies:
            if isinstance(reply, dict):
                combined_reply += reply.get('text', '')
            elif isinstance(reply, str):
                combined_reply += reply

        # even if prompted to respond with JSON only, sometimes the model returns a mix of JSON and text
        first_index = min(combined_reply.find("{"), combined_reply.find("["))
        last_index = max(combined_reply.rfind("}"), combined_reply.rfind("]")) + 1

        json_portion = combined_reply[first_index:last_index]

        try:
            quiz = json.loads(json_portion)
        except json.JSONDecodeError:
            # if the JSON is not well-formed, try to repair it
            quiz = json_repair.loads(json_portion)

        # sometimes the JSON contains a list instead of a dictionary
        if isinstance(quiz, list):
            quiz = quiz[0]

        return {"quiz": quiz}


# initialize pipeline
quiz_generation_pipeline = Pipeline()
# add components to the pipeline
quiz_generation_pipeline.add_component(
    "prompt_builder", PromptBuilder(template=roadmap_template)
)

quiz_generation_pipeline.add_component(
    "generator",
    OpenAIGenerator(
        api_key=Secret.from_token(os.environ["GROQ_API_KEY"]),
        api_base_url="https://api.groq.com/openai/v1",
        model="mixtral-8x7b-32768",
        generation_kwargs = {"max_tokens": 2000}
    )
)

quiz_generation_pipeline.add_component("quiz_parser", QuizParser())

# connect the components
quiz_generation_pipeline.connect("prompt_builder", "generator")
quiz_generation_pipeline.connect("generator", "quiz_parser")


def generate_quiz(topic: str) -> Dict[str, Any]:
    result =  quiz_generation_pipeline.run(
        data={
                "prompt_builder": {"topic": [topic]},
        },
    )

    answer = result["quiz_parser"]["quiz"]

    return answer

# Get roadmap step description
def step_description(topic, step):
    # Retrieve and display detailed information about the roadmap topic
    # For this example, we'll just return a simple response
    chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768", max_tokens=200)
    system = "You are a content writer who provides description, uses, trends on a specified skill or Job Role"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "Write a short and consise description of about 50 words on {step} with repect to {topic}. Please only provide plain text. Do not give markdowns or special characters.")])

    chain = prompt | chat
    response = chain.invoke({"step": step, "topic": topic})
    description = response.content

    return description




course_recommendation_template = """Given the following skill/topic - "{{topic}}", provide a list of recommended online courses. 
The courses relevant to the topic. 
Respond with a JSON format containing the course name and URL. Give at least 10 course url.

example JSON format you should absolutely follow:
{
  "topic": "{{topic}}",
  "courses": [
    {
      "name": "Course name",
      "url": "Course URL"
    },
    {
      "name": "Course name",
      "url": "Course URL"
    }, ...
  ]
}

"""

@component
class CourseParser:
    @component.output_types(courses=Dict)
    def run(self, replies: List[Union[str, Dict[str, str]]]):
        # Combine all replies into a single string if they are not already
        combined_reply = ""
        for reply in replies:
            if isinstance(reply, dict):
                combined_reply += reply.get('text', '')
            elif isinstance(reply, str):
                combined_reply += reply

        # even if prompted to respond with JSON only, sometimes the model returns a mix of JSON and text
        first_index = min(combined_reply.find("{"), combined_reply.find("["))
        last_index = max(combined_reply.rfind("}"), combined_reply.rfind("]")) + 1

        json_portion = combined_reply[first_index:last_index]

        try:
            courses = json.loads(json_portion)
        except json.JSONDecodeError:
            # if the JSON is not well-formed, try to repair it
            courses = json_repair.loads(json_portion)

        # sometimes the JSON contains a list instead of a dictionary
        if isinstance(courses, list):
            courses = courses[0]

        return {"courses": courses}

# initialize pipeline
course_recommendation_pipeline = Pipeline()
# add components to the pipeline
course_recommendation_pipeline.add_component(
    "prompt_builder", PromptBuilder(template=course_recommendation_template)
)

course_recommendation_pipeline.add_component(
    "generator",
    OpenAIGenerator(
        api_key=Secret.from_token(os.environ["GROQ_API_KEY"]),
        api_base_url="https://api.groq.com/openai/v1",
        model="mixtral-8x7b-32768",
        generation_kwargs = {"max_tokens": 2000}
    )
)

course_recommendation_pipeline.add_component("course_parser", CourseParser())

# connect the components
course_recommendation_pipeline.connect("prompt_builder", "generator")
course_recommendation_pipeline.connect("generator", "course_parser")

def check_course_urls(courses: Dict[str, Any], topic: str) -> Dict[str, Any]:
    valid_courses = []
    print("Yoooo")
    for course in courses["courses"]:
        # Check if the URL exists
        print("hello")
        try:
            response = requests.head(course["url"])
            if response.status_code == 200:
                print("Status 200 OK")
                # # Check if the topic keyword is in the course name
                # if topic.lower() in course["name"].lower():
                valid_courses.append(course)
        except requests.RequestException:
            continue
    return {"topic": courses["topic"], "courses": valid_courses}

def recommend_courses(topic: str) -> Dict[str, Any]:
    result = course_recommendation_pipeline.run(
        data={
            "prompt_builder": {"topic": [topic]},
        },
    )

    answer = result["course_parser"]["courses"]
    # validated_answer = check_course_urls(answer, topic)

    return answer


project_recommendation_template = """Given the following skill/topic - "{{topic}}", provide a list of recommended projects. 
The projects should be relevant to the topic and should have detailed descriptions and tech stacks. Provide at least 5 good level projects.
Respond with a JSON format containing the project title, description, and tech stack. 

example JSON format you should absolutely follow:
{
  "topic": "{{topic}}",
  "projects": [
    {
      "title": "Project title",
      "description": "Project description",
      "tech_stack": "Project tech stack"
    },
    {
      "title": "Project title",
      "description": "Project description",
      "tech_stack": "Project tech stack"
    }, ...
  ]
}

"""

@component
class ProjectParser:
    @component.output_types(projects=Dict)
    def run(self, replies: List[Union[str, Dict[str, str]]]):
        # Combine all replies into a single string if they are not already
        combined_reply = ""
        for reply in replies:
            if isinstance(reply, dict):
                combined_reply += reply.get('text', '')
            elif isinstance(reply, str):
                combined_reply += reply

        # even if prompted to respond with JSON only, sometimes the model returns a mix of JSON and text
        first_index = min(combined_reply.find("{"), combined_reply.find("["))
        last_index = max(combined_reply.rfind("}"), combined_reply.rfind("]")) + 1

        json_portion = combined_reply[first_index:last_index]

        try:
            projects = json.loads(json_portion)
        except json.JSONDecodeError:
            # if the JSON is not well-formed, try to repair it
            projects = json_repair.loads(json_portion)

        # sometimes the JSON contains a list instead of a dictionary
        if isinstance(projects, list):
            projects = projects[0]

        return {"projects": projects}
    



# initialize pipeline
project_recommendation_pipeline = Pipeline()
# add components to the pipeline
project_recommendation_pipeline.add_component(
    "prompt_builder", PromptBuilder(template=project_recommendation_template)
)

project_recommendation_pipeline.add_component(
    "generator",
    OpenAIGenerator(
        api_key=Secret.from_token(os.environ["GROQ_API_KEY"]),
        api_base_url="https://api.groq.com/openai/v1",
        model="mixtral-8x7b-32768",
        generation_kwargs = {"max_tokens": 2000}
    )
)

project_recommendation_pipeline.add_component("project_parser", ProjectParser())

# connect the components
project_recommendation_pipeline.connect("prompt_builder", "generator")
project_recommendation_pipeline.connect("generator", "project_parser")

def recommend_projects(topic: str) -> Dict[str, Any]:
    result = project_recommendation_pipeline.run(
        data={
            "prompt_builder": {"topic": [topic]},
        },
    )

    answer = result["project_parser"]["projects"]

    return answer


quiz_generation_template = """Given the following skill/topic - "{{topic}}", create 5 multiple choice quizzes in JSON format.
Each quiz should have a question related to the topic, with 4 different options, and only one correct answer.
Ensure that the options are clear and unambiguous, and each option should start with a letter followed by a period and a space (e.g., "a. option").
The questions should cover various aspects of the topic and require reasoning and understanding of the topic.
Avoid giving hints in one question that could help answer other questions.

Example JSON format you should absolutely follow:
{
  "topic": "A brief explanation of the topic",
  "questions": [
    {
      "id": 1,
      "question": "Text of the question",
      "options": [
        "a. First option",
        "b. Second option",
        "c. Third option",
        "d. Fourth option"
      ],
      "right_option": "c"  # The letter corresponding to the correct option ("a", "b", "c", or "d")
    },
    {
      "id": 2,
      "question": "Text of the question",
      "options": [
        "a. First option",
        "b. Second option",
        "c. Third option",
        "d. Fourth option"
      ],
      "right_option": "a"  # The letter corresponding to the correct option
    },
    ...
  ]
}

Respond with JSON only, no markdown or descriptions.
"""



@component
class QuizParser:
    @component.output_types(quizzes=Dict)
    def run(self, replies: List[Union[str, Dict[str, str]]]):
        # Combine all replies into a single string if they are not already
        combined_reply = ""
        for reply in replies:
            if isinstance(reply, dict):
                combined_reply += reply.get('text', '')
            elif isinstance(reply, str):
                combined_reply += reply

        # even if prompted to respond with JSON only, sometimes the model returns a mix of JSON and text
        first_index = min(combined_reply.find("{"), combined_reply.find("["))
        last_index = max(combined_reply.rfind("}"), combined_reply.rfind("]")) + 1

        json_portion = combined_reply[first_index:last_index]

        try:
            quizzes = json.loads(json_portion)
        except json.JSONDecodeError:
            # if the JSON is not well-formed, try to repair it
            quizzes = json_repair.loads(json_portion)

        # sometimes the JSON contains a list instead of a dictionary
        if isinstance(quizzes, list):
            quizzes = quizzes[0]

        return {"quizzes": quizzes}


# Initialize the pipeline
quiz_generation_pipeline = Pipeline()

# Add components to the pipeline
quiz_generation_pipeline.add_component(
    "prompt_builder", PromptBuilder(template=quiz_generation_template)
)

quiz_generation_pipeline.add_component(
    "generator",
    OpenAIGenerator(
        api_key=Secret.from_token(os.environ["GROQ_API_KEY"]),
        api_base_url="https://api.groq.com/openai/v1",
        model="mixtral-8x7b-32768",
        generation_kwargs={"max_tokens": 2000}
    )
)

quiz_generation_pipeline.add_component("quiz_parser", QuizParser())

# Connect the components
quiz_generation_pipeline.connect("prompt_builder", "generator")
quiz_generation_pipeline.connect("generator", "quiz_parser")

def generate_quizzes(topic: str) -> Dict[str, Any]:
    result = quiz_generation_pipeline.run(
        data={
            "prompt_builder": {"topic": [topic]},
        },
    )

    answer = result["quiz_parser"]["quizzes"]

    return answer




# Main Function
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
