import os
from crewai import Agent, Task, Crew, Process

# Loading OpenAI API key from .env file
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Loading DuckDuckGoSearch
from langchain_community.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()

# Loading Human Tools
from langchain.agents import load_tools
human_tools = load_tools(["human"])

# Context about LA
la_context = (
    "LA is a fresh startup in the software industry, aiming to quickly establish a reputation for "
    "innovation and excellence. The company seeks to participate in exciting projects and explore "
    "opportunities for revenue generation. The three founders, Stashka, Sale, and Rajko, bring diverse "
    "and complementary skills to the table. Stashka has expertise in leadership, management, and general "
    "software and AI knowledge. Sale is proficient in Java, SpringBoot, AWS, and backend development. "
    "Rajko specializes in ReactJS, Svelte, Flutter, web and mobile apps development, animations, and has a "
    "strong interest in AI. LA's primary goals are to identify market opportunities, develop a solid business "
    "strategy, and leverage its founders' skills to grow quickly and sustainably."
)

# Define agents with their roles, expertise, and informed backstory
market_research_analyst = Agent(
    role='Market Research Analyst',
    goal='Analyze the software market and competitors to provide insights for LA.',
    backstory=f'{la_context} As a Market Research Analyst, your task is to analyze the software market with a focus on areas where LA can leverage its founders\' skills and knowledge.',
    allow_delegation=False,
    tools=[search_tool, *human_tools],
    verbose=True
)

business_development_strategist = Agent(
    role='Business Development Strategist',
    goal='Identify growth opportunities and strategic partnerships for LA.',
    backstory=f'{la_context} As a Business Development Strategist, your aim is to identify growth strategies and partnerships that align with LA\'s expertise and goals.',
    allow_delegation=True,
    tools=[search_tool, *human_tools],
    verbose=True
)

technology_trend_analyst = Agent(
    role='Technology Trend Analyst',
    goal='Assess current and emerging technologies relevant to LA’s expertise.',
    backstory=f'{la_context} As a Technology Trend Analyst, your role is to assess technology trends, especially in AI, web, and mobile app development, to find areas where LA can excel.',
    allow_delegation=False,
    tools=[search_tool, *human_tools],
    verbose=True
)

brand_and_marketing_expert = Agent(
    role='Brand and Marketing Expert',
    goal='Develop branding and marketing strategies for LA.',
    backstory=f'{la_context} As a Brand and Marketing Expert, you need to devise strategies that effectively establish LA’s brand in the tech industry, considering its unique strengths and goals.',
    allow_delegation=True,
    tools=[search_tool, *human_tools],
    verbose=True
)

# Define tasks for each agent
task1 = Task(
    description='Conduct a thorough market and competitor analysis for the software industry, focusing on areas relevant to LA’s expertise.',
    agent=market_research_analyst
)

task2 = Task(
    description='Identify potential growth strategies, business models, and strategic partnerships that LA can pursue.',
    agent=business_development_strategist
)

task3 = Task(
    description='Evaluate current and upcoming technology trends that LA can leverage, particularly in AI, Java, ReactJS, and mobile app development.',
    agent=technology_trend_analyst
)

task4 = Task(
    description='Develop a comprehensive branding and marketing strategy to effectively establish and promote LA’s presence in the tech industry.',
    agent=brand_and_marketing_expert
)

# Create the crew with a sequential process
crew = Crew(
    agents=[market_research_analyst, business_development_strategist, technology_trend_analyst, brand_and_marketing_expert],
    tasks=[task1, task2, task3, task4],
    process=Process.sequential,
    verbose=True
)

# Kickoff the crew to start the process
result = crew.kickoff()

# Output the results
print(result)
