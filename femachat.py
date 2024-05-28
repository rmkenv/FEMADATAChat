import os
import requests
import pandas as pd
from IPython.display import display, HTML
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain import hub, LLMMathChain
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain.agents import AgentExecutor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

#use your own API key, from (https://makersuite.google.com/), remember to use VPN if you are not in US
#put your key inside the apostrophe....
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Setup model
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    convert_system_message_to_human=True,
    handle_parsing_errors=True,
    temperature=0.6,
    max_tokens=200,
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
)

ddg_search = DuckDuckGoSearchAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

tools = [
    Tool(
        name="DuckDuckGo Search",
        func=ddg_search.run,
        description="Useful to browse information from the internet to know recent results and information you don't know. Then, tell user the result.",
    ),
    Tool.from_function(
        func=llm_math_chain.run,
        name="Calculator",
        description="Useful for when you need to answer questions about very simple math. This tool is only for math questions and nothing else. Only input math expressions. Not supporting symbolic math.",
    ),
]

agent_prompt = hub.pull("mikechan/gemini")
prompt = agent_prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([tool.name for tool in tools]),
)
llm_with_stop = llm.bind(stop=["\nObservation"])

memory = ConversationBufferMemory(memory_key="chat_history")

agent = {
    "input": lambda x: x["input"],
    "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    "chat_history": lambda x: x["chat_history"],
} | prompt | llm_with_stop | ReActSingleInputOutputParser()

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

def get_fema_data(parameters={}):
    base_url = 'https://www.fema.gov/api/open/v2/FimaNfipClaims'
    try:
        response = requests.get(base_url, params=parameters)
        response.raise_for_status()
        data = response.json()
        return data.get('FimaNfipClaims', [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return []
    except ValueError as e:
        print(f"Error parsing JSON response: {e}")
        return []

# Collect user input for zip code
zip_code = input("Please enter your zip code: ")

# Example invocation with FEMA API
params = {
    'reportedZipCode': zip_code  # Set the user's zip code
}

data = get_fema_data(params)

if data:
    # Select columns to display
    columns = [
        'asOfDate',
        'basementEnclosureCrawlspaceType',
        'policyCount',
        'crsClassificationCode',
        'dateOfLoss',
        'elevationCertificateIndicator',
        'elevationDifference',
        'baseFloodElevation',
        'ratedFloodZone',
        'primaryResidenceIndicator',
        'buildingDamageAmount',
        'contentsDamageAmount',
        'yearOfLoss'
    ]
    table_data = [{col: record.get(col, '') for col in columns} for record in data]
    df = pd.DataFrame(table_data, columns=columns)
    
    # Display the DataFrame in a nice HTML format
    display(HTML(df.to_html(classes='table table-striped table-bordered table-hover', index=False)))
    
    # Export the DataFrame to CSV
    csv_file = f'fema_data_{zip_code}.csv'
    df.to_csv(csv_file, index=False)
    print(f'Data has been exported to {csv_file}')
else:
    print("No results found for the specified zip code.")

# Define a function to calculate the total building damage amount
def total_building_damage_amount(*args):
    try:
        total_amount = df['buildingDamageAmount'].sum()
        return f"The total building damage amount is ${total_amount:,.2f}."
    except Exception as e:
        return str(e)

# Add the function as a tool to the agent
total_damage_tool = Tool.from_function(
    func=total_building_damage_amount,
    name="Total Building Damage Amount",
    description="Returns the total building damage amount from the DataFrame."
)

# Define a function to calculate the average contents damage amount
def average_contents_damage_amount(*args):
    try:
        average_amount = df['contentsDamageAmount'].mean()
        return f"The average contents damage amount is ${average_amount:,.2f}."
    except Exception as e:
        return str(e)

# Add the function as a tool to the agent
average_damage_tool = Tool.from_function(
    func=average_contents_damage_amount,
    name="Average Contents Damage Amount",
    description="Returns the average contents damage amount from the DataFrame."
)

# Define a function to find the most recent loss date
def most_recent_loss_date(*args):
    try:
        most_recent_date = df['dateOfLoss'].max()
        return f"The most recent date of loss is {most_recent_date}."
    except Exception as e:
        return str(e)

# Add the function as a tool to the agent
recent_loss_tool = Tool.from_function(
    func=most_recent_loss_date,
    name="Most Recent Date of Loss",
    description="Returns the most recent date of loss from the DataFrame."
)

# Define a function to count policies by flood zone
def count_policies_by_flood_zone(*args):
    try:
        count_by_zone = df['ratedFloodZone'].value_counts().to_dict()
        result = "Policies by flood zone:\n" + "\n".join([f"{zone}: {count}" for zone, count in count_by_zone.items()])
        return result
    except Exception as e:
        return str(e)

# Add the function as a tool to the agent
policies_by_zone_tool = Tool.from_function(
    func=count_policies_by_flood_zone,
    name="Count Policies by Flood Zone",
    description="Returns the count of policies by flood zone from the DataFrame."
)

# Define a function to calculate the total number of claims
def total_number_of_claims(*args):
    try:
        total_claims = df.shape[0]
        return f"The total number of claims is {total_claims}."
    except Exception as e:
        return str(e)

# Add the function as a tool to the agent
total_claims_tool = Tool.from_function(
    func=total_number_of_claims,
    name="Total Number of Claims",
    description="Returns the total number of claims from the DataFrame."
)

# Define a function to calculate the sum of building and contents damage amount
def total_building_and_contents_damage(*args):
    try:
        total_building_damage = df['buildingDamageAmount'].sum()
        total_contents_damage = df['contentsDamageAmount'].sum()
        total_damage = total_building_damage + total_contents_damage
        return f"The total sum of building and contents damage amounts is ${total_damage:,.2f}."
    except Exception as e:
        return str(e)

# Add the function as a tool to the agent
total_damage_tool = Tool.from_function(
    func=total_building_and_contents_damage,
    name="Total Building and Contents Damage Amount",
    description="Returns the total sum of building and contents damage amounts from the DataFrame."
)

# Add new tools to the existing tools list
tools.extend([total_damage_tool, average_damage_tool, recent_loss_tool, policies_by_zone_tool, total_claims_tool])

# Reinitialize the agent with the new tools
agent_prompt = hub.pull("mikechan/gemini")
prompt = agent_prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([tool.name for tool in tools]),
)

agent = {
    "input": lambda x: x["input"],
    "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    "chat_history": lambda x: x["chat_history"],
} | prompt | llm_with_stop | ReActSingleInputOutputParser()

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

# Use the agent executor to handle more complex logic if necessary
response = agent_executor.invoke({
    "input": f"Fetch FEMA data for zip code {zip_code}."
})

print(response["output"])

# Example of querying the total building damage amount
query_response = agent_executor.invoke({
    "input": "What is the total building damage amount?"
})

print(query_response["output"])

# Example of querying the average contents damage amount
query_response = agent_executor.invoke({
    "input": "What is the average contents damage amount?"
})

print(query_response["output"])

# Example of querying the most recent loss date
query_response = agent_executor.invoke({
    "input": "What is the most recent date of loss?"
})

print(query_response["output"])

# Example of querying the number of policies by flood zone
query_response = agent_executor.invoke({
    "input": "How many policies are there by flood zone?"
})

print(query_response["output"])

# Example of querying the total number of claims
query_response = agent_executor.invoke({
    "input": "What is the total number of claims?"
})

print(query_response["output"])

# Example of querying the total building and contents damage amounts
query_response = agent_executor.invoke({
    "input": "What is the total sum of building and contents damage amounts?"
})

print(query_response["output"])
