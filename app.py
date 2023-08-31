!pip install bs4

import streamlit as st
# used to load text
from langchain.document_loaders import WebBaseLoader
# used to create the retriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
# used to create the retrieval tool
from langchain.agents import tool
# used to create the memory
from langchain.memory import ConversationBufferMemory
# used to create the prompt template
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder
# used to create the agent executor
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor

# Page title
st.set_page_config(page_title='🦜🔗 Conversational Question Answering App')
st.title('🦜🔗 Conversational Question Answering App')

# create the database and retriever
# load data
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()
# split the text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(data)
# create embediings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# load embeddings into a vectorstore
db = FAISS.from_documents(texts, embeddings)
# instantiate the retriever
retriever = db.as_retriever()

# create the document retrieval tool
@tool
def tool(query):
    "Searches and returns documents regarding the llm powered autonomous agents blog"
    docs = retriever.get_relevant_documents(query)
    return docs

tools = [tool]

#instantiate the agent memory
memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True)

#agent prompt
system_message = SystemMessage(
        content=(
            "Do your best to answer the questions. "
            "Feel free to use any tools available to look up "
            "relevant information, only if neccessary"
        )
)

prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
    )

# Generate LLM response
def generate_response(input_query):
    llm = ChatOpenAI(temperature = 0, openai_api_key=openai_api_key)
    # Create Agent
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    # Execute Agent
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
    response = agent_executor({"input": input_query})
    return st.success(response["output"])

with st.form('conversational_qa_form'):
  topic_text = st.text_input('Enter your question:', '')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if submitted and openai_api_key.startswith('sk-'):
    generate_response(topic_text)
