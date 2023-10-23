import streamlit as st
import openai

# used to load text
from langchain.document_loaders import WebBaseLoader

# used to create the retriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# used to create the agent
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# set the secure key
openai_api_key = st.secrets.openai_key

# add a heading for your app.
st.header("Chat with the LLM Agents blog 💬 📚")

# Initialize the chat message history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about LLM based agents!"}
    ]
# create the document database
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the LLM blog – hang tight!."):
        loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
        data = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(data)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = FAISS.from_documents(texts, embeddings)
        return db

db = load_data()

# instantiate the database retriever
retriever = db.as_retriever()

# instantiate the large language model
llm = ChatOpenAI(temperature = 0, openai_api_key=openai_api_key)

# instantiate question answering service
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Prompt for user input and display message history
if prompt := st.chat_input("Your LLM based agent related question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Pass query to chat engine and display response
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    print("Messages: ".format(st.session_state.messages))
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            print("Prompt: ".format(prompt))
            response = qa.run(prompt)
            print("Response: {}".format(response))
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message) # Add response to message history

