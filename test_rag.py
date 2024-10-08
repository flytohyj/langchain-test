
from langchain_openai import ChatOpenAI
from config import GLM_API_KEY

import os
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

llm  = ChatOpenAI(
        temperature=0.95,
        model="glm-4-flash",
        api_key=GLM_API_KEY,
        base_url="https://open.bigmodel.cn/api/paas/v4/")
os.environ["ZHIPUAI_API_KEY"]  = GLM_API_KEY



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents  = text_splitter.split_documents(docs)


vectorstore = Chroma.from_documents(documents=documents , embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition?")