from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def build_qa_chain(vectorstore):
    llm = ChatOpenAI(temperature=0)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    
    return qa_chain
