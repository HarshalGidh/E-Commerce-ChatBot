from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from src.data_ingestion import ingestdata
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def generation(vstore):
    retriever = vstore.as_retriever(search_kwargs={"k": 3})

    PRODUCT_BOT_TEMPLATE = """
    Your ecommercebot bot is an expert in product recommendations and customer queries.
    It analyzes product titles and reviews to provide accurate and helpful responses.
    Ensure your answers are relevant to the product context and refrain from straying off-topic.
    Your responses should be concise and informative.If you get any general messages reply to them 
    by saying you are an Ecommerce Chatbot and you can help user buy earbuds/headphones.
    If its anything out of context dont give any information about products.If user asks about products suggest them
    good products for them and ask what they are looking for.

    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:
    
    """


    prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_TEMPLATE)

    # llm = ChatOpenAI()
    llm = ChatGoogleGenerativeAI(
            model = "gemini-1.5-flash",
            temperature=0.7,
            top_p=0.85,
            google_api_key=GOOGLE_API_KEY
        )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

if __name__=='__main__':
    vstore = ingestdata("done")
    chain  = generation(vstore)
    # print(chain.invoke({"context": "", "question": "can you tell me the best bluetooth buds?"}))
    print(chain.invoke("can you tell me the best bluetooth buds?"))