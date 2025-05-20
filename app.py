from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vector import retriever

model = OllamaLLM(model="deepseek-r1:1.5b", temperature=0.2)


prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. You will be given a question and you need to answer it in detail.
Below is the context to help you answer the question.
{context}
Question: {question}
Answer: 
""")

parser=StrOutputParser()

# Create the chain
chain = prompt | model | parser

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    context = retriever.invoke(question)
    result = chain.invoke({"context": context, "question": question})
    print(result)

