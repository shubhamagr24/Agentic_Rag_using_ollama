from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = OllamaLLM(model="deepseek-r1:1.5b", temperature=0.2)


prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. You will be given a question and you need to answer it in detail.
Question: {question}
Answer: 
""")

parser=StrOutputParser()

# Create the chain
chain = prompt | model | parser

print(chain.invoke({"question":"Key Difference between Rag And Agaentic Rag"}))


