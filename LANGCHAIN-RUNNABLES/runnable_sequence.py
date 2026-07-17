from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-20b',
    task='text-genertaion'
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)

prompt1 = PromptTemplate(
    template='Explain this joke to me {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser, prompt1, model, parser)

print(chain.invoke({'topic': 'AI'}))
