from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://www.amazon.in/HP-Smartchoice-i7-13620H-Upgradeable-Office24/dp/B0FDKPZHB7/?_encoding=UTF8&pd_rd_w=RIzmS&content-id=amzn1.sym.340182bc-8d5c-49c7-8b69-c0403f7ba3a7%3Aamzn1.symc.752cde0b-d2ce-4cce-9121-769ea438869e&pf_rd_p=340182bc-8d5c-49c7-8b69-c0403f7ba3a7&pf_rd_r=PN8W63RTCG2JSMA3HA1S&pd_rd_wg=MrXK4&pd_rd_r=df4c13c9-d5f4-4fd3-bddb-b27ebec695c0&ref_=pd_hp_d_atf_ci_mcx_mr_'
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({'question':'What is the product that we are talking about?', 'text':docs[0].page_content}))
