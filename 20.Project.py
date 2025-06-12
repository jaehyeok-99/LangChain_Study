# 인공지능 PDF Q&A 챗봇 프로젝트

#from dotenv import load_dotenv #api 키 .env으로 관리할경우
#load_dotenv()

import gradio as gr
#from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

# 환경 변수 불러오기
#load_dotenv()

# LLM 설정(유료,무료)
#llm = ChatOpenAI(model="gpt-4o-mini", api_key="아이피 키 입력") 
llm = ChatOllama(model="llama3.1:8b")

# 텍스트 분리
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# 임베딩 모델
hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# 프롬프트 템플릿
message = """
당신은 사용자의 질문에 답변을 하는 친절한 AI 어시스턴트입니다.
당신의 입무는 주어진 문맥을 토대로 사용자 질문에 답하는 것입니다.
만약, 문맥에서 답변을 위한 정보를 찾을 수 없다면 '질문에 대한 정보를 찾을 수 없습니다' 라고 답하세요.
정보를 찾을 수 있다면 한글로 답변해 주세요.

## 주어진 문맥:
{context}

## 사용자 질문:
{input}
"""
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("human", message)
    ]
)

# 출력 파서
parser = StrOutputParser()

# 전역 변수
db = None
retriever = None
rag_chain = None

def load_pdf(file):
    global db, retriever, rag_chain

    loader = PyPDFLoader(file.name)
    docs = loader.load_and_split(text_splitter=text_splitter)

    db = FAISS.from_documents(docs, hf_embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    rag_chain = {
        "context": retriever,
        "input": RunnablePassthrough()
    } | prompt_template | llm | parser

    return "PDF 파일 업로드 완료 질문을 입력하세요."

def answer_question(question):
    if rag_chain is None:
        return "먼저 PDF 파일을 업로드하세요"
    return rag_chain.invoke(question)

with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    gr.Markdown("""
    # ✨RAG 
    **PDF 파일을 업로드하고 질문을 입력하면 AI가 답변을 제공(로컬o)**
    ###### 💬LLM 모델 : llama3.1:8b
    ###### 🔗임베딩 모델 : BAAI/bge-m3
    ###### 📈벡터DB : Chroma
    
""")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="PDF 파일 업로드")
            upload_button = gr.Button("📤 업로드 및 처리")
            status_output = gr.Textbox(label="📢 상태 메시지")

        with gr.Column(scale=2):
            question_input = gr.Textbox(label="💬 질문 입력", placeholder="궁금한 내용을 적어주세요.")
            submit_button = gr.Button("✅답변 받기")
            answer_output = gr.Textbox(label="📝 AI 답변")

    upload_button.click(load_pdf, inputs=file_input, outputs=status_output)
    submit_button.click(answer_question, inputs=question_input, outputs=answer_output)

demo.launch()