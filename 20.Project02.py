from langchain_ollama import ChatOllama
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
import gradio as gr

# 컴포넌트 초기화
llm = ChatOllama(model="llama3.1:8b")
hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

prompt_template = ChatPromptTemplate.from_template("""
<context>{context}</context>
<question>{input}</question>
당신은 사용자의 질문에 답변을 하는 친절한 AI 어시스턴트입니다.
당신의 입무는 주어진 문맥을 토대로 사용자 질문에 답하는 것입니다.
만약, 문맥에서 답변을 위한 정보를 찾을 수 없다면 '질문에 대한 정보를 찾을 수 없습니다' 라고 답하세요.
정보를 찾을 수 있다면 한글로 답변해 주세요.
""")

# 전역 상태 변수
db, retriever, rag_chain = None, None, None

def load_pdf(file):
    global db, retriever, rag_chain
    try:
        loader = PyPDFLoader(file.name)
        docs = loader.load_and_split(text_splitter)
        db = Chroma.from_documents(docs, hf_embeddings, collection_name="temp_collection")
        retriever = db.as_retriever(search_kwargs={"k": 3})
        rag_chain = (
            {"context": retriever, "input": RunnablePassthrough()} 
            | prompt_template 
            | llm 
            | StrOutputParser()
        )
        return "✅ 문서 처리 완료! 질문을 입력하세요."
    except Exception as e:
        return f"❌ PDF 파일 처리 오류: {str(e)}"

def respond(message, chat_history):
    if not rag_chain:
        return "📌 먼저 PDF 파일을 업로드해 주세요", chat_history
    try:
        response = rag_chain.invoke(message)
        chat_history.append((message, response))
        return "", chat_history
    except Exception as e:
        return f"❌ 답변 생성 오류: {str(e)}", chat_history

def clear_all():
    global db, retriever, rag_chain
    if db is not None:
        try:
            db._client.delete_collection(db._collection.name)
        except Exception as e:
            print(f"컬렉션 삭제 오류: {e}")
    db, retriever, rag_chain = None, None, None
    return None, "", []

with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    gr.Markdown("""
    # ✨RAG PDF 기반 AI 챗봇 (로컬)
    **PDF 파일을 업로드하고 질문을 입력하면 AI가 답변을 제공**
    ###### 🌐웹 인터페이스 : Gradio          
    ###### 💬LLM 모델 : llama3.1:8b
    ###### 🔗임베딩 모델 : BAAI/bge-m3
    ###### 📈벡터DB : Chroma
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="📄 PDF 업로드", file_types=[".pdf"])
            with gr.Row():
                upload_btn = gr.Button("📤 업로드", variant="primary")
                clear_btn = gr.Button("🔄 초기화", variant="secondary")
            status = gr.Textbox(label="🔔 시스템 상태", interactive=False)
        
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(label="💬 메시지 입력", placeholder="PDF 내용에 대해 질문하세요...")
            submit_btn = gr.Button("📨 전송", variant="primary")

    # 이벤트 핸들러
    upload_btn.click(load_pdf, inputs=file_input, outputs=status)
    clear_btn.click(clear_all, outputs=[file_input, status, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])

demo.launch(inbrowser=True)