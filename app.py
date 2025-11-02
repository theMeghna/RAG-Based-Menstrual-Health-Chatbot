import streamlit as st
import os
from io import BytesIO
import warnings
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

# --- Core LangChain / Hugging Face Imports ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# --- Configuration ---
PREDEFINED_URLS = [
    "https://www.unicef.org/wash/menstrual-hygiene",
    "https://www.who.int/news/item/28-05-2023-menstrual-health-not-just-hygiene-the-path-toward-a-strong-cross-sectoral-response",
    "https://www.pib.gov.in/PressReleasePage.aspx?PRID=1846147",
    "https://nhm.gov.in/index1.php?lang=1&level=3&sublinkid=1021&lid=391",
    "https://www.who.int/europe/news/item/27-05-2022-education-and-provisions-for-adequate-menstrual-hygiene-management-at-school-can-prevent-adverse-health-consequences",
]

# --- Helper: Load & Process Docs ---
@st.cache_resource
def load_and_process_documents(uploaded_file, urls):
    all_documents = []

    st.sidebar.caption(f"Loading {len(urls)} web sources...")
    try:
        url_loader = UnstructuredURLLoader(urls=urls)
        all_documents.extend(url_loader.load())
    except Exception as e:
        st.sidebar.error(f"Could not load URL data: {e}")

    if uploaded_file is not None:
        st.sidebar.caption(f"Loading PDF: {uploaded_file.name}")
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        try:
            loader = PyPDFLoader(temp_path)
            all_documents.extend(loader.load())
        except Exception as e:
            st.sidebar.error(f"Error loading PDF: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if not all_documents:
        return None

    st.sidebar.caption("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(all_documents)

    st.sidebar.caption("Creating FAISS vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore


# --- Load Local Model ---
@st.cache_resource
def get_local_llm():
    try:
        st.sidebar.caption("Loading TinyLlama model (first run may take a minute)...")
        pipe = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device_map="cpu",
            max_new_tokens=256,
            temperature=0.7,
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")
        return None


# --- Translation Pipeline (fallback for non-English languages) ---
@st.cache_resource
def get_translator(target_lang):
    lang_map = {
        "Hindi": "en-hi",
        "Tamil": "en-ta",
        "Telugu": "en-te",
        "Kannada": "en-kn",
        "Bengali": "en-bn",
        "Marathi": "en-mr"
    }
    code = lang_map.get(target_lang)
    if code:
        model_name = f"Helsinki-NLP/opus-mt-{code}"
        try:
            return pipeline("translation", model=model_name)
        except Exception:
            return None
    return None


# --- Conversational Chain ---
def get_conversational_chain(vectorstore):
    llm = get_local_llm()
    if llm is None:
        return None

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain


# --- Streamlit App UI ---
st.set_page_config(page_title="LUNA: Menstrual Health Educator", page_icon="🩸", layout="wide")

st.title("🩸 LUNA: RAG Menstrual Health Educator")
st.markdown("""
<div style="text-align: center; background-color:#ffe8f0; padding:10px; border-radius:10px;">
I'm Luna 🌙, your AI menstrual health educator. I combine trusted web and PDF sources to answer your questions.
</div>
""", unsafe_allow_html=True)

st.sidebar.title("📚 Knowledge Base Builder")
st.sidebar.subheader("🌐 Choose Response Language")
language = st.sidebar.selectbox(
    "Language:",
    ["English", "Hindi", "Tamil", "Telugu", "Kannada", "Bengali", "Marathi"]
)

uploaded_file = st.sidebar.file_uploader("Upload Educational PDF:", type="pdf")

with st.spinner("Initializing knowledge base..."):
    st.session_state.vectorstore = load_and_process_documents(uploaded_file, PREDEFINED_URLS)

if st.session_state.vectorstore:
    st.sidebar.success("Knowledge base ready ✅")
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = get_conversational_chain(st.session_state.vectorstore)
else:
    st.sidebar.error("Failed to initialize. Please retry.")

# --- Chat Memory ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! Ask me anything about menstrual health."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --- Chat Handler ---
def handle_user_input(user_prompt):
    if "conversation_chain" not in st.session_state or st.session_state.conversation_chain is None:
        st.error("Chatbot not ready yet.")
        return

    lang_instruction = f"Please answer in {language}. Use clear, educational, and polite tone."
    full_prompt = f"{lang_instruction}\nQuestion: {user_prompt}"

    st.session_state.messages.append({"role": "user", "content": f"(in {language}) {user_prompt}"})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Luna is answering in {language}..."):
            try:
                result = st.session_state.conversation_chain.invoke({"question": full_prompt})
                ai_response = result.get("answer", "Sorry, I couldn’t find that information.")

                # --- Translate if not English ---
                if language != "English":
                    translator = get_translator(language)
                    if translator:
                        translated = translator(ai_response)[0]['translation_text']
                        ai_response = translated

                disclaimer = "\n\n**Disclaimer:** I'm an educational AI, not a medical professional. For health issues, consult a doctor."
                final = ai_response + disclaimer
                st.markdown(final)
                st.session_state.messages.append({"role": "assistant", "content": final})
            except Exception as e:
                st.error(f"Error: {e}")


if prompt := st.chat_input("Ask about menstrual health..."):
    if st.session_state.vectorstore:
        handle_user_input(prompt)
    else:
        st.warning("Please wait for initialization or upload a PDF.")

