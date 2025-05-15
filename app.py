import streamlit as st
import time
from functions import extract_text_from_pdf, create_vector_store, conversation_answering

st.set_page_config(page_title="🎓 Academic PDF Converser ✨", page_icon="🎓", layout="wide")
GEMINI_MODEL_ID = "gemini-1.5-flash-latest"

# Sidebar - API Key Input
with st.sidebar:
    st.image("https://python.langchain.com/assets/images/langchain_logo_text-92988aaf1943a679109696001030dd13.svg", width=200)
    GEMINI_API_KEY = st.text_input("Enter Google Gemini API Key:", type="password")
    if not GEMINI_API_KEY:
        st.warning("⚠️ Please enter your Gemini API Key.")
    st.markdown("---")

# Session State Initialization
st.session_state.setdefault("vector_store", None)
st.session_state.setdefault("chat_history_display", [])
st.session_state.setdefault("current_pdf_name", None)
st.session_state.setdefault("conversation_object", None)

# Layout
col1, col2 = st.columns([0.4, 0.6])

# PDF Upload Section
with col1:
    st.markdown("#### 📤 Upload PDF")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file and uploaded_file.name != st.session_state.current_pdf_name:
        st.session_state.current_pdf_name = uploaded_file.name
        st.session_state.chat_history_display = []
        st.session_state.conversation_object = None

        with st.spinner("Processing PDF..."):
            try:
                text = extract_text_from_pdf(uploaded_file.getvalue())
                st.session_state.vector_store = create_vector_store(text)
                st.success("✅ PDF processed successfully!")
            except Exception as e:
                st.error(f"❌ {e}")

# Chat Interface
with col2:
    st.markdown("#### 💬 Ask your PDF")

    chat_box = st.container(height=500)
    for msg in st.session_state.chat_history_display:
        with chat_box:
            with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🔬"):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("source_documents"):
                    with st.expander("Sources"):
                        for i, doc in enumerate(msg["source_documents"]):
                            st.caption(f"Source {i+1}:")
                            st.markdown(f"> {doc.page_content}")

    # Input Prompt
    if prompt := st.chat_input("Ask a question...", disabled=not GEMINI_API_KEY or not st.session_state.vector_store):
        st.session_state.chat_history_display.append({"role": "user", "content": prompt})
        with chat_box:
            with st.chat_message("user", avatar="🧑‍🎓"): st.markdown(prompt)

        with chat_box:
            with st.chat_message("assistant", avatar="🤖"):
                placeholder = st.empty()
                for i in range(10): placeholder.markdown("thinking"[:i % 9] + "▌"); time.sleep(0.03)

                result, updated_obj = conversation_answering(
                    vector_store=st.session_state.vector_store,
                    question=prompt,
                    api_key=GEMINI_API_KEY,
                    conversation_obj=st.session_state.conversation_object
                )

                st.session_state.conversation_object = updated_obj
                placeholder.markdown(result["answer"])

                st.session_state.chat_history_display.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "source_documents": result.get("source_documents", [])
                })
