import tempfile
from pathlib import Path

import streamlit as st

from qa_system import DocumentQA, describe_source

st.set_page_config(page_title="Document Q&A", page_icon="📄")


@st.cache_resource(show_spinner="Loading search models...")
def load_qa():
    """build this once. streamlit reruns the whole file on every click."""
    return DocumentQA()


def index_upload(qa, upload):
    """save the upload to a temporary file, since the loaders read from disk."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / Path(upload.name).name
        path.write_bytes(upload.getvalue())
        return qa.add_document(path, name=upload.name)


qa = load_qa()
st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)

with st.sidebar:
    st.header("Your documents")

    # changing the key empties the uploader, so a file isn't indexed again on the next rerun
    uploads = st.file_uploader(
        "Add PDF or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploads:
        for upload in uploads:
            with st.spinner(f"Indexing {upload.name}..."):
                try:
                    index_upload(qa, upload)
                except Exception as error:
                    st.session_state.upload_error = f"{upload.name}: {error}"
        st.session_state.uploader_key += 1
        st.rerun()

    if "upload_error" in st.session_state:
        st.error(st.session_state.pop("upload_error"))

    documents = qa.list_documents()
    for name in documents:
        st.write(f"📄 {name}")

    if documents and st.button("Remove all documents"):
        qa.clear()
        st.session_state.messages = []
        st.rerun()

st.title("Document Q&A")
st.caption("Each question is answered on its own, so name what you mean instead of saying \"it\".")

if not documents:
    st.info("Add a document in the sidebar to get started.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for label, text in message.get("sources", []):
            with st.expander(label):
                st.text(text)

if question := st.chat_input("Ask about your documents", disabled=not documents):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your documents..."):
                stream, sources = qa.ask(question)
            answer = st.write_stream(stream)
        except Exception as error:
            st.error(
                f"Couldn't get an answer: {error}\n\n"
                "If you're using the local model, check that Ollama is running "
                "and that you've downloaded the model."
            )
            st.stop()

        if not answer:
            answer = "The model returned an empty answer. Try rephrasing the question."
            st.warning(answer)

        labels = [
            (f"[{i}] {describe_source(doc)}", doc.page_content)
            for i, doc in enumerate(sources, start=1)
        ]
        for label, text in labels:
            with st.expander(label):
                st.text(text)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": labels}
    )
