from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
import pytesseract
import streamlit as st
import time

# Global variables (API_KEYS)
API_KEY = 'replace with yours'

# 1. Function to configure paths, session_state and page layout


def config():
    # Path to be used by pdf2images
    os.environ["PATH"] += os.pathsep + \
        'C:\\Users\\hp\\Downloads\\Release-23.11.0-0\\poppler-23.11.0\\Library\\bin'

    # Path to be used by pytesseract
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

    # Configure layout
    st.set_page_config(page_title='LegalGPT', page_icon='🤖')

    # Set session_state if not present
    if 'history' not in st.session_state:
        st.session_state.history = [
            'Hi LegalGPT', 'Hello user, how are you? Please upload a document to continue to chat with me']
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

# 2.Processing UI


def process(progress_bar, text, from_percent=0, to_percent=100):
    up = low = 0
    interval = (to_percent - from_percent) / 2
    for num in range(2):
        up = round(interval*(num+1)+from_percent)
        low = round(up - interval)

        for percent_complete in range(low, up):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1, text=text)
        time.sleep(0.1)

# 3.Function to validate let user input or not


def is_disable_widget() -> bool:
    # To user input or not
    return not st.session_state.conversation or len(st.session_state.history) % 2 == 1

# 4.Function to reset conversation and history


def reset():
    st.session_state.history = [
        'Hi LegalGPT', 'Hello user, how are you? Please upload a document to continue to chat with me']
    st.session_state.conversation = None
    success = st.success("Reset Successful!")
    time.sleep(2)
    success.empty()

# 5.Function to get text from pdf


def get_pdfs_text(pdfs):
    raw_text = ""

    for pdf in pdfs:

        # Reading contents of pdf as bytes
        images = convert_from_bytes(pdf.read())

        # Get text from images in the pdf first
        for image in images:
            raw_text += pytesseract.image_to_string(image)

        # Instantiate PdfReader object from the pdf
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()

    return raw_text

# 6.Function to chunk text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    text_chunks = text_splitter.split_text(raw_text)

    return text_chunks

# 7.Function to perform text embedding and store in vector database (FAISS)


def get_vector_store(text_chunks):
    text_embeddings = OpenAIEmbeddings(api_key=API_KEY)
    vector_store = FAISS.from_texts(
        texts=text_chunks, embedding=text_embeddings)

    return vector_store

# 8.Function to establish conversation chain


def get_conversation_chain(vector_store):
    llm = ChatOpenAI(temperature=0.5, max_tokens=60, api_key=API_KEY)
    buffer_memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=buffer_memory)

    return conversation_chain

# 9.Function to process pdf uploaded


def process_pdf(pdfs):

    processing_text = 'Processing'
    progress_bar = st.progress(0, text=processing_text)
    # Get text from PDF
    process(progress_bar, processing_text, 0, 25)
    raw_text = get_pdfs_text(pdfs)

    # Return error if no text is found
    if raw_text.isspace():
        st.error(
            'No text is found in the uploded pdf(s)\nPlease upload and submit again')
        progress_bar.empty()
        # time.sleep(3)
        return

    # Get text chunks
    process(progress_bar, processing_text, 25, 50)
    text_chunks = get_text_chunks(raw_text)

    # Create vector store
    process(progress_bar, processing_text, 50, 75)
    vector_store = get_vector_store(text_chunks)

    # Create conversation chain
    # Session_state is implemented to prevent values to be recomputed every time the script is run when certain button is pressed
    st.session_state.conversation = get_conversation_chain(vector_store)
    process(progress_bar, processing_text, 75, 100)

    progress_bar.empty()
    success = st.success("Process Successful!")
    time.sleep(2)
    success.empty()
    st.rerun()

# 10.Function to update conversation history


def update_history():
    st.session_state.history.append(st.session_state.user_input)

# 11.Function to write message


def print_conversation():

    st.header(":rainbow[Chat with your PDFs]", anchor=False)

    for i, message in enumerate(st.session_state.history):
        if i % 2 == 1:
            with st.chat_message('ai', avatar='🧑‍⚖️'):
                st.write(message)
        else:
            with st.chat_message('user', avatar='😶'):
                st.write(message)

# 12. Function to send query and get response from LLM


def get_response():
    response = st.session_state.conversation(
        {'question': (st.session_state.user_input)})
    st.session_state.history.append(response['chat_history'][-1].content)
    st.rerun()

# 13.Function to setup side bar to get pdf upload


def upload_pdf():

    st.subheader('Your Pdf(s)')

    st.write('Upload Pdf(s) and hit "Submit"')

    # Upload PDF, accept multiple files, but only pdf
    pdfs = st.file_uploader(
        label="Scanned pdfs are accepted too",
        accept_multiple_files=True,
        type=['pdf'])

    if st.button(label='Submit', disabled=True if not pdfs else False, use_container_width=True):
        process_pdf(pdfs)

    if st.button(label='Reset Conversation', disabled=is_disable_widget(), use_container_width=True):

        st.warning('Are you sure? This can not be undone')

        st.button('Yes, I\'m sure', on_click=reset, use_container_width=True)

        if st.button('Cancel', use_container_width=True):
            st.rerun()

# Main function of the program


def main():

    config()

    with st.sidebar:
        upload_pdf()

    print_conversation()

    if st.chat_input(placeholder='Submit your document(s) first' if not st.session_state.conversation else 'What would you like to ask?',
                     key='user_input',
                     on_submit=update_history,
                     disabled=is_disable_widget()):
        get_response()


# Execute main if this file is directly run
if __name__ == '__main__':
    main()
