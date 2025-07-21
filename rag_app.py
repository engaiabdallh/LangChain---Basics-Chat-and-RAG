import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import tiktoken


# Load .env file
_ = load_dotenv(override=True)


# Loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):    
    try:
        # Get the extension of the attached file
        _, extension = os.path.splitext(file)

        if extension == '.pdf':
            print(f'Loading {file}')
            loader = PyPDFLoader(file)
        elif extension == '.docx':
            print(f'Loading {file}')
            loader = Docx2txtLoader(file)
        elif extension == '.txt':
            loader = TextLoader(file)
        else:
            print('Document format is not supported!')
            return None

        # Load the file
        data = loader.load()
        return data
    except Exception as e:
        print(f"Error loading document: {str(e)}")
        return None


def chunk_data(data, chunk_size=512, chunk_overlap=50):
    # Used RecursiveCharacterTextSplitter with overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(data)
    return chunks


def create_embeddings(chunks, persist_directory='./assets/chroma_db'):
    try:
        # Create the directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # embeddings model
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  

        # Vector store using chroma with persistence
        vector_store = Chroma.from_documents(
            chunks, 
            embeddings,
            persist_directory=persist_directory
        )
        
        # Explicitly persist the vector store
        vector_store.persist()
        
        return vector_store
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")
        return None


def load_embeddings(persist_directory='./assets/chroma_db'):
    try:
        # Check if the directory exists
        if not os.path.exists(persist_directory):
            print(f"Directory {persist_directory} does not exist.")
            return None
            
        # embeddings model
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  

        # Load the vector store
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        return vector_store
    except Exception as e:
        print(f"Error loading embeddings: {str(e)}")
        return None


def questions_answering(vector_store, question: str, k=5, temperature=0.2):
    try:
        # Define a custom prompt template for better responses
        template = """
        Answer the question based only on the following context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
        # LLM with appropriate temperature
        llm = ChatOpenAI(model='gpt-4o-mini', temperature=temperature)
        
        # Retriever
        retriever = vector_store.as_retriever(
            search_type='similarity', 
            search_kwargs={'k': k}
        )
        
        # Chain 
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", 
            retriever=retriever,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        
        # Invoke question into the chain
        answer = chain.invoke(question)
        return answer['result']
    except Exception as e:
        print(f"Error in question answering: {str(e)}")
        return f"Error: {str(e)}"


def conversation_answering(vector_store, question: str, k=5, temperature=0.2):
    try:
        # Get or create conversation chain from session state
        if 'conversation' not in st.session_state:
            # Define custom prompt template for better responses
            QA_PROMPT = PromptTemplate.from_template("""
            Answer the question based only on the following context:
            {context}

            Question: {question}

            Provide a concise and accurate answer in English. If the information is not in the context, 
            state "I don't have enough information to answer this question."
            """)
            
            # LLM with appropriate temperature
            llm = ChatOpenAI(model='gpt-4o-mini', temperature=temperature)
            
            # Retriever
            retriever = vector_store.as_retriever(
                search_type='similarity', 
                search_kwargs={'k': k}
            )
            
            # Memory
            memory = ConversationBufferMemory(
                memory_key='chat_history',
                output_key='answer',
                return_messages=True
            )
            
            # Conversational chain with custom prompt
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                chain_type="stuff",
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": QA_PROMPT}
            )
            
            st.session_state.conversation = {
                "chain": chain,
                "memory": memory
            }
        
        # Get the conversation from session state
        conversation = st.session_state.conversation
        
        # Invoke question into the chain
        start_time = time.time()
        response = conversation["chain"].invoke({'question': question})
        end_time = time.time()
        
        # Add processing time to response
        response['processing_time'] = end_time - start_time
        
        return response
    except Exception as e:
        print(f"Error in conversation answering: {str(e)}")
        return {"answer": f"Error: {str(e)}"}


def get_embeddings_cost(texts):
    # Using 3-small model
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens * 0.020 / 1000000


# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']
    if 'conversation' in st.session_state:
        del st.session_state['conversation']
    st.success("Conversation memory cleared!")


def main():
    st.image('./assets/img.png')
    st.subheader('LLM Question-Answering with Memory 🤖')

    # sidebar
    with st.sidebar:
        api_key = st.text_input('OpenAI API Key', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        # File Uploader
        uploaded_file = st.file_uploader('Upload a file', type=['pdf', 'docx', 'txt'])

        # Some configs
        chunk_size = st.number_input('Chunk size', min_value=100, max_value=2048, value=512, on_change=clear_history)
        chunk_overlap = st.number_input('Chunk overlap', min_value=0, max_value=chunk_size//2, value=50, on_change=clear_history)
        k = st.number_input('Number of chunks to retrieve (k)', min_value=1, max_value=20, value=5, on_change=clear_history)
        temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, value=0.3, step=0.1, on_change=clear_history)
        
        # Memory mode toggle
        memory_mode = st.checkbox('Enable conversation memory', value=True)
        
        # Options for loading existing database
        st.divider()
        st.subheader("Database Options")
        load_existing_db = st.checkbox("Load existing database", value=False)
        
        if load_existing_db:
            if st.button("Load Existing Database"):
                with st.spinner("Loading existing database..."):
                    vector_store = load_embeddings('./assets/chroma_db')
                    if vector_store:
                        st.session_state.vs = vector_store
                        st.success("Database loaded successfully!")
                    else:
                        st.error("Failed to load database.")

        # Add data button
        st.divider()
        add_data = st.button('Process New Document', on_click=clear_history)
        
        # Clear memory button
        st.button('Clear Conversation Memory', on_click=clear_history)

        # if the user uploaded data
        if uploaded_file and add_data:
            with st.spinner('Reading, Chunking, and Embedding your file ..'):
                # Ensure assets directory exists
                os.makedirs('./assets', exist_ok=True)
                
                # Save the uploaded file to assets directory
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./assets', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)
                
                # Load the file
                data = load_document(file=file_name)
                if data:
                    chunks = chunk_data(data=data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    st.write(f'Chunk Size: {chunk_size}, Overlap: {chunk_overlap}, Chunks: {len(chunks)}')

                    # Get the cost of embeddings
                    total_tokens, cost = get_embeddings_cost(texts=chunks)
                    st.write(f'Embedding cost: ${cost:.4f}')

                    # Creating the embeddings
                    vector_store = create_embeddings(chunks)

                    # Saving the vector store in the streamlit session state (to be persistent between reruns)
                    st.session_state.vs = vector_store
                    st.success('File uploaded, chunked and embedded successfully.')
                else:
                    st.error("Failed to load document.")

    # Main area
    # Create containers for different sections
    question_container = st.container()
    answer_container = st.container()
    history_container = st.container()

    # Create history variable if not existed
    if 'history' not in st.session_state:
        st.session_state.history = []

    # User Question
    with question_container:
        q = st.text_input('Ask a question related to the content of your uploaded file:')
        
    if q:
        standard_answer = "Answer only based on the text you received as input. Don't search external resources."
        q_with_instructions = f'{q} {standard_answer}'

        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            
            with answer_container:
                st.write(f'Retrieving {k} most relevant chunks with temperature {temperature}')
                
                with st.spinner('Thinking...'):
                    # Use memory mode or standard QA based on checkbox
                    if memory_mode:
                        response = conversation_answering(
                            vector_store=vector_store, 
                            question=q_with_instructions, 
                            k=k,
                            temperature=temperature
                        )
                        answer = response.get("answer", "No answer generated")
                        processing_time = response.get("processing_time", 0)
                        
                        # Get source documents if available
                        sources = response.get("source_documents", [])
                        num_sources = len(sources) if sources else 0
                        
                        # Display source info
                        st.write(f"Retrieved {num_sources} sources in {processing_time:.2f} seconds")
                    else:
                        # Standard QA without memory
                        answer = questions_answering(
                            vector_store=vector_store, 
                            question=q_with_instructions, 
                            k=k,
                            temperature=temperature
                        )

                # Answer
                st.text_area('Answer:', value=answer, height=150)
                
                # Show sources option
                if memory_mode and "sources" in locals() and sources and st.checkbox("Show source documents"):
                    st.subheader("Source Documents")
                    for i, doc in enumerate(sources):
                        with st.expander(f"Source {i+1}"):
                            st.markdown(f"**Content:** {doc.page_content}")
                            if hasattr(doc, 'metadata'):
                                st.markdown(f"**Metadata:** {doc.metadata}")
                
            # Add to history
            timestamp = time.strftime("%H:%M:%S")
            st.session_state.history.append({"role": "user", "content": q, "time": timestamp})
            st.session_state.history.append({"role": "assistant", "content": answer, "time": timestamp})
            
            # Display history
            with history_container:
                st.subheader('Conversation History')
                
                # Reversed to show newest messages at the bottom
                for message in st.session_state.history:
                    if message["role"] == "user":
                        st.markdown(f"**You ({message['time']}):** {message['content']}")
                    else:
                        st.markdown(f"**Assistant ({message['time']}):** {message['content']}")
                    st.divider()
        else:
            st.warning("Please upload and process a document first or load an existing database.")


if __name__ == '__main__':
    main()