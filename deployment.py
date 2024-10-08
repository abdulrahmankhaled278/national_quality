import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from pinecone import PodSpec, Pinecone, ServerlessSpec
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import speech_recognition as sr
import whisper
import re
from google.cloud import texttospeech
import time  



@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")

if st.sidebar.button("Load whisper model"):
    st.session_state.model = load_whisper_model()
    st.success("Model loaded successfully")

def load_document(file):
    print("Loading PDF document: " + file)
    loader = PyPDFLoader(file)
    data = loader.load()
    return data

def chunk_data(data, chunk_size, chunk_overlap=0): # chunk_size = 1024 #chunk_overlap=256
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    chunks = text_splitter.split_documents(data)
    return chunks


def delete_pinecone_index(index_name='all') :
    pc = Pinecone()
    if index_name == 'all' :
        indexes = pc.list_indexes().names()
        print(f"Deleting all indexes ...")
        for index in indexes :
            pc.delete_index(index)
        print("done")
    else :
        print(f"Deleting index {index_name} ...")
        pc.delete_index(index_name)
        print("done")

def insert_or_fetch_embeddings(index_name, chunks):
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec
    pc = pinecone.Pinecone()
    
    
    
    # Use the 'text-embedding-3-large' model with 3072 dimensions
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

    # Check if the index already exists
    if index_name in pc.list_indexes().names():
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        # Create a new index with the correct dimension
        print(f'Creating index {index_name} and embeddings ...', end='')
        
        # Creating a new index with the correct dimensions for 'text-embedding-3-large'
        pc.create_index(
            name=index_name,
            dimension=3072,  # Set to the dimension size of the embedding model
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        # Process the input documents, generate embeddings, and insert them into the index
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Ok')

    return vector_store


# def get_pinecone_index_stats(index_name):
#     pc = Pinecone()
#     index = pc.Index(index_name)
#     stats = index.describe_index_stats()
#     return stats

# index_name = 'iti-indexx'
# stats = get_pinecone_index_stats(index_name)
# st.write(f"Number of vectors stored in Pinecone: {stats['total_vector_count']}")

# *********************************************************** Using chromadb ***********************************************************
def create_embeddings_chroma(chunks , persist_directory='./chromadb'):
    from langchain.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    st.write("Creating embeddings using Chroma ...")
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large',dimensions = 3072)
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    st.write("Embeddings created successfully.")
    return vector_store

def load_embeddings_chroma(persist_directory='./chromadb'):
    from langchain.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    st.write("Loading embeddings from Chroma ...")
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large',dimensions = 3072)
    vector_store = Chroma(persist_directory=persist_directory,embedding_function=embeddings)
    return vector_store

# *********************************************************** END Using chromadb ***********************************************************

def initialize_memory_and_chains(vector_store):
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

        llm = ChatOpenAI(model='gpt-4o', temperature=0.6)

        prompt = ChatPromptTemplate(
            input_variables=["content", "chat_history"],
            messages=[
                SystemMessage(content="""
                                    You are an enthusiastic and helpful customer service representative for the a software company. 
                                    Your role is to provide  information to users, as if you are speaking on behalf of the company.
                                    You should always address users' questions directly and with confidence and it's preferred to provide your answer in steps. 
                                    Avoid using phrases that carries the meaning of 'According to the text provided' or 'في السياق المقدم' . 
                                    provide information without providing anything about the date of your latest knowledge .
                                    Instead, answer in a way that feels personal, supportive, and informative, as if you have a deep knowledge of company's services, programs, and offerings. 
                                    If additional details are not provided in the context, make a helpful and educated guess based on typical company offerings or invite the user to provide more specifics so you can assist further. 
                                    Your goal is to ensure that the user feels well-informed and supported at all times.
                                    """),
                
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{content}")
            ]
        )

        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

        st.session_state.crc_document = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            chain_type='stuff',
            memory=st.session_state.memory,
            verbose=False
        )

        st.session_state.chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=st.session_state.memory,
            verbose=False
        )

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_answer(text) :
    return re.sub(r'#', '', text)

def ask_context_aware(vector_store, q, k=3):
    initialize_memory_and_chains(vector_store)
    
    lack_of_knowledge_phrases = ["I don't have","I don't know","لا أعلم", "ليس لدي", "آسف", "لا أملك", "لا اعرف","لا استطيع", "عذرًا", "لا تتوفر"]
    
    chain = st.session_state.chain
    crc_document = st.session_state.crc_document

    similarity_score = vector_store.similarity_search_with_score(q)[0][1]
    print(f"Similarity score: {similarity_score}")
    
    def invoke_llm_directly(query):
        llm = ChatOpenAI(model='gpt-4o', temperature=1)
        return llm.invoke(query).content

    # Record the start time
    start_time = time.time()
    
    if similarity_score < 0.3:
        st.write("No similar documents found, invoking LLM directly.")
        response = chain.invoke({'content': q})
        response_text = response['text']
        
        if any(phrase in response_text for phrase in lack_of_knowledge_phrases):
            st.write("Lack of Knowledge detected, invoking LLM directly.")
            # Record the end time just before starting the response
            end_time = time.time()
            response_time = end_time - start_time
            st.session_state.response_time = response_time
            return invoke_llm_directly(q)
        else:
            print(response)
            end_time = time.time()
            response_time = end_time - start_time
            st.session_state.response_time = response_time
            return response_text
    else:
        st.write("Similar documents found, invoking CRC.")
        response = crc_document.invoke({'question': q})
        response_answer = response['answer']
        
        if any(phrase in response_answer for phrase in lack_of_knowledge_phrases):
            st.write("Lack of Knowledge detected, invoking LLM directly.")
            end_time = time.time()
            response_time = end_time - start_time
            st.session_state.response_time = response_time
            return invoke_llm_directly(q)
        else:
            print("No Lack of Knowledge detected")
            print(response)
            end_time = time.time()
            response_time = end_time - start_time
            st.session_state.response_time = response_time
            return response_answer

import os

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:\downloads\zeta-sky-437913-d2-88a8ecd24774.json"

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

def transcribe_model_selection_v2(project_id: str, model: str, audio_file: str, language: str) -> tuple:
    """Transcribe an audio file."""
    client = SpeechClient()

    with open(audio_file, "rb") as f:
        content = f.read()

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=[language],
        model=model,
    )

    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{project_id}/locations/global/recognizers/_",
        config=config,
        content=content,
    )

    response = client.recognize(request=request)

    highest_confidence = 0.0
    best_transcript = ""
    best_language = ""

    for result in response.results:
        transcript = result.alternatives[0].transcript
        confidence = result.alternatives[0].confidence
        detected_language = result.language_code
        print(f"Transcript: {transcript}, Confidence: {confidence}, Language: {detected_language}")
        
        if confidence > highest_confidence:
            highest_confidence = confidence
            best_transcript = transcript
            best_language = detected_language

    return best_transcript, best_language, highest_confidence

def transcribe_audio_file(project_id, audio_file):
    # Try transcribing with English
    model = "latest_long"
    highest_confidence = 0.0
    best_transcript = ""
    best_language = ""

    try:
        transcript, language, confidence = transcribe_model_selection_v2(project_id, model, audio_file, "en-US")
        if confidence > highest_confidence:
            highest_confidence = confidence
            best_transcript = transcript
            best_language = language
    except Exception as e:
        print(f"Error transcribing English: {e}")

    # Try transcribing with Arabic
    try:
        transcript, language, confidence = transcribe_model_selection_v2(project_id, model, audio_file, "ar-EG")
        if confidence > highest_confidence:
            highest_confidence = confidence
            best_transcript = transcript
            best_language = language
    except Exception as e:
        print(f"Error transcribing Arabic: {e}")

    return best_transcript, best_language


# Text to Speech Stage 

from google.cloud import texttospeech

def synthesize_speech(text, language_code, gender, output_file):
    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Build the voice request
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code, ssml_gender=gender
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # The response's audio_content is binary
    with open(output_file, "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        print(f'Audio content written to file "{output_file}"')



def transcribe_audio(file_path):
    model = st.session_state.get('model')
    if not model:
        st.error("The model is not loaded yet. Please load the model first.")
        return "", ""
    result = model.transcribe(file_path)
    if result['language'] == 'en':
        result['text'] = str.lower(result['text'])
        
    cleaned_text = clean_text(result['text'])
    return cleaned_text, result['language']

def record_and_save_audio(file_path):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Speak now...")
        audio = r.listen(source, timeout=5, phrase_time_limit=10)
        st.info("Recording complete.")

        with open(file_path, "wb") as f:
            f.write(audio.get_wav_data())
        print("Audio saved as " + file_path)
        




if __name__ == '__main__':
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
    
    st.markdown("""
        <div class="center-container logo-container">
        <h1 >Welcome to National Quality</h1>
        <img src="https://i.postimg.cc/90MjR8Gv/National.gif" alt="National Quality Logo" width="300">
        </div>
        """, unsafe_allow_html=True)
    
    with st.sidebar:
        api_key = st.text_input('OpenAI API Key: ', type='password')
        pinecone_api_key = st.text_input('Pinecone API Key: ', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        if pinecone_api_key:
            os.environ['PINECONE_API_KEY'] = pinecone_api_key

        uploaded_file = st.file_uploader('Upload a file: ', type=['pdf'])
        chunk_size = st.number_input('Chunk size', min_value=100, max_value=2048, value=256)
        k = st.number_input('k', min_value=1, max_value=20, value=3)
        add_data = st.button('Add Data')
        
        if uploaded_file and add_data:
            with st.spinner('Reading, Chunking and Embedding the data'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size)
                st.write(f'Chunksize: {chunk_size}, Number of chunks: {len(chunks)}')
                delete_pinecone_index()
                vector_store = insert_or_fetch_embeddings('iti-index', chunks)
                # vector_store = load_embeddings_chroma()
                st.session_state.vs = vector_store

                st.success('File is Uploaded, Chunked and Embedded successfully')

    if st.button('Use Microphone'):
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            audio_file_name = "user_input.wav"
            audio_file_path = os.path.join('./', audio_file_name)  # Relative path from where Streamlit is running
            record_and_save_audio(audio_file_path)
            project_id = "zeta-sky-437913-d2"
            q, language = transcribe_audio_file(project_id, audio_file_path)
            st.session_state.language = language
            st.write(f"Transcribed question: {q} (Language: {language})")
            answer = clean_answer(ask_context_aware(vector_store, q, k))
            st.session_state.answer = answer
            response_time = st.session_state.response_time
            st.text_area('Answer:', value=answer)

            # Update chat history
            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {q}\nA: {answer}\nResponse Time: {response_time:.2f} seconds'
            st.session_state.history = f'{value}\n{"-"*100}\n{st.session_state.history}'
            h = st.session_state.history
            st.text_area(label='Chat History', value=h, key='history', height=400)
            
            # Text-to-Speech output


    q = st.text_input('Ask a question: ')
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = clean_answer(ask_context_aware(vector_store, q, k))
            # answer = st.session_state.answer
            response_time = st.session_state.response_time
            st.text_area('Answer:', value=answer)

            # Update chat history
            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {q}\nA: {answer}\nResponse Time: {response_time:.2f} seconds'
            st.session_state.history = f'{value}\n{"-"*100}\n{st.session_state.history}'
            h = st.session_state.history
            st.text_area(label='Chat History', value=h, key='history', height=400)
            
    if st.button('Generate Speech'):
        if 'vs' in st.session_state:
            answer = st.session_state.answer
            language = st.session_state.language
            tts_output_file = "D:\or_output.mp3"
            st.write("Generating speech output...")
            synthesize_speech(text=answer, language_code=language, gender=texttospeech.SsmlVoiceGender.NEUTRAL, output_file=tts_output_file)
            st.write("Speech output generated.")

            if os.path.exists(tts_output_file):
                st.audio(tts_output_file, format="audio/mp3")
                
            if 'history' not in st.session_state:
                st.session_state.history = ''
            h = st.session_state.history
            st.text_area(label='Chat History', value=h, key='history', height=400)
            


