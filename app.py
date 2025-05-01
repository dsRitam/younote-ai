from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import streamlit as st
import re
import io
import markdown

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="YouTube AI Chat & Note Maker", layout="wide")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "notes" not in st.session_state:
    st.session_state.notes = ""
if "video_id" not in st.session_state:
    st.session_state.video_id = ""
if "show_notes" not in st.session_state:
    st.session_state.show_notes = True
if "show_chat" not in st.session_state:
    st.session_state.show_chat = True

# Function
def combine_retrieved_chunks(four_chunks_from_retriever):
    return "\n".join(chunk.page_content for chunk in four_chunks_from_retriever)

# App layout
st.title("💬 YouNote AI: AI-Powered Video Companion")
st.subheader("Watch YouTube videos, chat in real-time, and take notes effortlessly!")

# For note box
st.markdown("""
    <style>
    .scrollable-notes {
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #ccc;
        padding: 10px;
        background-color: #f9f9f9;
        color: black !important;
    }
    /* Increase sidebar width */
    [data-testid="stSidebar"] {
        width: 400px !important;
    }
    [data-testid="stSidebar"] .st-expander {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Youtube URL
with st.form(key="url_form"):
    url = st.text_input("Enter YouTube Video URL:")
    submit_button = st.form_submit_button(label="Submit")
    
    if submit_button and url:
        pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
        match = re.search(pattern, url)
        if match:
            st.session_state.video_id = match.group(1)
        else:
            st.session_state.video_id = ""
            st.error("Invalid YouTube URL!")
    elif submit_button and not url:
        st.session_state.video_id = ""
        st.error("Please enter a YouTube URL!")

# LOGIC
if st.session_state.video_id:
    video_id = st.session_state.video_id
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_string = ""

        for chunk in transcript:
            transcript_string += chunk['text'] + " "
        transcript_string = transcript_string.strip()

        # SPLITTING --->
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript_string])

        # EMBEDDING & STORING --->
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embedding)

        # MODEL --->
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

        # PROMPTS --->
        note_prompt_visual = PromptTemplate(
            template='''
                You are a student taking notes during a lecture. 
                Summarize the following transcript chunk into concise bullet points under a relevant subheading (e.g., "Introduction", "Key Concepts"). 
                Capture key concepts, definitions, codes, formulas, derivations, or examples if present. 
                Since this chunk mentions visuals (e.g., 'diagram', 'slide', 'chart'), include a snapshot placeholder with an estimated timestamp and an embedded YouTube link.
                You can use emojis if you want.

                Transcript chunk: {context}

                Output format:
                - Subheading (e.g., Introduction)
                - Point 1
                - Point 2
                - Point 3
                - [Snapshot: Description of visual at {timestamp}](https://www.youtube.com/watch?v={video_id}&t={seconds}s)
            ''',
            input_variables=['context', 'video_id', 'timestamp', 'seconds']
        )

        note_prompt_non_visual = PromptTemplate(
            template='''
                You are a student taking notes during a lecture. 
                Summarize the following transcript chunk into concise bullet points under a relevant subheading (e.g., "Introduction", "Key Concepts"). 
                Capture key concepts, definitions, codes, formulas, derivations, or examples if present.
                You can use emojis if you want.

                Transcript chunk: {context}

                Output format:
                - Subheading (e.g., Introduction)
                - Point 1
                - Point 2
                - Point 3
            ''',
            input_variables=['context']
        )

        chat_prompt = PromptTemplate(
            template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context in a friendly, conversational tone.
                If the context is insufficient, say you don't know.
                You can use emojis if you want.

                {context}
                Question: {question}
            """,
            input_variables=['context', 'question']
        )

        
        # RETRIEVER --->
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # GENERATE NOTES --->
        notes = []
        visual_keyword = ["diagram", "slide", "chart", "graph", "image", "visual"]

        for chunk in chunks:
            chunk_content = chunk.page_content
            has_visual = any(tag in chunk_content.lower() for tag in visual_keyword) 
            
            start_time = 0
            for t in transcript:
                if t["text"] in chunk_content:
                    start_time = t["start"]
                    break

            seconds = int(start_time)
            timestamp = f"{int(start_time // 60)}:{int(start_time % 60):02d}"

            if has_visual:
                chunk_note = llm.invoke(note_prompt_visual.format(context=chunk_content, video_id=video_id, timestamp=timestamp, seconds=seconds)).content
            else:
                chunk_note = llm.invoke(note_prompt_non_visual.format(context=chunk_content)).content

            notes.append(chunk_note)

        total_notes = "# Lecture Notes\n\n" + "\n".join(notes)
        st.session_state.notes = total_notes
        st.session_state.show_notes = True

        # Layout : Video-Chat & Notes

        col1 = st.container()
        with col1:
            # Streaming YouTube video
            st.markdown(f'<iframe width="100%" height="400" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>', unsafe_allow_html=True)

            # Chat Box (collapse option enabled)
            with st.expander("Chat with Gimi", expanded=st.session_state.show_chat):
                query = st.text_input("Ask me anything about the video", key="chat_input")
                
                if query:
                    four_chunks_from_retriever = retriever.invoke(query)
                    combine_four_chunks = combine_retrieved_chunks(four_chunks_from_retriever)
                    final_prompt = chat_prompt.invoke({"context": combine_four_chunks, "question": query})
                    answer = llm.invoke(final_prompt).content
                    st.session_state.chat_history.append({"question": query, "answer": answer})

                # Chat history display
                for chat in st.session_state.chat_history:
                    st.write(f" **You:** {chat['question']}")
                    st.write(f" **Gimi:** {chat['answer']}")

        with st.sidebar:
            with st.expander("Lecture Notes 📝", expanded=True):

                # Download notes
                if st.session_state.notes:
                    buffer = io.StringIO()
                    buffer.write(total_notes)
                    buffer.seek(0)
                    st.download_button(
                        label="Download Notes as Markdown",
                        data=buffer.getvalue(),
                        file_name="lecture_notes.md",
                        mime="text/markdown",
                        key="download_notes"
                    )


                # Notes display
                if st.session_state.notes:
                    html_notes = markdown.markdown(total_notes)
                    # st.markdown(html_notes, unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="scrollable-notes">{html_notes}</div>',
                        unsafe_allow_html=True
                    )


    except TranscriptsDisabled:
        st.error("Transcript Not Available!")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")