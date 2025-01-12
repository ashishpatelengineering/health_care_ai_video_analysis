# Source: https://www.youtube.com/watch?v=Ih1LDnPijFU

import streamlit as st 
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file,get_file
import google.generativeai as genai
import time
from pathlib import Path
import tempfile
import os

# from dotenv import load_dotenv
# load_dotenv()

# API_KEY=os.getenv("GOOGLE_API_KEY")
# if API_KEY:
#     genai.configure(api_key=API_KEY)

# Page configuration
st.set_page_config(
    page_title="HealthScope AI",
    # layout="wide"
)

st.title("Health Care AI")
st.write("Health Care is an intelligent platform that empowers healthcare professionals and researchers to analyze medical videos and gain actionable insights. With integrated AI and web search capabilities, it simplifies learning and decision-making in healthcare.")


@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

## Initialize the agent
multimodal_Agent=initialize_agent()

# API Key Input
api_key = st.text_input(
    "Enter your Google API Key:",
    type="password",
    help="Provide your Google API Key to enable AI functionalities."
)

if api_key:
    genai.configure(api_key=api_key)
    st.success("API Key successfully configured!")
else:
    st.warning("Please enter a valid API Key to proceed.")

# File uploader
video_file = st.file_uploader(
    "Upload a Video File for AI Analysis:", type=['mp4', 'mov', 'avi'], help="Upload a Video File for AI Analysis"
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path, format="video/mp4", start_time=0)

    user_query = st.text_area(
        "What insights are you seeking from the video?",
        placeholder="Ask anything about the video content.",
        help="Provide specific questions or insights you want from the video."
    )

    if st.button("Analyse Video", key="analyze_video_button"):
        if not user_query:
            st.warning("Please enter a question or insight to analyze the video.")
        else:
            try:
                with st.spinner("Processing video and gathering insights..."):
                    # Upload and process video file
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    # Prompt generation for analysis
                    analysis_prompt = (
                        f"""
                        Analyze the uploaded video for content and context.
                        Respond to the following query using video insights and supplementary web research:
                        {user_query}

                        Provide a detailed, user-friendly, and actionable response.
                        """
                    )

                    # AI agent processing
                    response = multimodal_Agent.run(analysis_prompt, videos=[processed_video])

                # Display the result
                st.subheader("Analysis Result")
                st.markdown(response.content)

            except Exception as error:
                st.error(f"An error occurred during analysis: {error}")
            finally:
                # Clean up temporary video file
                Path(video_path).unlink(missing_ok=True)

# Customize text area height
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)