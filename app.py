import streamlit as st 
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai
import time
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import os

def main():
    # Page configuration
    st.set_page_config(
        page_title="Medical Video Analysis",
    )

    st.title("Medical Video Analysis")
    st.write("This app utilizes artificial intelligence to assist healthcare professionals and researchers in analyzing medical videos and gaining insights. With integrated Agentic AI and web search features, it supports learning and informed decision-making in healthcare.")

    # Input API key through Streamlit UI
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    # Configure the API key
    genai.configure(api_key=api_key)

    @st.cache_resource
    def initialize_agent():
        return Agent(
            name="Video AI Summarizer",
            model=Gemini(id="gemini-2.0-flash-exp"),
            tools=[DuckDuckGo()],
            markdown=True,
        )

    # Initialize the agent
    try:
        multimodal_Agent = initialize_agent()
    except Exception as error:
        st.error("Failed to initialize the AI agent. Please check your API Key.")
        st.stop()

    # File uploader
    video_file = st.sidebar.file_uploader(
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
                    error_message = str(error)
                    if "API_KEY_INVALID" in error_message:
                        st.error("The provided API key is invalid. Please check and enter a valid key.")
                    else:
                        st.error("An unexpected error occurred during analysis. Please try again later.")
                        st.write(error)
                finally:
                    # Clean up temporary video file
                    Path(video_path).unlink(missing_ok=True)
    else:
        st.info("Please upload a video file to proceed.")

# Run the app
if __name__ == "__main__":
    main()

