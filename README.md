# YouNote AI: AI-Powered YouTube Video Companion

YouNote AI is a Streamlit web application that enhances your YouTube video-watching experience by providing real-time chat and automated note-taking. Enter a YouTube video URL, watch the video, chat with an AI assistant about the content, and download concise Markdown notes with timestamped links for visuals.

🔗 Visit the app: https://younote-ai.streamlit.app

## Features
- **Video Streaming**: Embed and watch YouTube videos directly in the app.
- **Real-Time Chat**: Ask questions about the video content and get conversational answers based on the transcript.
- **Automated Notes**: Generate concise, structured notes with bullet points and timestamped YouTube links for visuals (e.g., diagrams, slides).
- **Downloadable Notes**: Export notes as a Markdown file for easy reference.

## Prerequisites
- Python 3.8 or higher
- A Google API key for Gemini AI (set up via Google Cloud Console)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/dsRitam/younote-ai.git
   cd younote-ai
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   - Create a `.env` file in the project root.
   - Add your Google API key:
     ```
     GOOGLE_API_KEY=your_google_api_key_here
     ```

## Usage
1. **Run the App Locally**:
   ```bash
   streamlit run streamlit_app.py
   ```
   The app will open in your default browser.

2. **Interact with the App**:
   - Enter a YouTube video URL and click **Submit**.
   - Watch the video in the main panel.
   - Use the chat section below the video to ask questions about the content.
   - View automatically generated notes in the sidebar’s scrollable box.
   - Collapse the chat or notes sections as needed.
   - Click **Download Notes as Markdown** to save the notes.
