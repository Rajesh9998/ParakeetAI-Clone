# ParakeetAI - Live  Question Answering Tool

>ParakeetAI is a versatile web application designed to capture live audio from your browser tab and microphone. This tool goes beyond basic transcription by integrating question answering and dialog analysis. It utilizes Deepgram for real-time transcription, Groq for intelligent question routing and answering, Pinecone for RAG capabilities and DuckDuckGo for up-to-date information.

## Here’s a quick demo:

[![Watch the video](https://img.youtube.com/vi/your_video_id/maxresdefault.jpg)](https://github.com/user-attachments/assets/f654bd0b-b6b1-4007-bbe5-22ce8edbc083)

## Features

-   **Live Transcription:** Transcribes audio in real-time from both your browser tab and microphone, displaying the text as it's spoken.
-   **Intelligent Question Answering:**
    -   **Context-Aware Question Generation:** Uses the transcribed dialogue to generate relevant questions.
    -   **Adaptive Answering:** Routes queries to different models based on whether it requires RAG tool, search tool or general model.
-  **Comprehensive Dialog Analysis:** Analyzes conversations from both the browser tab and the microphone, presenting a formatted dialogue.
-   **User-Friendly Interface:** Provides clear controls to start/stop transcription and clear the current transcript.

## Tech Stack

-   **Frontend:** HTML, CSS, JavaScript
-   **Backend:** Python (Flask, aiohttp)
-   **Transcription:** Deepgram
-   **AI/ML Models:** Groq, Pinecone
-   **Search:** DuckDuckGo

## Getting Started

### Prerequisites

-   Python 3.11
-   Deepgram API key
-   Groq API key
-   Pinecone API key
-   Internet connection

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/Rajesh9998/ParakeetAI-Clone.git
    cd ParakeetAI-Clone
    ```
2.  Install backend dependencies:

    ```bash
    pip install -r requirements.txt
    ```
3.  Create a `.env` file and add the following, replacing the placeholders with your actual API keys:

    ```env
    DEEPGRAM_API_KEY=YOUR_DEEPGRAM_API_KEY
    GROQ_API_KEY=YOUR_GROQ_API_KEY
    PINECONE_API_KEY=YOUR_PINECONE_API_KEY
    ```
4.  Set `YOUR_PINECONE_RAG_ASSISTANT_NAME` in `test4.py`.
5.  Run the application:

    ```bash
    python streamlit_app.py
    ```
    This will start the Flask server on port 5555 and aiohttp server on port 5556.

6. Open your web browser and go to  `http://localhost:5555`
7. Edit the name of Pinecone RAG Assistant name in tesr4.py.
8. Also make sure to replace  the Deepgram_API_key in indext.html at line 163 accordingly.

## Usage

1.  Click the "Start Sharing & Transcription" button to begin screen and audio capture.
2.  Speak clearly into your microphone to capture your audio.
3.  The transcriptions for both tab and microphone audio appear in the respective boxes.
4.  Click the "Generate" button to generate a question and answer based on the transcriptions.
5.  Click the "Clear" button to clear current transcriptions.
6.  Click the "Analyze" button to get conversation dialogue from both the transcriptions.
7.  Click the "Stop Transcription" button to end the transcription process.
8.  A streamlit Version of the app is also available as Streamlit_app.py

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests with your improvements.

