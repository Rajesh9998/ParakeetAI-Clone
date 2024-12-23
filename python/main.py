from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from deepgram import Deepgram
from dotenv import load_dotenv
import os
from aiohttp import web
from typing import Dict, Callable
from duckduckgo_search import DDGS
import test4
import threading

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

dg_client = Deepgram(os.getenv('DEEPGRAM_API_KEY'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_question_answer():
    data = request.get_json()
    dialogues = data.get('dialogues')
    try:
        Question = DDGS().chat(f"Query : form the above transcripts Just Identify the Question that the applicant should be answering note that stick to the last question that the Recruiter asked To the applicant. The previous conversations are: {dialogues} Note: Don't Return anything other than what was Asked no need for any explanations or no need for any additional start words like here is the asked questions like that . just return the question and nothing else")
        print('\n\n\n')
        print(Question)
        result = test4.process_query(Question)
        answer = result["response"]
        return jsonify({'question': Question, 'answer': answer})
    except Exception as e:
        return jsonify({'error': f"Error in generating question or answer: {str(e)}"})

@app.route('/analyze', methods = ['POST'])
def analyze_transcriptions():
      data = request.get_json()
      tabAnalysisTranscript = data.get('tabAnalysisTranscript')
      micAnalysisTranscript = data.get('micAnalysisTranscript')
      print("Received tabAnalysisTranscript:", tabAnalysisTranscript)
      print("Received micAnalysisTranscript:", micAnalysisTranscript)
      try:
        analysis =  DDGS().chat(f"Query: Given the following transcripts convert them into an understandable dialouge format between the recruiter and the candidate . Tab Audio Transcript : {tabAnalysisTranscript} Microphone Audio Transcript : {micAnalysisTranscript}. The output should only be in this Format: Recruiter : ................... Applicant: ................... and nothing else no additional start words, or notes, just the format",model = 'gpt-4o-mini')
        return jsonify({'analysis' : analysis})
      except Exception as e:
        return jsonify({'error' : f"Error in analyzing : {str(e)}"})

async def process_audio(fast_socket: web.WebSocketResponse):
    async def get_transcript(data: Dict) -> None:
        if 'channel' in data:
            transcript = data['channel']['alternatives'][0]['transcript']
            if transcript:
                await fast_socket.send_str(transcript)
    
    deepgram_socket = await connect_to_deepgram(get_transcript)
    return deepgram_socket

async def connect_to_deepgram(transcript_received_handler: Callable[[Dict], None]) -> str:
    try:
        socket = await dg_client.transcription.live({'punctuate': True, 'interim_results': False})
        socket.registerHandler(socket.event.CLOSE, lambda c: print(f'Connection closed with code {c}.'))
        socket.registerHandler(socket.event.TRANSCRIPT_RECEIVED, transcript_received_handler)
        
        return socket
    except Exception as e:
        raise Exception(f'Could not open socket: {e}')

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    deepgram_socket = await process_audio(ws)
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.BINARY:
                deepgram_socket.send(msg.data)
            elif msg.type == web.WSMsgType.ERROR:
                print('ws connection closed with exception %s' % ws.exception())
                break
    except Exception as e:
        print(f"Error in socket loop: {e}")
    
    return ws

def run_aiohttp_server():
    aio_app = web.Application()
    aio_app.router.add_route('GET', '/listen', websocket_handler)
    web.run_app(aio_app, port=5556)

def run_flask_server():
    app.run(port=5555, debug=True, use_reloader=False)

if __name__ == '__main__':
    # Create threads for Flask and aiohttp servers
    flask_thread = threading.Thread(target=run_flask_server)
    aiohttp_thread = threading.Thread(target=run_aiohttp_server)
    
    # Start both threads
    flask_thread.start()
    aiohttp_thread.start()
    
    # Wait for both threads to complete
    flask_thread.join()
    aiohttp_thread.join()