# ── 1. IMPORTS ──────────────────────────────────────────
import os

from ddgs import DDGS
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from groq import Groq
# ── 2. LOAD API KEY ─────────────────────────────────────
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Please add it in your .env file.")

# ── 3. THE TOOL — folder creator ────────────────────────
@tool
def create_folder(folder_name: str, location: str = "desktop") -> str:
    """
    Creates a folder at a specified location.
    Location can be: desktop, documents, downloads, or a full path like D:/MyFolder.
    """
    location_clean = location.strip()
    location_key = location_clean.lower()

    home = os.path.expanduser("~")

    shortcuts = {
        "desktop": os.path.join(home, "Desktop"),
        "documents": os.path.join(home, "Documents"),
        "downloads": os.path.join(home, "Downloads"),
    }

    if location_key in shortcuts:
        base_path = shortcuts[location_key]
    else:
        base_path = os.path.expandvars(os.path.expanduser(location_clean))

    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    return f"✅ Folder '{folder_name}' created at: {folder_path}"




#_____________tool for  searching in web _____________________

@tool
def search_web(query: str) -> str:
    """
    Searches the web and uses AI to summarize the results clearly.
    Input should be the search query as a string.
    """
    try:
        # Step 1 — Get raw results from DuckDuckGo
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "No results found for your query."

        # Step 2 — Format raw results into text
        raw_text = ""
        for i, result in enumerate(results, 1):
            raw_text += f"{i}. {result['title']}\n{result['body']}\n\n"

        # Step 3 — Send to Groq AI to summarize
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
           model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Summarize the following search results clearly and concisely for the user. Include the key points only."
                },
                {
                    "role": "user",
                    "content": f"Search query: {query}\n\nSearch results:\n{raw_text}"
                }
            ]
        )

        summary = response.choices[0].message.content

        return f"🔍 Query: {query}\n\n🤖 AI Summary:\n{summary}"

    except Exception as e:
        return f"Search failed: {str(e)}"
    





# ── TOOL: Voice input ────────────────────────────────
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import whisper
import tempfile

@tool
def listen_voice(query: str) -> str:
    """Listens to microphone for 5 seconds and converts speech to text using Whisper AI. Use when user says listen, speak, or use microphone."""
    try:
        sample_rate = 16000
        duration = 5
        print(f"\n🎙️ Listening for {duration} seconds... Speak now!")
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='int16'
        )
        sd.wait()
        print("✅ Recording done! Processing...")
        temp_file = tempfile.mktemp(suffix=".wav")
        wav_write(temp_file, sample_rate, audio)
        whisper_model = whisper.load_model("base")
        result = whisper_model.transcribe(temp_file)
        os.remove(temp_file)
        transcribed_text = result["text"].strip()
        return f"🎙️ You said: '{transcribed_text}'"
    except Exception as e:
        return f"Voice input failed: {str(e)}"
    
# ── 4. THE AI MODEL ─────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key
)

# ── 5. THE PROMPT ───────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful desktop assistant with three tools:

1. create_folder — ONLY when user says 'create folder', 'make folder', 'new folder'
2. search_web — ONLY for questions about facts, news, current events, or information
3. listen_voice — ONLY when user says 'listen', 'speak', 'use microphone'

STRICT RULES:
- For greetings like 'hello', 'how are you', 'hi', 'what's up' — reply naturally from yourself, NO tools
- For general chat or questions you already know the answer to — reply directly, NO tools
- NEVER use search_web for casual conversation or greetings
- NEVER create a folder for a search query
- Use only ONE tool per message
- Only use a tool when it is absolutely necessary
"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# ── 6. BUILD THE AGENT ──────────────────────────────────
tools = [create_folder, search_web,listen_voice]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ── 7. CHAT LOOP ────────────────────────────────────────
print("🤖 Agentic Chatbot ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    response = agent_executor.invoke({"input": user_input})
    print(f"\nAgent: {response['output']}\n")