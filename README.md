
# Hi! I am Abdullah!! 👋
I am an Artificial Intelligence & Data Science student. My Learning Attitude is what keeps me going everyday to explore new things as well as I keep myself motivated to achieve my daily goals.

🥂 Lets Connect : [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/habib-abdullah7239/)

# JARVIS AI

A virtual AI Assistant which can perform basic online as well as offline tasks on your local machine powered by GPT4All. 
This was created by Abdullah Habib as a part of his personal projects.





## Features

- Powered by GPT4All
- No API key is required
- Can run in both offline & online mode (online mode provides additonal features which are used in this project like pywhatkit etc.)
- Can be easily personalzied for your local machine
- Quick & Convenient Responses
- Can perform basic web surfing & local tasks



### Note : No API Key Required!!
Please install the packages properly & follow the steps carefully.
Windows & Mac have different ways of installation.  
Python Version Used - "3.10.11".   
Code Editor Used - PyCharm Community Edition
 - I recommend using a virtual environment for projects.
 - PyCharm is very effective in-terms of virtual environment & package management so i recommend using it.
 - For any errors, either DM on LinkedIn (mentioned above) or refer to [StackOverflow](https://stackoverflow.com/) which will get most of your errors resolved like path-related or execution-policy related errors.

To check what version you are using, use the following command :

```bash
  python3 --version
```








### Step 1: Get GPT4All installed on your local machine

Use this [GPT4ALL Download](https://gpt4all.io/index.html) to download GPT4All desktop app according to your OS.

### Step 2: Install the packages
- GPT4All 

```bash
  python3 -m pip install gpt4all
```

- OpenAI Whisper
```bash
  python3 -m pip install openai-whisper 
```

- Speech Recognition
```bash
  python3 -m pip install SpeechRecognition
```

- Playsound
```bash
  python3 -m pip install playsound
```

- PyAudio (Dependency)
```bash
  python3 -m pip install PyAudio
```

- SoundFile (Dependency)
```bash
  python3 -m pip install soundfile 
```

- ffmpeg (Dependency)
#### For MAC users, using [Homebrew](https://brew.sh/) use this command:
```bash
  brew install ffmpeg
```
#### For Windoes
```bash
  python3 -m pip install ffmpeg
```

- Port Audio (ONLY FOR MAC USERS)
```bash
  brew install portaudio
```

### Step 3 : To make it work offline (optional)

When you load the tiny & base models and use the transcribe function they always check for two files which we have to modify in order for it to work. 
Use your Terminal/Powershell to switch to the directory where your whisper models are stored (.cache/whisper/) 

- For MAC Users, run these 2 commands in Terminal:
```bash
  curl -o encoder.json https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe
```
```bash
  curl -o encoder.json https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json
```

- For WINDOWS Users, run these 2 commands in Powershell:
```bash
  Invoke-WebRequest -Uri "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe" -Outfile "encoder.json"
```
```bash
  Invoke-WebRequest -Uri "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json" -Outfile "encoder.json"
```

#### Modify the openai_public.py 
Change the directory to "python[your_version]/site-packages/tiktoken_ext/" 
There will be a file in it called "openai_public.py" edit it using IDE 

in gpt2() function, change the vocab_bpe_file & encoder_json_file

```bash
  vocab_bpe_file = os.path.expanduser('~/.cache/whisper/vocab.bpe')
  encoder_json_file = os.path.expanduser('~/.cache/whisper/encoder.json')
```

## Support

For support, email habib.abdullah7239@gmail.com or refer to [YouTube Video](https://www.youtube.com/watch?v=6zAk0KHmiGw) for step by step explanation if you are facing issues in the code.
You will find certain differences in code given.

