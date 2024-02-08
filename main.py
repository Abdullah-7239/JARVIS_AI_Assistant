from playsound import playsound
import whisper
import sys
import speech_recognition as sr
import os
import webbrowser
from gpt4all import GPT4All
import datetime
import wikipedia
import pywhatkit as wk
import warnings

model = GPT4All("/Users/abhdelirious/Library/Application Support/nomic.ai/GPT4All/gpt4all-falcon-q4_0.gguf", allow_download=False)

r = sr.Recognizer()
tiny_model_path = os.path.expanduser("~/.cache/whisper/tiny.pt")
base_model_path = os.path.expanduser("~/.cache/whisper/base.pt")
tiny_model = whisper.load_model(tiny_model_path)
base_model = whisper.load_model(base_model_path)
# tiny_model = whisper.load_model("tiny")  #for weight detection
# base_model = whisper.load_model("base")  #for complex voice input


warnings.filterwarnings("ignore", category=UserWarning, module='whisper.transcribe', lineno=114)

# FOR WINDOWS USERS ONLY
# if sys.platform != 'darwin':
#     import pyttsx3
#     engine = pyttsx3.init()
#     rate = engine.getProperty('rate')
#     engine.setProperty('rate',rate-50)  #speed tuning

sites = [
        ["Google", "https://www.google.com"],
        ["YouTube", "https://www.youtube.com"],
        ["Facebook", "https://www.facebook.com"],
        ["Amazon", "https://www.amazon.com"],
        ["Twitter", "https://www.twitter.com"],
        ["Instagram", "https://www.instagram.com"],
        ["LinkedIn", "https://www.linkedin.com"],
        ["Netflix", "https://www.netflix.com"],
        ["Reddit", "https://www.reddit.com"],
        ["Pinterest", "https://www.pinterest.com"],
        ["Wikipedia", "https://www.wikipedia.org"],
        ["GitHub", "https://www.github.com"],
        ["Stack Overflow", "https://stackoverflow.com"],
        ["Medium", "https://www.medium.com"],
        ["Quora", "https://www.quora.com"],
        ["Etsy", "https://www.etsy.com"],
        ["Dropbox", "https://www.dropbox.com"],
        ["Slack", "https://www.slack.com"],
        ["Twitch", "https://www.twitch.tv"],
        ["Airbnb", "https://www.airbnb.com"],
        ["Spotify","https://open.spotify.com"],
        ["Chat GPT","https://chat.openai.com"]

]

search_web = [
        ["Google", "https://www.google.com/search?q="],
        ["YouTube", "https://www.youtube.com/results?search_query="],
        ["Facebook", "https://www.facebook.com/search/top/?q="],
        ["Amazon", "https://www.amazon.com/s?k="],
        ["Twitter", "https://twitter.com/search?q="],
        ["Instagram", "https://www.instagram.com/explore/tags/"],
        ["LinkedIn", "https://www.linkedin.com/search/results/all/?keywords="],
        ["Netflix", "https://www.netflix.com/search?q="],
        ["Reddit", "https://www.reddit.com/search?q="],
        ["Pinterest", "https://www.pinterest.com/search/pins/?q="],
        ["Wikipedia", "https://en.wikipedia.org/wiki/"],
        ["GitHub", "https://github.com/search?q="],
        ["Stack Overflow", "https://stackoverflow.com/search?q="],
        ["Medium", "https://medium.com/search?q="],
        ["Quora", "https://www.quora.com/search?q="],
        ["Etsy", "https://www.etsy.com/search?q="],
        ["Dropbox", "https://www.dropbox.com/search?q="],
        ["Slack", "https://slack.com/search?query="],
        ["Twitch", "https://www.twitch.tv/search?term="],
        ["Airbnb", "https://www.airbnb.com/s/"],
        ["IMDb", "https://www.imdb.com/find?q="]
]
def prompt_gpt(prompt):
    try:
        output = model.generate(prompt, max_tokens=100)
        print('JARVIS AI: ', output)
        speak(output)
        main()
    except Exception as e:
        print("Error Occured ",e)
        exit()

def speak(text):
    if sys.platform == 'darwin': #for MAC Users
        ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!-_$: ")
        clean_text = ''.join(c for c in text if c in ALLOWED_CHARS)
        os.system(f"say '{clean_text}'")
    # FOR WINDOWS USERS ONLY
    # else :
    #     engine.say(text)
    #     engine.runAndWait()

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source :
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 1
        audio = r.listen(source,phrase_time_limit=5)
        try :
            query = r.recognize_google(audio,language='en-in')
            print(f'User said {query}')
            return query
        except Exception as E :
            speak("Sorry i Couldn't understand that")
            speak("Can you repeat, Please!")
            takeCommand()

def takeshortCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.energy_threshold = 300
        r.pause_threshold = 1
        audio = r.listen(source, phrase_time_limit=2)
        try :
            query = r.recognize_google(audio, language='en-in')
            print(f'User said {query}')
            return query
        except Exception as E:
            speak("Sorry I couldn't understand that")
            speak("Can you repeat, Please!")
            takeshortCommand()

def search_url(index,search_web,sk):
    url_search = search_web[index][1] + sk.replace(" ", "+")
    return url_search

def get_index2D(element,list):
    for i in range(len(list)):
        if element == list[i][0]:
            return i
def main() :
    while True :
        speak("I am Listening")

        print("Listening...")
        prompt = takeCommand()
        for site in sites:
            if f"Open {site[0]}".lower() in prompt.lower() and "search" not in prompt.lower():
                webbrowser.open(site[1])
                speak(f"Opening {site[0]} sir")
                main()

        if "open music" in prompt.lower():
            speak("Which song you want to listen to")
            speak("Please mention song name only")
            print("Listening")
            song = takeCommand()
            speak(f"Opening {song} for you")
            wk.playonyt(song)
            continue

        if "the time" in prompt.lower():
            strfTime = datetime.datetime.now().strftime("%H:%M:%S")
            speak(f'The time is {strfTime}')
            continue

        if "who are you" in prompt.lower():
            speak("I am JARVIS AI. Here to assist you with your basic tasks. I have been implemented using Python.")
            speak("I was created by Abdullah Habib")
            speak("I am powered by GPT 4 All")
            speak("How can i help you?")
            continue

        if "search" in prompt.lower() and "google" in prompt.lower():
            index = get_index2D("Google", search_web)
            speak(f"What do you want to search ")
            print("Listening for search...")
            sk = takeCommand()
            speak("Searching your query sir")
            webbrowser.open(search_url(index, search_web, sk))
            result = wikipedia.summary(sk, sentences=2)
            speak(result)
            continue

        if "search" in prompt.lower() and "website" in prompt.lower():
            speak("Sure Sir!, Which website you want to search on?")
            print("Listening for website...")
            website = takeshortCommand()
            index = get_index2D(website, search_web)
            speak(f"What do you want to search on {website}")
            print("Listening for search...")
            sk = takeCommand()
            speak("Searching your query sir")
            try:
                webbrowser.open(search_url(index, search_web, sk))
                speak("Here are the results")
                continue
            except:
                speak("My apologies sir, The website does not exist.")
                speak("Do you want me to tell all the websites in my database that you can use?")
                respond = takeshortCommand()
                if "yes" in respond.lower():
                    speak("The Websites in my Database are ")
                    for i in range(len(search_web)):
                        speak(f"{search_web[i][0]}")
                continue
        if "mood" in prompt.lower() :
            from Bilstm import detection_main, show_emotion
            speak("Hold on let me check your mood")
            detection_main()
            emotion = show_emotion()
            if emotion == "Happy" :
                speak("You look like you are in good mood today!")
                wk.playonyt("Good Vibes Instrumental")
                # response = model.generate('Tell me a good joke',max_tokens=50)
                # print(f"JARVIS AI : {response}")
                # speak(response)
                speak("Here is my effort to make your day better")
                continue
            elif emotion == "Neutral" :
                speak("You look fine! But lets make it better.")
                wk.playonyt("Invincible by mgk")
                speak("Now Cheer Up!!")
                continue


        if "close browser" in prompt.lower():
            speak("Closing browser for you, sir.")
            print("Closing browser...")
            os.system("taskkill /f /im brave.exe")
            continue

        if "thank you" in prompt.lower() and "exit" not in prompt.lower():
            speak("You're Welcome, I am glad I could help. Can I help you with anything else?")
            continue

        if "exit" in prompt.lower():
            speak("Have a good day. See you next time.")
            quit()
        if "write" in prompt.lower() or "summarize" in prompt.lower() or "make" in prompt.lower():
            response = prompt_gpt(prompt)
            speak("Sure! Here are the results")
            print(response)
            speak("Do you want me to read it to you?")
            print("Listening...")
            respond = takeshortCommand()
            if "yes" in respond.lower():
                speak("Certainly!")
                speak(response)
                continue
            continue

        prompt_gpt(prompt)

playsound("wake_detected.mp3")
speak("Hi I am JARVIS AI. How can i help you?")
main()

























































