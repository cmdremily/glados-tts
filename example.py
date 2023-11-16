from glados.tts import TextToSpeech

if __name__ == "__main__":
    glados = TextToSpeech(log=True)
    glados.speak("Some girls just want to watch the world burn.", filename="output")

