from glados.tts import TextToSpeech

if __name__ == "__main__":
    glados = TextToSpeech(log=True)
    glados.speak("Look at this test subject, soaring magnificently through the sky. "
                 "Like an eagle, piloting a blimp.", filename="output")
