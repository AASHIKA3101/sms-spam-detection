import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import pyttsx3  # Adding pyttsx3 for cross-platform text-to-speech

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def speak(text, rate=150):
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)  # Setting the speech rate
    engine.say(text)
    engine.runAndWait()

def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    # Stem the words
    text = [ps.stem(word) for word in text]

    return " ".join(text)

# Load the vectorizer and model from disk
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Title of the web app
st.title("SMS Spam Classifier")

# Text area for user input
input_sms = st.text_area("Enter the message")

# Slider for speech rate
speech_rate = st.slider('Select Speech Rate', 50, 300, 150)

# Prediction button
if st.button('Predict'):
    # 1. Preprocess the input
    transformed_sms = transform_text(input_sms)
    
    # 2. Vectorize the input text
    vector_input = tfidf.transform([transformed_sms])
    
    # 3. Predict the result
    result = model.predict(vector_input)[0]
    
    # 4. Display the result
    if result == 1:
        st.header("Spam")
        speak("This is Spam", rate=speech_rate)
    else:
        st.header("Not Spam")
        speak("This is Not Spam", rate=speech_rate)
