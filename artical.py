import streamlit as st
from newspaper import Article
from gtts import gTTS
import os

st.set_page_config(page_title="Article to Speech", layout="wide")

# Title
st.title("📰 Article Reader with Audio")

# URL input
url = st.text_input("Enter the article URL:")

if url:
    try:
        # Scrape article
        article = Article(url)
        article.download()
        article.parse()
        text = article.text

        # Display article content
        st.subheader(article.title)
        st.write(text)

        # Button to convert to audio
        if st.button("🔊 Convert to Audio"):
            tts = gTTS(text=text, lang='en')
            audio_file = "article_audio.mp3"
            tts.save(audio_file)
            st.audio(audio_file, format='audio/mp3')
            st.success("Audio ready! Listen above.")

    except Exception as e:
        st.error("Could not retrieve article. Error: " + str(e))
else:
    st.info("Please enter a valid URL to read the article.")
