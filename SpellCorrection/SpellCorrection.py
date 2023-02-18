from nltk import word_tokenize
from autocorrect import Speller
import streamlit as st

st.title("Spell Correction")
st.write("""
## A tool that helps to fix your error vocabulary?
""")

sentence = st.text_input('Your text is: ')

tokens = word_tokenize(sentence)

spell = Speller(lang='en')

def correct_spelling(tokens):
    sentence_corrected = ' '.join([spell(word) for word in tokens])
    return sentence_corrected

st.write(correct_spelling(tokens))

