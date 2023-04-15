import streamlit as st
from langchain import PromptTemplate
from langchain.llms import OpenAI

template = """
    Below is an email that may be poorly worded.
    Your goal is to:
    - Properly format the email
    - Convert the input text to a specified tone
    - Convert the input text to a specified dialect
    Here are some examples different Tones:
    - Formal: We went to Barcelona for the weekend. We have a lot of things to tell you.
    - Informal: Went to Barcelona for the weekend. Lots to tell you.  
    Here are some examples of words in different dialects:
    - American: French Fries, cotton candy, apartment, garbage, cookie, green thumb, parking lot, pants, windshield
    - British: chips, candyfloss, flag, rubbish, biscuit, green fingers, car park, trousers, windscreen
    Example Sentences from each dialect:
    - American: I headed straight for the produce section to grab some fresh vegetables, like bell peppers and zucchini. After that, I made my way to the meat department to pick up some chicken breasts.
    - British: Well, I popped down to the local shop just the other day to pick up a few bits and bobs. As I was perusing the aisles, I noticed that they were fresh out of biscuits, which was a bit of a disappointment, as I do love a good cuppa with a biscuit or two.
    Please start the email with a warm introduction. Add the introduction if you need to.

    Below is the email, tone, and dialect:
    TONE: {tone}
    DIALECT: {dialect}
    EMAIL: {email}

    YOUR {dialect} RESPONSE:
"""

prompt = PromptTemplate(
    input_variables=["tone", "dialect", "email"],
    template=template,
)

import os
os.environ["OPENAI_API_KEY"] = "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"

def load_LLM():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=.9)
    return llm

llm = load_LLM()

st.set_page_config(page_title="Globalize Email", page_icon=":robot:")
st.header("Globalize Text")

#col1, col2 = st.columns(2)

# with col1:
#     st.markdown("Often professionals would like to improve their emails, but don't have the skills to do so. \n\n This tool \
#                 will help you improve your email skills by converting your emails into a more professional format. This tool \
#                 is powered by [LangChain](https://langchain.com/) and [OpenAI](https://openai.com) and made by \
#                 [@GregKamradt](https://twitter.com/GregKamradt). \n\n View Source Code on [Github](https://github.com/gkamradt/globalize-text-streamlit/blob/main/main.py)")

# with col2:
#     st.image(image='TweetScreenshot.png', width=500, caption='https://twitter.com/DannyRichman/status/1598254671591723008')

st.write("Often professionals would like to improve their emails, but don't have the skills to do so. \n\n This tool \
                 will help you improve your email skills by converting your emails into a more professional format. This tool \
                 is powered by [LangChain](https://langchain.com/) and [OpenAI](https://openai.com) and made by \
                 [@GregKamradt](https://twitter.com/GregKamradt). \n\n View Source Code on [Github](https://github.com/gkamradt/globalize-text-streamlit/blob/main/main.py)")

st.markdown("## Enter Your Email To Convert")

col1, col2 = st.columns(2)
with col1:
    option_tone = st.selectbox(
        'Which tone would you like your email to have?',
        ('Formal', 'Informal'))

with col2:
    option_dialect = st.selectbox(
        'Which English Dialect would you like?',
        ('American English', 'British English'))

def get_text():
    input_text = st.text_area(label="", placeholder="Your Email...", key="email_input")
    return input_text

email_input = get_text()

st.write(email_input)

st.markdown("### Your Converted Email:")

if email_input:
    prompt_with_email = prompt.format(tone=option_tone, dialect=option_dialect, email=email_input)
    formatted_email = llm(prompt_with_email)
    st.write(formatted_email)



