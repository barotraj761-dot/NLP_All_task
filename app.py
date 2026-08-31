import streamlit as st
import nltk
import re
import string
from collections import Counter

# =========================
# NLTK Downloads
# =========================

@st.cache_resource
def download_nltk_data():
    packages = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "maxent_ne_chunker",
        "maxent_ne_chunker_tab",
        "words",
        "vader_lexicon"
    ]

    for package in packages:
        try:
            nltk.download(package, quiet=True)
        except Exception:
            pass


download_nltk_data()


# =========================
# Imports
# =========================

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.util import ngrams

# Optional WordCloud
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="NLP All-in-One Toolkit",
    page_icon="🧠",
    layout="wide"
)


# =========================
# Custom Functions
# =========================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_word_tokens(text):
    return word_tokenize(text)


def remove_stopwords(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))

    return [
        word for word in tokens
        if word.lower() not in stop_words
        and word not in string.punctuation
    ]


def stemming(text):
    stemmer = PorterStemmer()

    tokens = word_tokenize(text)

    result = []

    for word in tokens:
        if word.isalpha():
            result.append(
                (word, stemmer.stem(word))
            )

    return result


def lemmatization(text):
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(text)

    result = []

    for word in tokens:
        if word.isalpha():
            result.append(
                (word, lemmatizer.lemmatize(word.lower()))
            )

    return result


def get_ngrams(text, n):
    tokens = word_tokenize(clean_text(text))

    tokens = [
        word for word in tokens
        if word.isalpha()
    ]

    return list(ngrams(tokens, n))


def get_frequency(text):
    tokens = word_tokenize(clean_text(text))

    tokens = [
        word for word in tokens
        if word.isalpha()
    ]

    return Counter(tokens)


def get_pos_tags(text):
    tokens = word_tokenize(text)

    return pos_tag(tokens)


def get_ner(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    tree = ne_chunk(tagged)

    entities = []

    for subtree in tree:
        if hasattr(subtree, "label"):
            entity = " ".join(
                word for word, tag in subtree.leaves()
            )

            entities.append(
                (entity, subtree.label())
            )

    return entities


def get_sentiment(text):
    sia = SentimentIntensityAnalyzer()

    score = sia.polarity_scores(text)

    compound = score["compound"]

    if compound >= 0.05:
        sentiment = "Positive 😊"
    elif compound <= -0.05:
        sentiment = "Negative 😞"
    else:
        sentiment = "Neutral 😐"

    return sentiment, score


# =========================
# Header
# =========================

st.title("🧠 NLP All-in-One Toolkit")

st.write(
    "Perform multiple Natural Language Processing tasks "
    "using NLTK and Streamlit."
)

st.divider()


# =========================
# Sidebar
# =========================

st.sidebar.title("📌 NLP Tasks")

task = st.sidebar.radio(
    "Select NLP Task",
    [
        "Text Cleaning",
        "Tokenization",
        "Stopword Removal",
        "Stemming",
        "Lemmatization",
        "POS Tagging",
        "Named Entity Recognition",
        "Sentiment Analysis",
        "N-Grams",
        "Word Frequency",
        "Word Cloud",
        "Text Statistics"
    ]
)


# =========================
# Input Text
# =========================

default_text = """
Natural Language Processing is an exciting field of Artificial Intelligence.
Rahul is studying NLP at Jaipur. He loves Python and machine learning.
The weather is beautiful and the project is very interesting.
"""

text = st.text_area(
    "✍️ Enter your text",
    value=default_text,
    height=200
)


if not text.strip():
    st.warning("Please enter some text.")
    st.stop()


# =========================
# 1. Text Cleaning
# =========================

if task == "Text Cleaning":

    st.header("🧹 Text Cleaning")

    cleaned = clean_text(text)

    st.subheader("Original Text")
    st.write(text)

    st.subheader("Cleaned Text")
    st.success(cleaned)


# =========================
# 2. Tokenization
# =========================

elif task == "Tokenization":

    st.header("🔤 Tokenization")

    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentence Tokens")
        st.write(sentences)

    with col2:
        st.subheader("Word Tokens")
        st.write(words)

    st.info(f"Total Words: {len(words)}")


# =========================
# 3. Stopword Removal
# =========================

elif task == "Stopword Removal":

    st.header("🚫 Stopword Removal")

    original_tokens = word_tokenize(text)
    filtered_tokens = remove_stopwords(text)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Tokens")
        st.write(original_tokens)

    with col2:
        st.subheader("After Stopword Removal")
        st.write(filtered_tokens)


# =========================
# 4. Stemming
# =========================

elif task == "Stemming":

    st.header("🌱 Stemming")

    results = stemming(text)

    st.table(
        {
            "Original Word": [x[0] for x in results],
            "Stemmed Word": [x[1] for x in results]
        }
    )


# =========================
# 5. Lemmatization
# =========================

elif task == "Lemmatization":

    st.header("📚 Lemmatization")

    results = lemmatization(text)

    st.table(
        {
            "Original Word": [x[0] for x in results],
            "Lemma": [x[1] for x in results]
        }
    )


# =========================
# 6. POS Tagging
# =========================

elif task == "POS Tagging":

    st.header("🏷️ Part of Speech Tagging")

    results = get_pos_tags(text)

    st.table(
        {
            "Word": [x[0] for x in results],
            "POS Tag": [x[1] for x in results]
        }
    )

    st.info(
        "NN = Noun | VB = Verb | JJ = Adjective | "
        "RB = Adverb | PRP = Pronoun"
    )


# =========================
# 7. NER
# =========================

elif task == "Named Entity Recognition":

    st.header("🏢 Named Entity Recognition")

    entities = get_ner(text)

    if entities:

        st.table(
            {
                "Entity": [x[0] for x in entities],
                "Type": [x[1] for x in entities]
            }
        )

    else:
        st.warning("No named entities found.")


# =========================
# 8. Sentiment Analysis
# =========================

elif task == "Sentiment Analysis":

    st.header("😊 Sentiment Analysis")

    sentiment, scores = get_sentiment(text)

    st.subheader("Overall Sentiment")

    if "Positive" in sentiment:
        st.success(sentiment)

    elif "Negative" in sentiment:
        st.error(sentiment)

    else:
        st.info(sentiment)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Positive",
        round(scores["pos"], 3)
    )

    col2.metric(
        "Negative",
        round(scores["neg"], 3)
    )

    col3.metric(
        "Neutral",
        round(scores["neu"], 3)
    )

    col4.metric(
        "Compound",
        round(scores["compound"], 3)
    )


# =========================
# 9. N-Grams
# =========================

elif task == "N-Grams":

    st.header("🔢 N-Gram Generator")

    n = st.selectbox(
        "Select N",
        [1, 2, 3],
        format_func=lambda x: {
            1: "Unigram",
            2: "Bigram",
            3: "Trigram"
        }[x]
    )

    result = get_ngrams(text, n)

    if result:

        st.subheader(f"{n}-Gram Results")

        for item in result:
            st.write(" → ".join(item))

    else:
        st.warning("Not enough words.")


# =========================
# 10. Word Frequency
# =========================

elif task == "Word Frequency":

    st.header("📊 Word Frequency")

    frequency = get_frequency(text)

    most_common = frequency.most_common(15)

    words = [x[0] for x in most_common]
    counts = [x[1] for x in most_common]

    st.bar_chart(
        dict(zip(words, counts))
    )

    st.subheader("Frequency Table")

    st.table(
        {
            "Word": words,
            "Frequency": counts
        }
    )


# =========================
# 11. Word Cloud
# =========================

elif task == "Word Cloud":

    st.header("☁️ Word Cloud")

    if not WORDCLOUD_AVAILABLE:

        st.error(
            "WordCloud package installed nahi hai. "
            "Run: pip install wordcloud matplotlib"
        )

    else:

        cleaned = clean_text(text)

        if cleaned:

            wordcloud = WordCloud(
                width=1000,
                height=500,
                background_color="white",
                colormap="viridis"
            ).generate(cleaned)

            fig, ax = plt.subplots(
                figsize=(12, 6)
            )

            ax.imshow(
                wordcloud,
                interpolation="bilinear"
            )

            ax.axis("off")

            st.pyplot(fig)


# =========================
# 12. Text Statistics
# =========================

elif task == "Text Statistics":

    st.header("📈 Text Statistics")

    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    alphabetic_words = [
        word for word in words
        if word.isalpha()
    ]

    unique_words = set(
        word.lower()
        for word in alphabetic_words
    )

    characters = len(text)
    characters_no_spaces = len(
        text.replace(" ", "")
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Sentences",
        len(sentences)
    )

    col2.metric(
        "Words",
        len(alphabetic_words)
    )

    col3.metric(
        "Unique Words",
        len(unique_words)
    )

    col4, col5 = st.columns(2)

    col4.metric(
        "Characters",
        characters
    )

    col5.metric(
        "Characters (No Spaces)",
        characters_no_spaces
    )

    st.subheader("Detailed Information")

    st.write(
        f"**Total Sentences:** {len(sentences)}"
    )

    st.write(
        f"**Total Words:** {len(alphabetic_words)}"
    )

    st.write(
        f"**Unique Words:** {len(unique_words)}"
    )

    st.write(
        f"**Characters:** {characters}"
    )

    st.write(
        f"**Characters without spaces:** "
        f"{characters_no_spaces}"
    )


# =========================
# Footer
# =========================

st.divider()

st.caption(
    "NLP All-in-One Toolkit | Built with Python, NLTK & Streamlit"
)
