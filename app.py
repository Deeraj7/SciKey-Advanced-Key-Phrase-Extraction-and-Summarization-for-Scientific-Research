from flask import Flask, render_template, request
import pandas as pd
import tensorflow_hub as hub
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import heapq
import time

app = Flask(__name__)

# Load the dataset
file_path = 'random_data.csv'
data = pd.read_csv(file_path)

# Clean the data by removing rows where the abstract is missing
data_cleaned = data.dropna(subset=['abstract'])

# Load the Universal Sentence Encoder model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def find_relevant_abstracts_semantic(question, data, top_n=5):
    abstracts = data['abstract'].tolist()
    embeddings = embed(abstracts + [question])
    cosine_similarities = linear_kernel(embeddings[-1:], embeddings[:-1]).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    top_scores = cosine_similarities[top_indices]
    return [(data.iloc[index]['title'], data.iloc[index]['abstract'], data.iloc[index]['url'], top_scores[i])
            for i, index in enumerate(top_indices)]

def find_relevant_abstracts_with_scores(question, data, top_n=5):
    combined_data = data['abstract'].tolist() + [question]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(combined_data)
    cosine_similarities = linear_kernel(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    top_scores = cosine_similarities[top_indices]
    return [(data.iloc[index]['title'], data.iloc[index]['abstract'], data.iloc[index]['url'], top_scores[i])
            for i, index in enumerate(top_indices)]

def extract_sentences(text, num_sentences=2):
    formatted_text = re.sub('[^a-zA-Z]', ' ', text)
    formatted_text = re.sub(r'\s+', ' ', formatted_text)
    word_list = formatted_text.lower().split()
    word_frequencies = {}
    for word in word_list:
        if word not in word_frequencies:
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1
    maximum_frequncy = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)
    sentence_list = re.split('[.!?]', text)
    sentence_scores = {}
    for sentence in sentence_list:
        for word in re.findall(r'\w+', sentence.lower()):
            if word in word_frequencies and len(sentence.split(' ')) < 30:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_frequencies[word]
                else:
                    sentence_scores[sentence] += word_frequencies[word]
    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    return ' '.join(summary_sentences)

def generate_concise_summary_semantic(question, data, top_n_abstracts=5, num_sentences=1):
    relevant_abstracts = find_relevant_abstracts_semantic(question, data, top_n=top_n_abstracts)
    summaries = [extract_sentences(abstract, num_sentences) for _, abstract, _, _ in relevant_abstracts]
    combined_summary = ' '.join(summaries)
    top_articles_info = [{'title': title, 'url': url, 'score': score} for title, _, url, score in relevant_abstracts]
    return combined_summary, top_articles_info

def generate_concise_summary_tfidf(question, data, top_n_abstracts=5, num_sentences=1):
    relevant_abstracts_with_scores = find_relevant_abstracts_with_scores(question, data, top_n=top_n_abstracts)
    summaries = [extract_sentences(abstract, num_sentences) for _, abstract, _, _ in relevant_abstracts_with_scores]
    combined_summary = ' '.join(summaries)
    top_articles_info = [{'title': title, 'url': url, 'score': score} for title, _, url, score in relevant_abstracts_with_scores]
    return combined_summary, top_articles_info

def calculate_relevance(query, summary):
    vectorizer = TfidfVectorizer()
    combined_texts = [query, summary]
    tfidf_vectors = vectorizer.fit_transform(combined_texts)
    similarity = cosine_similarity(tfidf_vectors[0:1], tfidf_vectors[1:2])
    return similarity[0][0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    question = request.form['question']
    selected_method = request.form.get('method', 'tfidf')

    start_time = time.time()

    if selected_method == 'semantic':
        summary, top_articles_info = generate_concise_summary_semantic(question, data_cleaned)
    else:
        summary, top_articles_info = generate_concise_summary_tfidf(question, data_cleaned)

    end_time = time.time()
    runtime = round(end_time - start_time, 2)

    # Calculate the relevance score
    relevance_score = calculate_relevance(question, summary)

    return render_template('result.html', summary=summary, top_articles_info=top_articles_info, runtime=runtime, relevance_score=relevance_score)

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request
import pandas as pd
import tensorflow_hub as hub
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
import re
import heapq
import time

app = Flask(__name__)

# Load the dataset
file_path = 'random_data.csv'
data = pd.read_csv(file_path)

# Clean the data by removing rows where the abstract is missing
data_cleaned = data.dropna(subset=['abstract'])

# Load the Universal Sentence Encoder model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Your existing methods: find_relevant_abstracts_semantic, find_relevant_abstracts_with_scores, extract_sentences, etc.
# ...

def calculate_relevance(query, summary):
    vectorizer = TfidfVectorizer()
    combined_texts = [query, summary]
    tfidf_vectors = vectorizer.fit_transform(combined_texts)
    similarity = cosine_similarity(tfidf_vectors[0:1], tfidf_vectors[1:2])
    return similarity[0][0]

def calculate_rouge_score(summary, references):
    rouge = Rouge()
    scores = rouge.get_scores(summary, references, avg=True)
    return scores

@app.route('/process', methods=['POST'])
def process():
    question = request.form['question']
    selected_method = request.form.get('method', 'tfidf')

    start_time = time.time()

    if selected_method == 'semantic':
        summary, top_articles_info = generate_concise_summary_semantic(question, data_cleaned)
    else:
        summary, top_articles_info = generate_concise_summary_tfidf(question, data_cleaned)

    end_time = time.time()
    runtime = round(end_time - start_time, 2)

    # Calculate the relevance score
    relevance_score = calculate_relevance(question, summary)

    # Combine reference abstracts into a single string for ROUGE calculation
    reference_abstracts = ' '.join([info[1] for info in top_articles_info])

    # Calculate ROUGE score
    rouge = Rouge()
    try:
        rouge_score = rouge.get_scores(summary, reference_abstracts, avg=True)
    except Exception as e:
        rouge_score = {'error': str(e)}

    return render_template('result.html', summary=summary, top_articles_info=top_articles_info, runtime=runtime, relevance_score=relevance_score, rouge_score=rouge_score)

if __name__ == '__main__':
    app.run(debug=True)
