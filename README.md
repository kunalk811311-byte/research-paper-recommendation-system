# Research Paper Recommendation System

## About the Project

This project is an NLP-based **Research Paper Recommendation and Analysis System**. I built this project to understand how semantic search and text embeddings can be used to find relevant research papers.

Normally, keyword-based search depends on exact words. In this project, the system tries to understand the **meaning of the user's query** and recommends research papers with similar content.

For example, if a user searches for:

`Deep learning in medical imaging`

the system converts the query into a numerical vector and compares it with research paper embeddings to find the most relevant papers.

Along with paper recommendation, I also added features such as summarization, keyword extraction, topic classification, Named Entity Recognition, and question answering.

---

## Dataset and Data Preparation

I used the **ML-ArXiv-Papers dataset from Hugging Face**, which contains machine learning research papers.

After loading the dataset, I converted it into a Pandas DataFrame and selected the `title` and `abstract` columns.

I combined the title and abstract into a new column called `paper_text`.

```python
df["paper_text"] = df["title"] + " " + df["abstract"]
```

Combining both fields gives the model more information about the topic and content of each research paper.

I also removed newline characters and extra spaces from the text.

For this project, I used **15,000 research papers** to make the development and testing process manageable in Google Colab.

---

## Generating Text Embeddings

To understand the semantic meaning of research papers, I used the Sentence Transformer model:

`all-MiniLM-L6-v2`

The model converts text into **384-dimensional numerical vectors called embeddings**.

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
```

Research papers with similar meanings should have embeddings that are close to each other.

Before generating embeddings for the complete dataset, I tested the model on a few research papers and used **cosine similarity** to understand how similarity between two embeddings is calculated.

After testing, I generated embeddings for all 15,000 research papers.

Since generating thousands of embeddings takes time, I saved them in a NumPy file called:

`paper_embeddings.npy`

If the file already exists, the notebook loads the saved embeddings instead of generating them again. This saves a lot of processing time.

---

## Semantic Search using FAISS

After generating the embeddings, I used **FAISS** to perform fast vector similarity search.

First, I normalized the research paper embeddings and created a FAISS Inner Product index.

```python
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(384)

index.add(embeddings)
```

I also saved the FAISS index as:

`paper_faiss.index`

This allows the project to reuse the existing index instead of creating it every time.

When a user enters a search query, the following steps are performed:

1. The query is converted into an embedding.
2. The query embedding is normalized.
3. FAISS compares it with the research paper embeddings.
4. The most similar papers are retrieved.
5. The paper title, abstract, and similarity score are displayed.

I created a `search_paper()` function to perform this complete process.

---

## Research Paper Summarization

Research paper abstracts can be long, so I added an automatic summarization feature.

For this, I used the Hugging Face model:

`facebook/bart-large-cnn`

The model generates a shorter version of the research paper abstract while keeping the main information.

I combined semantic search and summarization in the `search_and_summarize()` function.

```python
search_and_summarize(
    "Deep learning in medical imaging",
    k=5
)
```

The function searches for the five most relevant research papers and generates a short summary for every recommended paper.

---

## Keyword and Keyphrase Extraction

I used **KeyBERT** to extract important keywords and keyphrases from research paper abstracts.

The system can extract one-word, two-word, and three-word keyphrases.

This helps in quickly understanding the important topics discussed in a research paper.

For example, a paper related to medical image analysis may contain keywords such as:

* Deep Learning
* Medical Image Analysis
* Neural Networks
* Image Classification

---

## Additional NLP Features

I also experimented with some additional NLP tasks to understand how different pre-trained models can be used.

### Named Entity Recognition

I used **spaCy** to identify important entities from research text.

The system can detect entities such as organizations, locations, dates, and people.

### Research Topic Classification

I used the `facebook/bart-large-mnli` model for zero-shot classification.

The system classifies research text into topics such as Artificial Intelligence, Machine Learning, Natural Language Processing, Computer Vision, Medical Research, Robotics, Cyber Security, and Data Science.

### Question Answering

I used the `distilbert-base-cased-distilled-squad` model to build a simple question-answering feature.

The user provides research text and asks a question. The model finds the answer from the given text.

### Text Summarization

I also created a separate text summarization function using BART.

The user can enter research text and the system generates a shorter summary of the content.

---

## Project Workflow

The complete workflow of the project is:

`Research Paper Dataset`

↓

`Select and Clean Title + Abstract`

↓

`Create Paper Text`

↓

`Generate Sentence Embeddings`

↓

`Save Embeddings`

↓

`Create FAISS Vector Index`

↓

`Convert User Query into Embedding`

↓

`Search Similar Research Papers`

↓

`Display Relevant Papers`

↓

`Generate Paper Summaries and Keywords`

---

## Technologies Used

* Python
* Google Colab
* Pandas
* NumPy
* Hugging Face Datasets
* Sentence Transformers
* FAISS
* Scikit-learn
* Hugging Face Transformers
* BART
* KeyBERT
* spaCy
* DistilBERT

---

## What I Learned

Through this project, I learned how **text embeddings and semantic search work in a practical NLP application**.

I understood how Sentence Transformers convert text into numerical vectors and how cosine similarity can be used to compare those vectors.

I also learned how FAISS performs fast similarity searches on thousands of embeddings.

One important challenge was that generating embeddings for thousands of papers takes time. I solved this by saving the generated embeddings and FAISS index so they can be loaded again when the notebook is restarted.

I also got practical experience using different pre-trained NLP models for summarization, keyword extraction, Named Entity Recognition, topic classification, and question answering.

---

## Future Improvements

In the future, I plan to:

* Build a Streamlit web interface
* Add PDF research paper upload
* Add original arXiv paper links
* Improve question answering for long research papers
* Add better research topic filters
* Deploy the project as a web application

---

## Author

**Kishore Kunal**
