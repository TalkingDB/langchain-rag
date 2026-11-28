<!-- short-readme -->
<h1 align="center">LangChain RAG (TalkingDB fork)</h1>

<p align="center"><b>A research prototype comparing vector retrieval with word-sense retrieval.</b></p>

<p align="center"><a href="#try-it">Try it</a> · <a href="docs/reference.md">Full reference</a></p>

<p align="center"><a href="https://github.com/TalkingDB/langchain-rag/stargazers"><img src="https://img.shields.io/github/stars/TalkingDB/langchain-rag?style=social" alt="GitHub stars"></a></p>

It started as a LangChain [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) tutorial. Status: prototype, not a supported product.

## What it does

- For supported document retrieval, use [module-ttt](https://github.com/TalkingDB/module-ttt).
- For linking words to known entities, use [package-named-entity-linker](https://github.com/TalkingDB/package-named-entity-linker).
- The original setup, API and evaluation steps are in the [full reference](docs/reference.md).

## Try it

```bash
pip install -r requirements.txt
python create_database.py
python query_data.py "How does Alice meet the Mad Hatter?"
```

Needs an `OPENAI_API_KEY`.

## Where it fits

Superseded by [module-ttt](https://github.com/TalkingDB/module-ttt).

## Learn more

| Page | Read it to |
| :--- | :--- |
| [Full reference](docs/reference.md) | read everything that used to be on this page |
