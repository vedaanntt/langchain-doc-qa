# Document Q&A with LangChain

Ask questions about your own PDF and text files, and get answers with the sources they came from. By default everything runs on your own computer, for free.

## Setup (Windows)

You need [Ollama](https://ollama.com) installed and Python 3.13. The code was tested on 3.13, not 3.14.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull llama3.2
```

The first run also downloads two small search models (about 150 MB) into `.model_cache/`.

## Run it

Web app:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Command line:

```powershell
.venv\Scripts\python.exe qa_system.py your_file.pdf
```

## How it works

The idea is called **RAG** (retrieval-augmented generation). An AI model can't read a whole library for every question, so you find the few relevant passages first and give it only those.

**When you add a document:**

1. **Load:** pull the text out of the file.
2. **Chunk:** cut it into pieces of about 1,000 characters. Each piece shares 200 characters with the next, so a sentence on the boundary isn't lost.
3. **Embed:** turn each chunk into a list of numbers that captures its meaning. Similar meanings get similar numbers.
4. **Store:** save the chunks and numbers in a search index (Chroma) in `chroma_db/`.

**When you ask a question:**

5. **Search two ways:** meaning search finds similar ideas even in different words. Keyword search finds exact names and codes like `E-417`.
6. **Rerank:** a more careful model reads the candidates against your question, and the best 4 are kept.
7. **Answer:** the 4 chunks and your question go into a prompt, and the AI model writes the reply.

### Where LangChain fits

LangChain gives every part a standard shape, so parts can be swapped without rewriting the rest. That's why changing the AI model here is a single setting.

| LangChain piece | What it does here |
| --- | --- |
| `Document` | A chunk of text plus notes about it, like its file and page |
| `RecursiveCharacterTextSplitter` | Cuts text into chunks, preferring paragraph and sentence breaks |
| `Chroma` | Stores chunks and searches them by meaning |
| `ChatPromptTemplate` | The instructions, with blanks for the excerpts and the question |
| `ChatOllama`, `ChatAnthropic` | Interchangeable AI models |
| `prompt \| llm \| parser` | A "chain": each step's output feeds the next |

## Switching to Claude

Local models are free and private. Claude is better at questions that need reasoning across several parts of a document. It costs roughly 1 to 2 cents per question on the default model, and the matching excerpts of your documents are sent to Anthropic.

1. Copy `.env.example` to `.env`.
2. Set `LLM_PROVIDER=anthropic` and paste in your `ANTHROPIC_API_KEY`.

To use a cheaper model, set `ANTHROPIC_MODEL=claude-haiku-5-5`.

The request settings were checked, but this path hasn't been run against the live API.

## What made answers better

Measured with `eval/evaluate.py` on a fictional handbook (62 chunks, 83 questions), using llama3.2 on a 6 GB GPU:

| Change | Result |
| --- | --- |
| Rewrote the prompt, with instructions right next to the question | Correct answers went from 15/38 to 77/77 (the question set also grew between runs) |
| Added keyword search and a reranker | Right text found 77/77 times, up from 75/77 with meaning search alone |
| Switched PDF reader from `pypdf` to `pypdfium2` | Fixed glued-together words like `answeringsystemusingLangChainto` |
| Set `num_ctx=8192` and `temperature=0` | Excerpts aren't silently cut off, and answers are repeatable |

Two ideas were tried and rejected because the numbers didn't support them:

- **A bigger model:** `gemma3:4b` was no better than `llama3.2` on these tests.
- **Declining based on the reranker's score:** it couldn't separate real questions from unanswerable ones.

The handbook is an easy, purpose-built test, so treat these scores as a way to compare settings, not as accuracy you'd get on any document. Re-run the evaluation after any change:

```powershell
.venv\Scripts\python.exe eval\evaluate.py --answers
```

## Known limits

- **Reasoning across sections is weak on the local model.** In testing on a resume, it wrongly linked one project to an API that another project used. Check important answers against the sources shown below each reply, or switch to Claude.
- **It can still get a wrong "not found", or ignore its rules.** It sometimes adds nearby facts after saying something isn't mentioned, and small prompt changes can shift this behaviour.
- **Each question stands alone.** There's no memory of the conversation, so "what about the second one?" won't work.
- **Scanned PDFs aren't supported.** They contain pictures of text, which need OCR first. The app shows an error instead of indexing nothing.
- **Very large collections will be slow.** Keyword search keeps every chunk in memory.

## Files

| File | Purpose |
| --- | --- |
| `qa_system.py` | The whole pipeline, plus the command-line version |
| `app.py` | The web interface |
| `eval/` | Quality test: fictional handbook, questions and scoring script |
| `.env.example` | Settings you can change |
