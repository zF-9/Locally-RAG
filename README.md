# Retrieval-Augmented Generation (RAG) on local
sandbox for testing out various LLM model locally.

 ## Project Requirements ##
  ```
    - Python 3.X.X 
    - Ollama
    - Langchain-Ollama
    - ChromaDB
  ```

## Concept ##
### Vectors
  ```
  numerical representation of data points that capture their semantic meaning.
  ```
### Ollama
  ```
  flexible environment for running, modifying and managing various LLM.
  ```
[Ollama Official Website](https://ollama.com/)
### ChromaDB
  ```
  open-source AI-native vector database designed for storing and searching large sets of vector embeddings.
  ```
[ChromaDB Official Website](https://www.trychroma.com/)

## Workflow 
  ```
    1. Indexing : converting data into embeddings using a model and storing them in the ChromaDB vector database.
    2. Retrieval : similarity search within the vector database to retrieve the most relevant document or contexual information. 
    3. Augmentation : enriching context provided by the LLM to produce more precise and contexually relevant response. (combined query and retrieved vectors)
    4. Generation : using augmented input, LLM generates a response using pre-trained and additional context
  ```

  ### Install Dependencies (Linux/Windows)
  ```
    Ollama : "pip install ollama"
    ChromaDB : "pip install chromadb"
    Langchain-Ollama : "pip install langchain-ollama" 
  ```


  ### Prospect
  ```
    - offloading any worload on existing subscriptions.
    - entry point for locally deployment option.
    - prototyping customized AI assistant.
    - Avoiding censorship & moderated layers which are built into commercially available AI services.
  ```

### Reading material 
[Running LLM Locally: A Beginner’s Guide to Using Ollama](https://medium.com/@arunpatidar26/run-llm-locally-ollama-8ea296747505);
[ChromaDB Documentations](https://docs.trychroma.com/docs/overview/getting-started);
[How to Use Ollama Effectively with LangChain](https://medium.com/towards-agi/how-to-use-ollama-effectively-with-langchain-tutorial-546f5dbffb70).

### Changelog 
added opentelemetry : pip install opentelemetry-instrumentation-fastapi>=0.41b0
added chromadb-telemetry : pip install opentelemetry-instrumentation-chromadb
