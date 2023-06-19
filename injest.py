from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, NLTKTextSplitter
import glob
import os
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()
llm_model_name_or_path = os.environ.get("LLM_MODEL_NAME_OR_PATH")
embedding_model_name_or_path = os.environ.get("EMBEDDING_MODEL_NAME_OR_PATH")
vectorstore_persist_directory = os.environ.get("VECTORSTORE_PERSIST_DIRECTORY")

# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name_or_path)
text_splitter = CharacterTextSplitter()

file_paths = glob.glob("./source_documents/**/*.txt", recursive=True)
documents = []

for file_path in file_paths:
    print(f"{file_path}: Loading")
    loader = TextLoader(file_path, autodetect_encoding=True)
    docs = loader.load()

    print(f"{file_path}: Splitting")
    # text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer=tokenizer)
    # text_splitter = NLTKTextSplitter()
    docs = text_splitter.split_documents(docs)
    documents.extend(docs)

    # page_contents = []
    # page_metadatas = []
    # for document in texts:
    #     page_contents.append(document.page_content)
    #     page_metadatas.append(document.metadata)
    # vectors = embedding.embed_documents(texts=page_contents)

print(f"(ALL): Embedding and saving")
db = Chroma(persist_directory=vectorstore_persist_directory, embedding_function=embedding)
db.add_documents(documents=documents)
db.persist()

print(f"(ALL): Done")