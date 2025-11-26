from embeddings.embedder import EmbeddingModel
from ingestion.text_loader import load_text
from ingestion.chunker import chunk_text
from vector_store.faiss_store import FaissStore
from query_pipeline.retrieve import Retriever
import os
from llm.local_llm import LLM

class MetaData:
    def __init__(self,file_path=None,texts=None,chunks=None,vectors=None):
        self.file_path = file_path
        self.texts = texts
        self.chunks = chunks
        self.vectors = vectors
        self.num_chunks = len(chunks) if chunks else 0
        self.total_chars = sum(len(c) for c in chunks if chunks) 

    def to_dict(self):
        return {
            "file_path": self.file_path,
            "texts": self.texts,
            "chunks": self.chunks,
            "vectors": self.vectors.tolist() if hasattr(self.vectors, "tolist") else self.vectors,
            "num_chunks": self.num_chunks,
            "total_chars": self.total_chars
        }

def ingest_directory(folder_path: str):
    """
    Ingest every supported file inside a directory (recursively).
    """
    fs = FaissStore()                 # loads existing index automatically
    embedder = EmbeddingModel()

    supported_extensions = {".txt", ".md", ".pdf", ".docx"}

    for root, _, files in os.walk(folder_path):
        for file in files:
            path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()

            if ext not in supported_extensions:
                print(f"Skipping unsupported file: {path}")
                continue

            try:
                text = load_text(path)
            except Exception as e:
                print(f"Could not read {path}: {e}")
                continue

            chunks = chunk_text(text)
            vectors = embedder.embed_texts(chunks)

            fs.add_vectors(vectors, chunks, file_path=path, embedder_model=embedder.model_name)
            print(f"[Ingested] {path} → {len(chunks)} chunks")

    fs.save_index()
    print("\n>>> Directory ingestion complete.\n")


def ingest_file(path:str):
    texts = load_text(path)
    chunks = chunk_text(texts)
    if not chunks or len(chunks) == 0:
        print(f'[File Skipped] No text found in {path}')
        return
    
    embeds = EmbeddingModel()
    vectors = embeds.embed_texts(chunks)
    data = MetaData(path,texts,chunks,vectors)
    fs = FaissStore()
    # print(len(data.vectors),len(data.chunks))
    # print(data.chunks)
    fs.add_vectors(data.vectors,data.chunks,embedder_model=embeds.model_name)
    fs.save_index()
    print(f"[OK] Ingested {path} — {len(chunks)} chunks")

def main():

    while True:
        
        print("\n==================== OFFLINE AI ENGINE ====================")
        print("1. Ingest a single file")
        print("2. Ingest an entire directory")
        print("3. Reset vector index")
        print("4. Run semantic query")
        print("5. Exit")
        print("============================================================")

        choice = input("Select an operation: ").strip()

        if choice == "1":
            
            file_path = input("Enter file path: ").strip()
            ingest_file(file_path)
    
        elif choice == "2":
            folder = input("Enter directory path: ").strip()
            ingest_directory(folder)

        elif choice == "3":
            FaissStore.reset_index(index_path='faiss/vector_index.faiss')

        elif choice == "4":
            query = input("Enter your query: ").strip()
            
            result = Retriever().search(query)
            


            print("\n----- Query Response -----\n")
            
            if not result:
                print("No relevant results found.\nYour query does not match anything in the indexed data.")
                print("--------------------------\n")
                input('> ')
                continue
            
            for i,r in enumerate(result):
                print(f"[{i}] Score: {r['score']:.4f}")
                print("    Text :",r['text'].replace('\n',' '))
            
            print("\n--------------------------\n")

        elif choice == "5":
            print("Shutting down operational workflow.")
            # sys.exit(0)
            return

        # LLM (RAG)
        elif choice == "6":
            from query_pipeline.rag import run_rag
            question = input("Enter your question: ").strip()
            answer, chunks = run_rag(question)

            print('Answer:\n>',answer)
            input('\nPress Enter to see sources ')
            print("\nSources:\n")
            for i,c in enumerate(chunks,start=1):
                # print(c)
                print(f"[{i}] id={c['id']} score={c['score']:.4f}")
                print(c['text'][:200].replace('\n',' '), "...\n")





        else:
            print("Invalid selection. Please choose a valid menu option.")    
        input('> ')



if __name__ == "__main__":
    main()