from encoder.embedder import EmbeddingModel
from ingestion.text_loader import load_text
from ingestion.chunker import chunk_text
from storage.faiss_store import FaissStore
from pipeline.retrieve import Retriever
import os
from InquirerPy import inquirer
from rich.console import Console


MODEL_NAME = 'sentence-transformers/all-MiniLM-L12-v2'
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

    fs = FaissStore()
    embedder = EmbeddingModel()

    supported_extensions = {".txt", ".md", ".pdf", ".docx"}

    all_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            all_files.append(os.path.join(root, file))

    print(f"[Scan] Found {len(all_files)} files.")

    for path in all_files:
        ext = os.path.splitext(path)[1].lower()
        if ext not in supported_extensions:
            continue

        try:
            text = load_text(path)
        except:
            continue

        chunks = chunk_text(text)

        vectors = embedder.model.encode(chunks, show_progress_bar=True)

        fs.add_vectors(vectors, chunks, file_path=path, embedder_model=embedder.model_name)

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
    try:
        console = Console()
        
        logo = """
        ██╗     ██╗   ██╗██████╗  █████╗ 
        ██║     ██║   ██║██╔══██╗██╔══██╗
        ██║     ██║   ██║██████╔╝███████║
        ██║     ██║   ██║██╔══██╗██╔══██║
        ███████╗╚██████╔╝██║  ██║██║  ██║
        ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
        """
        console.print(logo)
        while True:

            # console.rule("[bold blue]Lura")

            choice = inquirer.select(
                message="Select an option:",
                choices=[
                    "1. Ingest a single file",
                    "2. Ingest an entire directory",
                    "3. Reset vector index",
                    "4. Run semantic search",
                    "5. Run RAG query",
                    "6. Show index stats",
                    "7. Exit"
                ],
                default=None,
                pointer=">",
                
            ).execute()
            choice = (choice[0])


            if choice == "1":
                
                file_path = input("Enter file path: ").strip()
                
                ingest_file(file_path)
        
            elif choice == "2":
                folder = input("Enter directory path: ").strip()
                ingest_directory(folder)

            elif choice == "3":
                index_path = 'src/faiss/vector_index.faiss'
                if os.path.exists(index_path):
                    FaissStore.reset_index(index_path=index_path,model_name=MODEL_NAME)
                    Retriever._instance = None

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
                    print(f"[{i}] id:{r['id']} Score: {r['score']:.4f}")
                    print("    Text :",(r['text'].replace('\n',' '))[:300])
                
                print("\n--------------------------\n")

            # LLM (RAG)
            elif choice == "5":
                from pipeline.rag import run_rag
                question = input("Enter your question: ").strip()
                console.print("\n[dim]Loading...[/]\n")
                answer, chunks = run_rag(question)

                print('Answer:\n>',answer)
                input('\nPress Enter to see sources ')
                print("\nSources:\n")
                for i,c in enumerate(chunks,start=1):
                    # print(c)
                    print(f"[{i}] id={c['id']} score={c['score']:.4f}")
                    print(c['text'][:300].replace('\n',' '), "...\n")
            
                # 6. Show index stats
            elif choice == "6":
                fs = FaissStore()
                print("\nIndex Stats\n")
                print("  Embedding model:", fs.model_name)
                print("  Vector dimension:", fs.dim)
                print("  Chunks indexed:", len(fs.documents))
                print("  FAISS index path:", fs.index_path)
            
            
            elif choice == "7":
                print("Shutting down operational workflow.")
                # sys.exit(0)
                return

            else:
                print("Invalid selection. Please choose a valid menu option.")    
            input('> ')

    except KeyboardInterrupt as keyboardintp:
        print('\n[Ctrl + C pressed] Exiting...')

if __name__ == "__main__":
    main()