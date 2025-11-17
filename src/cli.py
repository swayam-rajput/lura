from embeddings.embedder import EmbeddingModel
from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.text_loader import load_text
from ingestion.chunker import chunk_text
from vector_store.faiss_store import FaissStore
from query_pipeline.retrieve import Retriever
import os

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

# def ingest_directory(folder_path):
#     embeds = EmbeddingModel()

#     fs = FaissStore()   # uses existing index file or creates new one
#     all_chunks = []
#     all_vectors = []

#     for root, _, files in os.walk(folder_path):
#         for f in files:
#             file_path = os.path.join(root, f)

#             try:
#                 text = load_text(file_path)
#             except:
#                 print(f"Skipping unreadable file: {file_path}")
#                 continue

#             chunks = chunk_text(text)

#             vectors = embeds.embed_texts(chunks)

#             fs.add_vectors(vectors, chunks)
#             print(f"[Ingested] {file_path} → {len(chunks)} chunks")

#     fs.save_index()
#     print("\n>>> Directory ingestion complete.")

def main():

    while True:
        
        print("\n==================== OFFLINE AI ENGINE ====================")
        print("1. Ingest a single file")
        print("2. Ingest an entire directory")
        print("3. Build / Rebuild vector index")
        print("4. Run semantic query")
        print("5. Exit")
        print("============================================================")

        choice = input("Select an operation: ").strip()

        if choice == "1":
            
            file_path = input("Enter file path: ").strip()
            texts = load_text(file_path)
            chunks = chunk_text(texts)
            embeds = EmbeddingModel()
            
            vectors = embeds.embed_texts(chunks)
            data = MetaData(file_path,texts,chunks,vectors)
            fs = FaissStore()
            print(len(data.vectors),len(data.chunks))
            print(data.chunks)
            fs.add_vectors(data.vectors,data.chunks)
            fs.save_index()
            
            


        # elif choice == "2":
        #     folder = input("Enter directory path: ").strip()
        #     ingest_directory(folder)

        # elif choice == "3":
        #     build_vector_index()

        elif choice == "4":
            query = input("Enter your query: ").strip()
            embedded_text = EmbeddingModel().embed_query(query)

            result = Retriever().search(query)
            print("\n----- Query Response -----\n")
            for i,value in enumerate(result):
                print(i,value['text'][:10],value['score'])
            # print(result)
            print("\n--------------------------\n")

        # elif choice == "5":
        #     print("Shutting down operational workflow.")
        #     sys.exit(0)

        else:
            print("Invalid selection. Please choose a valid menu option.")    
        input('> ')

    print("No operation executed. Use --help for available commands.")


if __name__ == "__main__":
    main()