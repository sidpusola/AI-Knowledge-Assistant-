from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():            
            embeddings=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={
                        "device":"cpu"
                },
                encode_kwargs={
                        "normalize_embeddings":True
                }
            )
            return embeddings

if __name__=="__main__":
        embeddings=get_embeddings()
        vector=embeddings.embed_query("What is HTML?")
        print(type(vector))
        print(len(vector))
        print(vector[:10])