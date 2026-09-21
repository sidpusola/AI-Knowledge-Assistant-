from langchain_huggingface import HuggingFaceEmbeddings
from config import settings


def get_embeddings():
            embeddings=HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={
                        "device":settings.EMBEDDING_DEVICE
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