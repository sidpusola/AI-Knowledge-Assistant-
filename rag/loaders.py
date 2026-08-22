from pathlib import Path
from langchain_community.document_loaders import TextLoader,PyPDFLoader,Docx2txtLoader,CSVLoader

def get_loader(path:str):
    extension=Path(path).suffix.lower()

    loaders={
        ".txt":lambda:TextLoader(path,encoding="utf-8"),
        ".md":lambda:TextLoader(path,encoding="utf-8"),
        ".pdf":lambda:PyPDFLoader(path),
        ".docx":lambda:Docx2txtLoader(path),
        ".csv":lambda:CSVLoader(path),
        }
    if extension not in loaders:
        raise ValueError(f"unsupported file type:{extension}")
        
    return loaders[extension]()

def load_document(path:str):
    loader=get_loader(path)
    return loader.load()


if __name__=="__main__":
    file_path=input("enter the path:").strip()

    if not Path(file_path).exists():
        print(f"Error: file '{file_path}' does not exists")
        exit()

    try:
        docs=load_document(file_path)

        print("\n"+"="*60)
        print(f"Loaded {len(docs)} documents")   
        print("="*60)


        for i, doc in enumerate(docs, start=1):
            print(f"\nDocument {i}")
            print("-" * 60)

            print("Metadata:")
            print(doc.metadata)

            print("\nContent Preview:\n")

            preview = doc.page_content.strip()

            if len(preview) > 1000:
                preview = preview[:1000] + "\n...(truncated)"

            print(preview)

            print("-" * 60)

    except Exception as e:
        print(f"Error: {e}")