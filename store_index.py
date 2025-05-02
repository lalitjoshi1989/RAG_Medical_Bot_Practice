from src.helper import load_pdf_file, text_split, download_embed_model
from pinecone.grpc import PineconeGRPC as pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()

pinecone_api_key = os.environ.get('pinecone_api_key')
os.environ["pinecone_api_key"] = pinecone_api_key

extracted_data = load_pdf_file(data='Data/')
text_chunks = text_split(extracted_data)
embeddings = download_embed_model()

pc = pinecone(api_key=pinecone_api_key)

index_name = "medicalbot"

pc.create_index(
    name=index_name,
    dimension = 384, # Replace with your model dimensions
    metric = "cosine", # Replace with your model metric
    spec = ServerlessSpec(
        cloud = "aws",
        region = "us-east-1"
    ) 
)

docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks,
    index_name = index_name,
    embedding = embeddings
)