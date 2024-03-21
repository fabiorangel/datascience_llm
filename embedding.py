from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

load_dotenv()

f = open("assets/basetext.txt")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

fout = open("assets/embedding_coke2.txt", "w")

for line in f:
    vector = embeddings.embed_query(line)
    fout.write(str(vector).replace('[','').replace(']',''))
    fout.write("\n")
    time.sleep(1)

fout.close()
f.close()

