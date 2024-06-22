import argparse
import os
import re
import sys
import config
from openai import OpenAI

client = OpenAI(api_key=config.OPENAI_API_KEY)

# Llama indexs frameworks
from llama_index.core import (Document, SimpleDirectoryReader)
from llama_index.core.node_parser import SentenceSplitter

# Initialize VectorStore
import pymongo
from pymongo import MongoClient

#class Upload:
os.environ["MONGODB_URI"] = config.MONGODB_URI
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

def read_data(file_path):
    #a. file case
    if file_path != None:
        file_path = os.path.join(file_path)

    if os.path.isfile(file_path):
        data_file = [file_path]
        docs = SimpleDirectoryReader(input_files = data_file).load_data()
    #b. folder case
    else:
        docs = SimpleDirectoryReader(file_path).load_data()

    print("Total pages read:", len(docs))
    #return documents
    return docs

# Define a function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text

def chunk(docs, chunk_size = 10000, chunk_overlap=0.25):
    #Chunk -> nodes
    chunk_overlap = int(chunk_size*chunk_overlap)
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if isinstance(docs[0],str):
        documents = [Document(text=t, metadata={"file_name":"test","file_path": "test_path", "page_label": "{}".format(_idx)}) for _idx,t in enumerate(docs)]
        nodes = splitter.get_nodes_from_documents(documents)
    else:
        nodes = splitter.get_nodes_from_documents(docs)
        print("Chunck size :", chunk_size, "Chunck overlap :", chunk_overlap)
    return nodes

def get_embedding(text, model="text-embedding-3-large"):
    #Embedding output dimension = 384 : ada
    #Embedding output dimension = 3072 : large (latest one)
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def embedding(nodes):
    embeddings_list = []

    #Check the node types
    if isinstance(nodes[0],str):
        for node in nodes:
            res = get_embedding(node)
            embeddings_list.append(res)
    else:
        for node in nodes:
            print("...Progressing the indexing data:", node.metadata['file_path'] + '...Page number: ' + node.metadata['page_label']+"\n")
            res = get_embedding(node.text)
            embeddings_list.append(res)
        return embeddings_list

def init_mongoDB():
    global collection
    mongo_client = MongoClient(config.MONGODB_URI)
    db = mongo_client['test']
    collection = db['dinosaur']

    try:
        mongo_client.admin.command('ping')
        print("You successfully connected to MongoDB!")
    except Exception as e:
        print(e)

def upsert(vectors):
    for vector in vectors:
        filter_query = {"_id": vector[0]}  # '_id' is the unique identifier for documents
        update_query = {
            "$set": {
                "embedding": vector[1],
                "text": vector[2]['text']
            }
        }
        collection.update_one(filter_query, update_query, upsert=True)


def upsert_data(docs, ch_size = 10000, ch_overlap =0.25):
    nodes= chunk(docs, ch_size, ch_overlap)
    embeded_text = embedding(nodes)
    ## upsert
    upsert(vectors=[(node.metadata['file_name'][:2]+node.metadata['page_label'], emb, {'text': node.text}) for node, emb in zip(nodes,embeded_text)])

def show_vectordb():
    print("Vector DB (MongoDB) - Number of Documents:", collection.count_documents({}))

def deletes():
    collection.delete_many({})

if __name__== "__main__":
    # parsing
    parser = argparse.ArgumentParser(description= 'Process the pdf file for uploading the file to vector DB')
    parser.add_argument('--file_name', type=str, default= None, help='A path of input file')
    parser.add_argument('--chunk_size', type=int, default=10000, help='Enter the chunk size over 100 range')
    parser.add_argument('--chunk_overlap', type=float, default=0.25, help='The portion of the overlap chunks: 25% = 0.25 range[0,1]')
    parser.add_argument('--delete', type=bool, default=False)
    parser.add_argument('--folder', type=str, default= './documents/', help='A folder path for input files')

    args = parser.parse_args()
    init_mongoDB()

    #Check the task
    if args.file_name is None and args.delete == False:
        deletes()
    elif args.file_name:
        #File read
        data = read_data(args.file_name)
        upsert_data(data, args.chunk_size, args.chunk_overlap)
    else:
        print("Choose the tasks - upload pdf or delete pdf")
