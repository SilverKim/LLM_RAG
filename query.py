""" 
    Copyright: NarrateAR
    Contact: Sarah Daeun Kim (daeun@narratear.com) 
"""

import argparse
import os
from openai import OpenAI
import config
import upload

client = OpenAI()

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#Enter the API keys for accessing Pipecone
os.environ["MONGODB_URI"] = config.MONGODB_URI

def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def generate_response(prompt, max_tokens=384):#3072
    response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=1.0
    )
    return response.choices[0].message.content.strip()

def send_query(query, top_k=5):
    query_embedding = get_embedding(query)

    # Retrieve all stored embeddings from MongoDB
    stored_documents = list(upload.collection.find())
    stored_embeddings = np.array([doc['embedding'] for doc in stored_documents])

    # Compute cosine similarities
    similarities = cosine_similarity([query_embedding], stored_embeddings)[0]
    
    # Get top_k results
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_documents = [stored_documents[idx] for idx in top_indices]
    
    ## Generate new nodes for Vector Index
    _nodes= []
    print("Query: ", query ,"\n")
    
    context = "\n".join([doc['text'] for doc in top_documents])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    response = generate_response(prompt)

    # Calculate the similarity between the response and the query
    response_embedding = get_embedding(response)
    similarity = cosine_similarity([query_embedding], [response_embedding])[0][0]

    return response, similarity

if __name__== "__main__":
    parser = argparse.ArgumentParser(description= 'Process the pdf file for uploading the file to MongoDB (Vector DB)')
    parser.add_argument('--question', type=str, default= None, required=True, help='A query')
    parser.add_argument('--top_k', type=int, default=5, help='top_k')
    args = parser.parse_args()
    upload.init_mongoDB()

    #Initiate the vector space
    response, similarity= send_query(args.question, args.top_k)

    print("Answer: ")
    print(str(response))
    print("with similarity: ", similarity)
