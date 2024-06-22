## Copywrite
* NarrateAR
* Contact for inquiry: Sarah Daeun Kim (daeun@narratear.com)

- **documents**: This folder contains input files (pdf)
- **demos**: This folder contains the demo short video
- **upload.py**: This code reads the file or files in the default folder('documents'), and indexes and stores the file/files in a vector DB ('Pipecone').
- **requirement.txt**: A list of packages or libraries needed to work on a project that can all be installed with the file.


#### Getting start
1. conda activate 'rag'
2. python3 upload.py / query.py

1. Upload pdf: parameters - *chunk size* and *chunk overlap*
2. Request the query: parameter -  *top-k*

#### Arguments
    parser = argparse.ArgumentParser(description= 'Process the pdf file for uploading the file to Pinecone (Vector DB)')
    parser.add_argument('--file_name', type=str, default= None, help='A path of input file')
    parser.add_argument('--chunck_size', type=int, default=200, help='Enter the chunck size over 100 range')
    parser.add_argument('--chunck_overlap', type=float, default=0.25, help='The portion of the overlap chunks: 25% = 0.25 range[0,1]')
    parser.parse_args()

#### 1. Single file upload
The file have to store in 'documents' folder.

>>> conda activate rag
>>> python3 upload.py --file_name sample.pdf

#### 2. With optional arguments
 
>>> python upload.py --name_space test_case --chunck_size 100 --chunk_overlap 0.20 # overlap = 100*0.2 = 20.

* '--file_name': *[Optional]* Enter the PDF file name. The file have to store in 'documents/'.  If do not enter it, the code will **Reads the files in the documents folder**

    >>> python upload.py --file_name sample.pdf

* '--chunk_size': *[Optional]* Enter the chunk size as integer. The default is 200

    >>> python upload.py --chunk_size 100

* '--chunk_overlap': *[Optional]* Enter the ratio of chunk overlap ranging [0,1]. The default is 0.25
 
    >>> python upload.py --chunck_overlap 0.25


#### Steps of the sub-task
1. Read the query : Read a query + embedding the query with the same model
2. Retrieval k-top chunks: The default of k-top: 5
3. Language Model: a small pre-trained language model by using Llma-CPP with Hugging face- `llama-2-13b-chat.Q4_0.gguf`


#### Arguments
    parser.add_argument('--question', type=str, default= None, required=True, help='A query')
    parser.add_argument('--top_k', type=int, default=5, help='top_k')
Embedding output dimension = 3072 : large (latest one)
#### 1. Answer generation with (top-k= 5) based on the query
>>> python query.py --question "What is ~?"

#### 2. Answer generation with (user-defined top-k= 5) based on the query
>>> python query.py --question "What is the attention model?" --top_k 8
