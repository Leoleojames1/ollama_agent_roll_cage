""" Local_Sledge_Embedding.py
    
        A Local Nomic Embedding agent frame-work to be modulated for the lablabai hackathon.
    
    Local_Sledge_Embedding Written by @smokeybandit a.k.a. @Wayne Sletcher,
    
    Modified as Agent_Flow.py by
    @Borch, @Moez Ali Khan, @Faiqai 7/19/2024
     
"""

import ollama
import chromadb
import psycopg
import ast
from colorama import Fore
from tqdm import tqdm
from psycopg.rows import dict_row

#TODO LIST
# 1. CREWAI
# 2. DUCKDUCKGO SEARCH API
# 3. GOOGLE SPEECH RECOGNITION
# 4. COQUI TEXT TO SPEECH
# 5. VERCEL JAVASCRIPT WEBUI
# 6. CLASS STRUCTURE TO PROGRAM

client = chromadb.Client()

system_prompt = (
    "You are an AI assistant that has memory of every conversation you have ever had with this user. "
    "On every prompt from the user, the system has checked for any relevant messages you have had with the user. "
    "If any embedded previous conversations are attached, use them for context to responding to the user, "
    "if the context is relevant and useful to responding. If the recalled conversations are irrelevant, "
    "disregard speaking about them and respond normally as an AI assistant. Do not talk about recalling conversations. "
    "Just use any useful data from the previous conversations and respond normally as an intelligent AI assistant."
)
convo = [{'role': 'system', 'content': system_prompt}]

DB_PARAMS ={
    'dbname': 'memory_agent_2',
    'user': 'postgres',
    'password': 'admin',
    'host': 'localhost',
    'port': '5432'
}

def connect_db():
    conn = psycopg.connect(**DB_PARAMS)
    return conn 

def fetch_conversations():
    conn = connect_db()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute('SELECT * FROM conversations')
        conversations = cursor.fetchall()
    conn.close()
    return conversations

def store_conversations(prompt, response):
    try:
        conn = connect_db()
        with conn.cursor() as cursor:
            cursor.execute(
                'INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, %s, %s)',
                (prompt, response)
            )
            conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error storing conversation: {e}")
        return False

def remove_last_conversation():
    conn = connect_db()
    try:
        with conn.cursor() as cursor:
            cursor.execute('DELETE FROM conversations WHERE id = (SELECT MAX(id) FROM conversations)')
        conn.commit()  # Move this line outside of the 'with' block
        print(Fore.YELLOW + "\nLast conversation removed successfully.\n")
    except Exception as e:
        print(Fore.RED + f"\nError removing last conversation: {e}\n")
    finally:
        conn.close() 

def stream_response(prompt, store=False):
    response = ''
    stream = ollama.chat(model='phi3', messages=convo, stream=True)
    print(Fore.LIGHTGREEN_EX + '\nASSISTANT:')

    for chunk in stream:
        content = chunk['message']['content']
        response += content
        print(content, end='', flush=True)

    print('\n')
    
    if store:
        store_conversations(prompt=prompt, response=response)
    
    convo.append({'role': 'assistant','content':response})

def create_vector_db(conversations):
    vector_db_name = 'conversations'

    try:
        client.delete_collection(name=vector_db_name)
    except ValueError:
        pass
    
    vector_db = client.create_collection(name=vector_db_name)

    for c in conversations:
        serialized_convo = f"prompt: {c['prompt']} response: {c['response']}"
        response = ollama.embeddings(model='nomic-embed-text', prompt=serialized_convo)
        embedding = response['embedding']

        vector_db.add(
            ids=[str(c['id'])],
            embeddings=[embedding],
            documents=[serialized_convo]
        )

def retrieve_embeddings(queries, results_per_query=2):
    embeddings = set()

    for query in tqdm(queries,desc='Processing queries to vector database'):
        response = ollama.embeddings(model='nomic-embed-text', prompt=query)
        query_embedding = response['embedding']

        vector_db = client.get_collection(name='conversations')
        results = vector_db.query(query_embeddings=[query_embedding], n_results=results_per_query)
        best_embeddings = results['documents'][0]

        for best in best_embeddings:
            if best not in embeddings:
                if 'yes' in classify_embedding(query=query, context=best):
                    embeddings.add(best)

    return embeddings

def create_queries(prompt):
    query_msg = (
        "You are a first principle reasoning search query AI agent. "
        "Your list of search queries will be ran on an embedding database of all your conversations "
        "you have ever had with the user. With first principles create a Python list of queries to "
        "search the embeddings database for any data that would be necessary to have access to in "
        "order to correctly respond to the prompt. Your response must be a Python list with no syntax errors. "
        "Do not explain anything and do not ever generate anything but a perfect syntax Python list"
    )
    query_convo = [
        {'role': 'system', 'content': query_msg},
        {'role': 'user', 'content': 'What are some effective strategies for improving time management?'},
        {'role': 'assistant', 'content': '["What are the user\'s current time management habits?", "What specific areas of time management does the user struggle with?", "Has the user tried any time management techniques before?", "What are the user\'s main daily responsibilities or tasks?"]'},
        {'role': 'user', 'content': prompt}
]

    response = ollama.chat(model='phi3', messages=query_convo)
    print(Fore.YELLOW + f'\nVector database queries: {response["message"]["content"]} \n')

    try:
        evaluated_content = ast.literal_eval(response['message']['content'])
        if isinstance(evaluated_content, list):
            return evaluated_content
        else:
            print("Warning: Evaluated content is not a list. Falling back to default.")
            return [prompt]
    except (SyntaxError, ValueError) as e:
        print(f"Error evaluating response: {e}. Falling back to default.")
        return [prompt]

def classify_embedding(query, context):
    classify_msg = (
    "You are an embedding classification AI agent. Your input will be a prompt and one embedded chunk of text. "
    "You will not respond as an AI assistant. You only respond 'yes' or 'no'. "
    "Determine whether the context contains data that directly is related to the search query. "
    "If the context is seemingly exactly what the search query needs, respond 'yes'. If it is anything but directly "
    "related respond 'no'. Do not respond 'yes' unless the content is highly relevant to the search query."
)
    classify_convo = [
        {'role': 'system', 'content': classify_msg},
        {'role': 'user', 'content':f'SEARCH QUERY: What is the users name? \n\nEMBEDDED CONTEXT: You are [NAME]. How can I help you today [NAME]?'},
        {'role': 'user', 'content': 'yes'},
        {'role': 'user', 'content': f'SEARCH QUERY: Llama3 Python Voice Assistant \n\nEMBEDDED CONTEXT: Siri is a voice assistant on Apple iOS and Mac OS'},
        {'role': 'user', 'content': 'no'},
        {'role': 'user', 'content': f'SEARCH QUERY: {query} \n\nEMBEDDED CONTEXT: {context}'}
    ]

    response = ollama.chat(model='phi3', messages=classify_convo)

    return response['message']['content'].strip().lower()

def recall(prompt):
    queries = create_queries(prompt=prompt)
    embeddings = retrieve_embeddings(queries=queries)
    convo.append({'role': 'user', 'content': f'MEMORIES:{embeddings} \n\n USER PROMPT: {prompt}'})
    print(f'\n{len(embeddings)} message:response embeddings added for context.')

conversations = fetch_conversations()
create_vector_db(conversations=conversations)

while True:
    prompt = input(Fore.WHITE + 'USER: \n')
    
    if prompt.lower().startswith('/recall'):
        prompt = prompt[7:].strip()
        recall(prompt=prompt)
        stream_response(prompt=prompt, store=False)
    elif prompt.lower().startswith('/forget'):
        remove_last_conversation()
        if len(convo) >= 2:
            convo = convo[:-2]  # Remove the last user input and assistant response from the conversation history
        print(Fore.YELLOW + '\nLast conversation forgotten.\n')
    elif prompt.lower().startswith('/memorize'):
        memory_content = prompt[9:].strip()
        if store_conversations(prompt=memory_content, response='Memory stored.'):
            print(Fore.YELLOW + '\nMemory stored successfully.\n')
            convo.append({'role': 'user', 'content': f"Remember this: {memory_content}"})
            convo.append({'role': 'assistant', 'content': "I've remembered that information."})
        else:
            print(Fore.RED + '\nFailed to store memory.\n')
    else:
        convo.append({'role': 'user', 'content': prompt})
        stream_response(prompt=prompt, store=True)