import os
os.environ["OPENAI_API_KEY"] = "YOUR_OPEN_AI_KEY"

import subprocess
import uuid
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough

def format_element(
    element: Element, keep_refs: bool = False, latex_env: bool = False
) -> List[str]:
    """
    Formats a given Element into a list of formatted strings.
    Args:
        element (Element): The element to be formatted.
        keep_refs (bool, optional): Whether to keep references in the formatting. Default is False.
        latex_env (bool, optional): Whether to use LaTeX environment formatting. Default is False.
    Returns:
        List[str]: A list of formatted strings representing the formatted element.
    """
    if isinstance(element, Table):
        parts = [
            "[TABLE%s]\n\\begin{table}\n"
            % (str(uuid4())[:5] if element.id is None else ":" + str(element.id))
        ]
        parts.extend(format_children(element, keep_refs, latex_env))
        caption_parts = format_element(element.caption, keep_refs, latex_env)
        remove_trailing_whitespace(caption_parts)
        parts.append("\\end{table}\n")
        if len(caption_parts) > 0:
            parts.extend(caption_parts + ["\n"])
        parts.append("[ENDTABLE]\n\n")
        return parts

def june_run_nougat(file_path, output_dir):
    # Run Nougat and store results as Mathpix Markdown
    cmd = ["nougat", file_path, "-o", output_dir, "-m", "0.1.0-base", "--no-skipping"]
    res = subprocess.run(cmd) 
    if res.returncode != 0:
        print("Error when running nougat.")
        return res.returncode
    else:
        print("Operation Completed!")
        return 0
 
def june_get_tables_from_mmd(mmd_path):
    f = open(mmd_path)
    lines = f.readlines()
    res = []
    tmp = []
    flag = ""
    for line in lines:
        if line == "\\begin{table}\n":
            flag = "BEGINTABLE"
        elif line == "\\end{table}\n":
            flag = "ENDTABLE"
        
        if flag == "BEGINTABLE":
            tmp.append(line)
        elif flag == "ENDTABLE":
            tmp.append(line)
            flag = "CAPTION"
        elif flag == "CAPTION":
            tmp.append(line)
            flag = "MARKDOWN"
            print('-' * 100)
            print(''.join(tmp))
            res.append(''.join(tmp))
            tmp = []
    return res
    
file_path = "YOUR_PDF_PATH"
output_dir = "YOUR_OUTPUT_DIR_PATH"
 
if june_run_nougat(file_path, output_dir) == 1:
    import sys
    sys.exit(1)
 
mmd_path = output_dir + '/' + os.path.splitext(file_path)[0].split('/')[-1] + ".mmd" 
tables = june_get_tables_from_mmd(mmd_path)

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name = "summaries", embedding_function = OpenAIEmbeddings())
store = InMemoryStore()   # The storage layer for the parent documents
id_key = "doc_id"
 
retriever = MultiVectorRetriever(   # The retriever (empty to start)
    vectorstore = vectorstore,
    docstore = store,
    id_key = id_key,
    search_kwargs={"k": 1} # Solving Number of requested results 4 is greater than number of elements in index..., updating n_results = 1
)
 
prompt_text = """You are an assistant tasked with summarizing tables and text. \ 
Give a concise summary of the table or text. The table is formatted in LaTeX, and its caption is in plain text format: {element}  """
prompt = ChatPromptTemplate.from_template(prompt_text)
 
# Summary chain
model = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo")
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
# Get table summaries
table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
 
# Add tables
table_ids = [str(uuid.uuid4()) for _ in tables]
summary_tables = [
    Document(page_content = s, metadata = {id_key: table_ids[i]})
    for i, s in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, tables)))

# Prompt template
template = """Answer the question based only on the following context, which can include text and tables, there is a table in LaTeX format and a table caption in plain text format:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo")   # LLM
# Simple RAG pipeline
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
 
print(chain.invoke("when layer type is Self-Attention, what is the Complexity per Layer?"))  # Query about table 1
print(chain.invoke("Which parser performs worst for BLEU EN-DE"))  # Query about table 2
print(chain.invoke("Which parser performs best for WSJ 23 F1"))  # Query about table
