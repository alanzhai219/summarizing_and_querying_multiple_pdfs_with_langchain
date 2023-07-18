from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain import OpenAI, PromptTemplate
import glob
from langchain.indexes import VectorstoreIndexCreator
import argparse

llm = OpenAI(temperature=0.2)
def summarize_single_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file)
    docs = loader.load_and_split()
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    print("Summary for: ", pdf_file)
    print(summary)
    print("\n")
    return summary

def summarize_pdfs_from_folder(pdfs_folder):
    summaries = []
    for pdf_file in glob.glob(pdfs_folder + "/*.pdf"):
        summary = summarize_single_pdf(pdf_file)
        summaries.append(summary)
    return summaries

def custom_summary(pdf_folder, custom_prompt):
    summaries = []
    for pdf_file in glob.glob(pdf_folder + "/*.pdf"):
        loader = PyPDFLoader(pdf_file)
        docs = loader.load_and_split()
        prompt_template = custom_prompt + """

        {text}

        SUMMARY:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm, chain_type="map_reduce", 
                                    map_prompt=PROMPT, combine_prompt=PROMPT)
        summary_output = chain({"input_documents": docs},return_only_outputs=True)["output_text"]
        summaries.append(summary_output)
        
    return summaries

# Save all summaries into one .txt file
def export(summaries):
    with open("summaries.txt", "w") as f:
        for summary in summaries:
            f.write(summary + "\n"*3)

def run_summary(file, dump):
    if file.endswith(".pdf"):
        summary = summarize_single_file(file)
        summaries = [summary]
    else:
        summaries = summarize_pdfs_from_folder(file)

    if dump is not None:
        export(summaries)


def run_query(file):
    loader = PyPDFDirectoryLoader("./pdfs/")
    
    docs = loader.load()
    
    # Create the vector store index
    index = VectorstoreIndexCreator().from_loaders([loader])

def parse():
    parser = argparse.ArgumentParser(description='Summarize and Query PDFs')
    parser.add_argument('--file', type=str, required=True, help='pass the PDF file or folder')
    parser.add_argument('--mode', type=str, default="summary", help="modes of analysis PDFs. summary or query")
    parser.add_argument('--dump', type=str, default=None, help="write the results to offline file")
    args = parser.parse_args()
    return args

def main():
    args = parse()
    mode = args.mode
    file = args.file
    dump = args.dump
    if mode == "summary":
        print("summary")
        run_summary(file, dump)
    elif mode == "query":
        print("query")
    else:
        raise RuntimeError("Doesn't support the mode of {}".format(mode))
    pass

if __name__ == "__main__":
    main()
