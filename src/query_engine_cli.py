from query_engine import load_rag_components, query_rag

def main():
    """
    Main function to run query engine in CLI for testing
    """
    
    print("Loading RAG components...")
    
    model, index, metadata = load_rag_components()
    
    if model is None or index is None or metadata is None:
        print("Failed to loading RAG Components. Exiting...")
        return
    
    print("\n----------------------------------------------------")
    print("RAG Query Enginer Test CLI")
    print("Type your question to query the mnodel. Type 'exit' to quit")
    print("------------------------------------------------------")
    

    while True:
        query = input("Query: ").strip()
        if query.lower() in {"exit"}:
            print("Exiting program...")
            break
        
        if query:
            response = query_rag(query, model, index, metadata)
            print("\n" + response)

if __name__ == "__main__":
    main()