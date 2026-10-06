from search import build_vector_store, build_prompt, build_llm, build_chain, search_prompt 

def main():
    n_results = 10
    vector_store = build_vector_store()
    prompt = build_prompt()
    llm = build_llm()
    chain = build_chain(prompt=prompt, llm=llm)
    question = ''
    answer = search_prompt(vector_store=vector_store,
                          question=question,
                          n_results=n_results,
                          chain=chain,
                          )
                          

    if not answer:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return
    
    pass

if __name__ == "__main__":
    main()