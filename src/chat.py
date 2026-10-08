from dotenv import load_dotenv
from ingest import require_env
from search import build_vector_store, build_prompt, build_llm, build_chain, search_prompt


load_dotenv()

N_RESULTS = require_env("N_RESULTS")


def main():
    try:
        n_results = int(N_RESULTS)
        vector_store = build_vector_store()
        prompt = build_prompt()
        llm = build_llm()
        chain = build_chain(prompt=prompt, llm=llm)
    except Exception as e:
        print(f"Program stopped given this error: {e}")
        return

    print("Bem-vindo ao sistema de consultas da SuperTech Brazil.\nO nosso assitente vai atender o seu requerimento.\nEscreva 'sair' pra terminar a sessão")

    try:
        while True:
            try:
                question = input("Escreva a sua pergunta: ")
                if question.strip().lower() == "sair":
                    break
                answer = search_prompt(
                    vector_store=vector_store,
                    question=question,
                    n_results=n_results,
                    chain=chain,
                )
                if not answer:
                    print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
                    return

                print(answer)
            except Exception as e:
                print(f"An error occurred during the retrieval and llm process: {e}")
    except (KeyboardInterrupt, EOFError):
        print()

    print("Foi um prazer. Até a próxima...")


if __name__ == "__main__":
    main()
