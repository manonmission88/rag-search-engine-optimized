import argparse
from lib.augmented_generation import (rag_command, 
                                      summarize_command,
                                      citations_command,
                                      question_command)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    
    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize Search Results")
    summarize_parser.add_argument("query", type = str, help = "User Search QUery")
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to store",
    )
    
    citations_parser = subparsers.add_parser(
        "citations", help="Add source of truth to the LLM generated output")
    citations_parser.add_argument("query", type = str, help = "User Search QUery")
    citations_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to store",
    )
    
    question_parser = subparsers.add_parser(
        "question", help="Conversational question answering")
    question_parser.add_argument("question", type=str, help="User question")
    question_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search results to use",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            results = rag_command(query)
            search_results = results.get("search_results",[])
            print("Search Results")
            for result in search_results:
                print(f"    - {result[1]['title']}")
            print("RAG Response:")
            print(f"{results.get('answer','')}")
        case "summarize":
            query = args.query
            results = summarize_command(query)
            search_results = results.get("search_results",[])
            print("Search Results")
            for result in search_results:
                print(f"    - {result[1]['title']}")
            print("LLM Summary:")
            print(f"{results.get('answer','')}")
        case "citations":
            query = args.query
            results = citations_command(query)
            search_results = results.get("search_results",[])
            print("Search Results")
            for result in search_results:
                print(f"    - {result[1]['title']}")
            print("LLM Summary:")
            print(f"{results.get('answer','')}")
        case "question":
            question = args.question
            limit = args.limit
            results = question_command(question, limit=limit)
            search_results = results.get("search_results", [])
            print("Search Results:")
            for _, result in search_results:
                print(f"  - {result['title']}")
            print("\nAnswer:")
            print(f"{results.get('answer','')}")
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()