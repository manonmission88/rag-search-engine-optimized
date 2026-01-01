import argparse
from lib.augmented_generation import rag_command


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            results = rag_command(query)
            search_results = results.get("search_results",[])
            print("Search Results")
            for result in search_results:
                print(f"    - {result[1]["title"]}")
            print("RAG Response:")
            print(f"{results.get("answer","")}")
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()