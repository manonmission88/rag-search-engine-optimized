#!/usr/bin/env python3
import argparse
from lib.search_keyword import search_command, build_command, tf_command
def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("build", help="Build inverted index")
 
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    tf_parser = subparsers.add_parser("tf", help="Count the frequencies")
    tf_parser.add_argument("doc_id", type=int, help="Doc Id ")
    tf_parser.add_argument("term", type=str, help="Term for the frequency")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("building inverted idx")
            build_command()
            print("inverted index build successfully")
            
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            all_results = search_command(args.query)
            for idx,res in enumerate(all_results,1):
                print(f"{idx}. ({res['id']}) {res['title']}")
        case "tf":
            tf = tf_command(args.doc_id,args.term)
            print(f"Term frequency of '{args.term}' in document '{args.doc_id}': {tf}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()