#!/usr/bin/env python3

import argparse
from lib.semantic_search import (verify_model, 
                                 embed_text, 
                                 verify_embeddings, 
                                 embed_query_text,
                                 search_command, 
                                 LIMIT) 

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help = "Available Commands")
    subparsers.add_parser("verify", help = "Verify Model")
    
    embed_text_parser = subparsers.add_parser("embed_text",help = "Embedding text")
    embed_text_parser.add_argument("text", type =str, help="Text To Embed" )
    
    subparsers.add_parser("verify_embeddings",help = "verify if embeddings is done")
    
    embed_query_parser = subparsers.add_parser("embedquery",help = "Embedding Query")
    embed_query_parser.add_argument("query", type = str, help = "Query to embed")
    
    search_parser = subparsers.add_parser("search", help="Search Query")
    search_parser.add_argument("query", type=str, help="Query to search")
    search_parser.add_argument("--limit", type=int, nargs='?', default=LIMIT, help="Maximum number of results (default: 5)")
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_command(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()