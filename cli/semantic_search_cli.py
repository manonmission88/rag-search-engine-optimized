#!/usr/bin/env python3

import argparse
from lib.semantic_search import (verify_model, 
                                 embed_text, 
                                 verify_embeddings, 
                                 embed_query_text,
                                 search_command,
                                 chunk_command,
                                 semantic_chunk_command,
                                 DEFAULT_CHUNK_SIZE) 

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
    search_parser.add_argument("--limit", type=int, nargs='?', default= 5, help="Maximum number of results (default: 5)")
    
    chunk_parser = subparsers.add_parser("chunk", help="Chunk the documents")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default= DEFAULT_CHUNK_SIZE, help="Maximum number of results (default: 200)")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap value")
    
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Semantic Chunk the documents")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default= 4, help="Maximum number of results (default: 200)")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap value")
    
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
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()