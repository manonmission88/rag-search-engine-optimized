#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help = "Available Commands")
    subparsers.add_parser("verify", help = "Verify Model")
    
    embed_text_parser = subparsers.add_parser("embed_text",help = "Embedding text")
    embed_text_parser.add_argument("text", type =str, help="Text To Embed" )
    
    subparsers.add_parser("verify_embeddings",help = "verify if embeddings is done")
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()