
import argparse
from lib.hybrid_search import (normalize_command,
                               weighted_search_command)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    
    normalize_parser = subparser.add_parser("normalize", help="Normalize the given values")
    normalize_parser.add_argument("scores", nargs="+", type=float, help="List of scores")
    
    weighted_search_parser = subparser.add_parser("weighted-search", help="Normalize the given values")
    weighted_search_parser.add_argument("query",type=str,help="Search Query")
    weighted_search_parser.add_argument("--alpha", nargs="?", type=float, help="List of scores")
    weighted_search_parser.add_argument("--limit", nargs="?", type=int, help="List of scores")
    
    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case "weighted-search":
            search_results = weighted_search_command(args.query, args.alpha, args.limit)
            
            for idx,data in enumerate(search_results,1):
                print(f"{idx}.  {data["title"]}")
                print(f"    Hybrid Score: {data["hybrid_score"]:.4f}")
                print(f"    BM25: {data["bm25_score"]:.4f}, {data["semantic_score"]:.4f}")
                print(f"    Semantic Score: {data["document"][:100]}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
    