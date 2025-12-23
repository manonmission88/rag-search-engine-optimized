import argparse
from lib.hybrid_search import (normalize_command,
                               weighted_search_command,
                               rrf_search_command)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    
    normalize_parser = subparser.add_parser("normalize", help="Normalize the given values")
    normalize_parser.add_argument("scores", nargs="+", type=float, help="List of scores")
    
    weighted_search_parser = subparser.add_parser("weighted-search", help="Weighted hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="Search Query")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 (default: 0.5)")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Number of results (default: 5)")
    
    rrf_search_parser = subparser.add_parser("rrf-search", help="RRF search with ranking")
    rrf_search_parser.add_argument("query", type=str, help="Search Query")
    rrf_search_parser.add_argument("-k", type=int, default=60, help="RRF constant (default: 60)")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell","rewrite", "expand"], help="Fix the typos")
    rrf_search_parser.add_argument("--rerank-method", type = str, choices = ["individual", "batch", "cross_encoder"], help = "Re Ranking using LLM" )
    rrf_search_parser.add_argument("--limit", type=int, default=10, help="Number of results (default: 10)")
    
    
    
    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case "weighted-search":
            search_results = weighted_search_command(args.query, args.alpha, args.limit)
            
            for idx, data in enumerate(search_results, 1):
                print(f"{idx}.  {data['title']}")
                print(f"    Hybrid Score: {data['hybrid_score']:.4f}")
                print(f"    BM25: {data['bm25_score']:.4f}, Semantic: {data['semantic_score']:.4f}")
                print(f"    Description: {data['document'][:100]}...")
        case "rrf-search":
            search_results = rrf_search_command(args.query, args.k, args.enhance, args.rerank_method, args.limit)

            # Print header with query enhancement info
            if search_results["enhance_method"]:
                print(f"Enhanced query ({search_results['enhance_method']}): '{search_results['original_query']}' -> '{search_results['query']}'\n")
            
            # Print reranking info if applicable
            if search_results.get("reranked") and search_results.get("rerank_method"):
                rerank_limit = args.limit * 5
                print(f"Reranking top {rerank_limit} results using {search_results['rerank_method']} method...\n")
            
            print(f"Reciprocal Rank Fusion Results for '{search_results['query']}' (k={search_results['k']}):")
            
            for idx, (doc_id, data) in enumerate(search_results["results"], 1):
                bm25_rank = data['bm25_rank'] if data['bm25_rank'] is not None else '-'
                semantic_rank = data['semantic_rank'] if data['semantic_rank'] is not None else '-'
                print(f"{idx}. {data['title']}")
                
                # Show cross-encoder score if available
                if 'cross_encoder_score' in data:
                    print(f"   Cross Encoder Score: {data['cross_encoder_score']:.3f}")
                    print(f"   RRF Score: {data['rrf_score']:.3f}")
                else:
                    print(f"   RRF Score: {data['rrf_score']:.4f}")
                    print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                
                # Only show preview if no cross-encoder reranking
                if 'cross_encoder_score' not in data:
                    print(f"   {data['document'][:100]}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
