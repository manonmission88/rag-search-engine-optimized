import argparse
import json 
from lib.search_utils import load_golden_dataset
from lib.evaluation import evaluate_search

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    datasets = load_golden_dataset()
    
    results = evaluate_search(datasets, limit)
    print(f"k = {limit}")
    for result in results:
        print(f"-Query: {result['query']}")
        print(f"    -Precision@{limit}: {result['precision']:.4f}")
        print(f"    -Retrieved: {', '.join(result['retrieved'])}")
        print(f"    -Relevant: {', '.join(result['relevant'])}")
        print()
    


if __name__ == "__main__":
    main()
    