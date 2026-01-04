import argparse
from lib.multimodal_search import (verify_image_embedding, image_search_command)


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # verify_image_embedding command
    verify_parser = subparsers.add_parser(
        "verify_image_embedding",
        help="Verify image embedding generation by printing the embedding shape"
    )
    verify_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file"
    )
    
    # image_search command
    search_parser = subparsers.add_parser(
        "image_search",
        help="Search for movies using an image"
    )
    search_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file"
    )
    
    args = parser.parse_args()
    
    if args.command == "verify_image_embedding":
        verify_image_embedding(args.image_path)
    elif args.command == "image_search":
        results = image_search_command(args.image_path)
        for i, result in enumerate(results, 1):
            desc = result['description']
            if len(desc) > 100:
                desc = desc[:97] + "..."
            print(f"{i}. {result['title']} (similarity: {result['similarity']:.3f})")
            print(f"   {desc}")
            print()


if __name__ == "__main__":
    main()
