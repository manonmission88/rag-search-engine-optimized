import argparse
import mimetypes
from lib.image_query_gen import (generate_image_enhanced_query_command)

def main():
    parser = argparse.ArgumentParser(description="Describe Image Cli")
    parser.add_argument(
        "--image",
        type=str,
        help="Input Image",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="text query",
    )

    args = parser.parse_args()
    
    # The function handles printing internally
    generate_image_enhanced_query_command(args.image, args.query)
        
if __name__ == "__main__":
    main()
    
    