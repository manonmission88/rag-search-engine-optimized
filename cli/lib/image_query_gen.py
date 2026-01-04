import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)
model = "meta-llama/llama-4-scout-17b-16e-instruct"
def generate_image_enhanced_query(image_path: str, query: str) -> str:
    """
    Generate an enhanced search query by analyzing an image with Groq.
    
    Args:
        image_path: Path to the image file
        query: Original text query
        
    Returns:
        Enhanced query string
    """
    # Open the image file in binary mode and read contents
    with open(image_path, "rb") as f:
        img = f.read()
    
    # Encode image to base64
    img_base64 = base64.b64encode(img).decode('utf-8')
    
    # Determine MIME type based on file extension
    if image_path.lower().endswith('.png'):
        mime = 'image/png'
    elif image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg'):
        mime = 'image/jpeg'
    elif image_path.lower().endswith('.webp'):
        mime = 'image/webp'
    else:
        mime = 'image/jpeg'  # default
    
    # System prompt for analyzing the image and rewriting the query
    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""
    
    # Build request with parts containing system prompt, image data, and text query
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{img_base64}"
                        }
                    },
                    {"type": "text", "text": query.strip()}
                ]
            }
        ],
        temperature=0.7,
        max_tokens=1024
    )
    
    # Print the rewritten query and total tokens used
    rewritten_query = response.choices[0].message.content.strip()
    print(f"Rewritten query: {rewritten_query}")
    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")
    
    return rewritten_query


def generate_image_enhanced_query_command(image_path,query):
    return generate_image_enhanced_query(image_path,query)
