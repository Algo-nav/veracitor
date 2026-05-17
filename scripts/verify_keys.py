import os
from dotenv import load_dotenv

load_dotenv()

def verify_keys():
    results = {}

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    results["ANTHROPIC_API_KEY"] = "ok" if anthropic_key and not anthropic_key.startswith("your_") else "missing or placeholder"

    tavily_key = os.getenv("TAVILY_API_KEY")
    results["TAVILY_API_KEY"] = "ok" if tavily_key and not tavily_key.startswith("your_") else "missing or placeholder"

    print("\nKey verification results:")
    for key, status in results.items():
        symbol = "+" if status == "ok" else "x"
        print(f"  [{symbol}] {key}: {status}")

    all_ok = all(v == "ok" for v in results.values())
    print("\nStatus:", "all keys loaded" if all_ok else "one or more keys missing")
    return all_ok

if __name__ == "__main__":
    verify_keys()