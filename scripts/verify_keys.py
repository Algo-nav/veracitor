import os
from dotenv import load_dotenv

load_dotenv()

def verify_keys():
    results = {}

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key and not anthropic_key.startswith("your_"):
        results["ANTHROPIC_API_KEY"] = "ok"
    else:
        results["ANTHROPIC_API_KEY"] = "missing or placeholder"

    print("\nKey verification results:")
    for key, status in results.items():
        symbol = "+" if status == "ok" else "x"
        print(f"  [{symbol}] {key}: {status}")

    all_ok = all(v == "ok" for v in results.values())
    print("\nStatus:", "all keys loaded" if all_ok else "one or more keys missing")
    return all_ok

if __name__ == "__main__":
    verify_keys()
