#!/usr/bin/env python3

import os
from google import genai


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("Set GEMINI_API_KEY first.")
    client = genai.Client()
    for model in client.models.list():
        name = getattr(model, "name", str(model))
        supported = getattr(model, "supported_actions", None)
        print(name, supported if supported is not None else "")


if __name__ == "__main__":
    main()
