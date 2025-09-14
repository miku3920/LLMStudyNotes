import os
from dotenv import load_dotenv
import ollama
import openai

load_dotenv()

openai_api_url = os.getenv("OPENAI_API_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")

ollama_models = [x.model for x in ollama.list().models]
openai_models = [x.id for x in openai.OpenAI(
    base_url=openai_api_url,
    api_key=openai_api_key,
).models.list().data]


class Embedding:

    def __init__(self, model="bge-m3:567m"):
        if model not in ollama_models and model not in openai_models:
            raise ValueError(
                f"Model '{model}' not found in Ollama or OpenAI models.")

        if model in openai_models and not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables")

        if model in openai_models:
            self.is_openai = True
            self.client = openai.OpenAI(
                base_url=openai_api_url,
                api_key=openai_api_key,
            )
        else:
            self.is_openai = False
            self.client = ollama

        self.model = model

    def __call__(self, message):
        if self.is_openai:
            response = self.client.embeddings.create(
                model=self.model,
                input=message
            )
            return response.data[0].embedding
        else:
            response = self.client.embed(model=self.model, input=message)
            return response.embeddings[0]


if __name__ == "__main__":
    embedder = Embedding(model="bge-m3:567m")
    embedding = embedder("Hello, world!")
    print(embedding)
