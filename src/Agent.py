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


class Agent():

    def __init__(self, model="meta-llama/llama-3.3-70b-instruct:free", system_prompt=None):
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
        self.messages = [
            {"role": "system", "content": system_prompt}] if system_prompt else []

    def execute(self):
        if self.is_openai:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages
            )
            return response.choices[0].message.content
        else:
            response = self.client.chat(
                model=self.model,
                messages=self.messages
            )
            return response.message.content

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result


if __name__ == "__main__":
    print("\nChoose a model from the above list, press a number to select, or type your own model name:\n")
    for i, model in enumerate(ollama_models + openai_models):
        print(f"[{i}] {model}")

    selected_model = None
    while not selected_model:
        model_input = input("\nModel: ").strip()

        if not model_input:
            print("No model selected, please try again.")
            continue

        if not model_input.isdigit() and model_input not in ollama_models + openai_models:
            print("Model not found, please try again.")
            continue

        if not model_input.isdigit():
            selected_model = model_input
            break

        model_index = int(model_input)

        if model_index < 0 or model_index >= len(ollama_models) + len(openai_models):
            print("Invalid selection, please try again.")
            continue

        if 0 <= model_index < len(ollama_models):
            selected_model = ollama_models[model_index]
            break

        if 0 <= model_index - len(ollama_models) < len(openai_models):
            selected_model = openai_models[model_index - len(ollama_models)]

    print(f"Using model: {selected_model}")

    agent = Agent(selected_model, "你是一位精通大語言模型的軟體工程師，請用繁體中文回答。")

    while True:
        try:
            user_input = input("\nYou:\n ")
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            break

        if user_input.lower() == "q":
            break

        print("\nAgent:\n", agent(user_input))
