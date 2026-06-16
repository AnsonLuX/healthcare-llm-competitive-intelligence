import requests

from src.config import Config


class LLMProviderError(Exception):
    """
    Raised when an LLM provider call fails.
    """


def call_llm(prompt: str) -> str:
    """
    Route the prompt to the selected LLM provider.

    Supported providers:
    - gemini
    - ollama
    - openai placeholder for future extension
    """
    provider = Config.LLM_PROVIDER.lower().strip()

    if provider == "gemini":
        return call_gemini(prompt)

    if provider == "ollama":
        return call_ollama(prompt)

    if provider == "openai":
        raise LLMProviderError(
            "OpenAI provider is reserved for future extension but is not implemented yet."
        )

    raise LLMProviderError(f"Unsupported LLM_PROVIDER: {Config.LLM_PROVIDER}")


def get_active_model_name() -> str:
    """
    Return the model name used by the selected provider.
    """
    provider = Config.LLM_PROVIDER.lower().strip()

    if provider == "gemini":
        return Config.GEMINI_MODEL

    if provider == "ollama":
        return Config.OLLAMA_MODEL

    if provider == "openai":
        return "openai-not-implemented"

    return "unknown-model"


def call_gemini(prompt: str) -> str:
    """
    Call Gemini API for text generation using the current Google GenAI SDK.
    """
    if not Config.GEMINI_API_KEY:
        raise LLMProviderError(
            "GEMINI_API_KEY is missing. Add it to your .env file or switch LLM_PROVIDER to ollama."
        )

    try:
        from google import genai
    except ImportError as exc:
        raise LLMProviderError(
            "google-genai is not installed. Run: pip install google-genai"
        ) from exc

    client = genai.Client(api_key=Config.GEMINI_API_KEY)

    try:
        response = client.models.generate_content(
            model=Config.GEMINI_MODEL,
            contents=prompt,
        )
    except Exception as exc:
        raise LLMProviderError(
            f"Gemini request failed for model '{Config.GEMINI_MODEL}'. "
            "Check that this model is available for your API key. "
            "You can also try GEMINI_MODEL=gemini-2.5-flash."
        ) from exc

    generated_text = getattr(response, "text", "")

    if not generated_text:
        raise LLMProviderError("Gemini returned an empty response.")

    return generated_text.strip()

def call_ollama(prompt: str) -> str:
    """
    Call local Ollama model.
    """
    payload = {
        "model": Config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(Config.OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise LLMProviderError(
            "Ollama request failed. Make sure Ollama is running and the model is pulled."
        ) from exc

    result = response.json()
    generated_text = result.get("response", "").strip()

    if not generated_text:
        raise LLMProviderError("Ollama returned an empty response.")

    return generated_text