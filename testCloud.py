from google import genai

client = genai.Client(
    vertexai=True,
    project="project-03f2d979-9de1-4903-886",
    location="global"  # zacznij od us-central1, tam jest najwięcej modeli
)

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents="Czym jest sztuczna inteligencja? Odpowiedz w 2 zdaniach."
)
print(response.text)