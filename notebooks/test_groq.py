# notebooks/test_groq.py
# Test de l'API Groq avec Llama 3
import os
from dotenv import load_dotenv
from groq import Groq

# Charger la clé depuis .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERREUR : GROQ_API_KEY non trouvee dans .env")
    exit()

# Créer le client Groq
client = Groq(api_key=api_key)

# Premier appel : question simple
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system",
         "content": "Tu es un assistant medical senegalais. "
                    "Reponds en francais simple. "
                    "Maximum 3 phrases."},
        {"role": "user",
         "content": "Quels sont les symptomes du paludisme ?"}
    ],
    max_tokens=200,
    temperature=0.3
)

print("=== Reponse de Llama 3 ===")
print(response.choices[0].message.content)
print(f"\nTokens utilises : {response.usage.total_tokens}")

response2 = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system",
         "content": """Tu es un assistant medical senegalais.
                       Tu recois un diagnostic et des donnees patient.
                        Explique le resultat en francais simple,
                        comme un medecin parlerait a son patient.
                        Sois rassurant mais recommande une consultation.
                        Maximum 3 phrases.
                        Ne fais JAMAIS de diagnostic toi-meme."""},
        {"role": "user",
         "content": """Patient : Femme, 28 ans, region Dakar
                        Symptomes : temperature 39.5, toux, fatigue, maux de tete
                        Diagnostic du modele : paludisme (probabilite 72%)
                        Explique ce resultat au patient."""}
    ],
    max_tokens=200,
    temperature=0.3
)

print("=== Explication SenSante ===")
print(response2.choices[0].message.content)

# ===== EXERCICE 1 : Prompt en Wolof =====
response_wolof = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system",
         "content": """Tu es un assistant medical senegalais.
Tu reponds en melange de francais et de wolof simple,
comme un agent de sante parlerait a un patient au Senegal.
Utilise des mots wolof simples comme : sa yaram bi dafa tang (ton corps a de la fievre), nga am toux (tu as de la toux), nga fatig (tu es fatigue), nga am maux tete (tu as mal a la tete), nga am frissons (tu as des frissons), nga am nausee (tu as des nausees),
dagay dem si docteur bi (aller chez le medecin).
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-meme."""},
        {"role": "user",
         "content": """Patient : Femme, 28 ans, region Dakar
Symptomes : temperature 39.5, toux, fatigue, maux de tete
Diagnostic du modele : paludisme (probabilite 72%)
Explique ce resultat au patient."""}
    ],
    max_tokens=200,
    temperature=0.3
)
print("\n=== Exercice 1 : Explication en Wolof/Francais ===")
print(response_wolof.choices[0].message.content)

# ===== EXERCICE 2 : Tester la temperature =====
for temp in [0.0, 0.5, 1.0]:
    response_temp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system",
             "content": "Tu es un assistant medical senegalais. "
                        "Reponds en francais simple. Maximum 2 phrases."},
            {"role": "user",
             "content": """Patient : Femme, 28 ans, region Dakar
Diagnostic du modele : paludisme (probabilite 72%)
Explique ce resultat au patient."""}
        ],
        max_tokens=150,
        temperature=temp
    )
    print(f"\n=== Exercice 2 : Temperature={temp} ===")
    print(response_temp.choices[0].message.content)