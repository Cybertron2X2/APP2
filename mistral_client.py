"""
mistral_client.py — Client Mistral v3 COMPLET
Toutes les fonctionnalités pédagogiques + prompts optimisés
"""

import os, json, re
from openai import OpenAI
from dotenv import load_dotenv
from dataset_loader import DioulaDataset

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MISTRAL_MODEL      = os.getenv("MISTRAL_MODEL", "MISTRAL_MODEL=google/gemma-2-9b-it:free")
APP_NAME           = os.getenv("APP_NAME", "Koumankan-Dioula-AI")
APP_URL            = os.getenv("APP_URL", "http://localhost:3000")
TEMPERATURE        = 0.7
MAX_TOKENS         = 800
NB_EXEMPLES_CTX    = 8

SYSTEM_BASE = """Tu es Koumankan, le meilleur professeur virtuel de langue Dioula (Dyula/Jula).
Le Dioula est une langue mandingue parlée en Côte d'Ivoire, Burkina Faso et Mali.

TES QUALITÉS :
- Expert linguistique du Dioula avec 20 ans d'expérience
- Pédagogue bienveillant et encourageant
- Tu utilises TOUJOURS des exemples réels du dataset d'entraînement
- Tu adaptes ton discours au niveau de l'apprenant
- Tes traductions sont précises et contextualisées

RÈGLES ABSOLUES :
1. Pour toute traduction : donner le Dioula EN PREMIER, puis la prononciation phonétique
2. Pour les quiz : format A/B/C/D strict avec la réponse indiquée clairement
3. Pour les dialogues : Dioula en gras, traduction française entre parenthèses
4. Toujours encourager l'apprenant ("Excellent !", "Bonne question !", "Félicitations !")
5. Mentionner toujours les variations régionales quand elles existent (CI vs Burkina vs Mali)
"""

class MistralDioula:

    def __init__(self, dataset: DioulaDataset | None = None):
        if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "sk-or-v1-VOTRE_CLE_ICI":
            raise ValueError(
                "❌ Clé API OpenRouter manquante !\n"
                "   → https://openrouter.ai/keys (gratuit)\n"
                "   → Mettre dans .env : OPENROUTER_API_KEY=sk-or-v1-..."
            )
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

        # Réutiliser le dataset si déjà chargé (évite le double chargement)
        if dataset is not None:
            self.db = dataset
            print("♻️  Dataset réutilisé (pas de double chargement)")
        else:
            print("📦 Chargement du dataset Dioula...")
            self.db = DioulaDataset()
            self.db.charger()

        self.system_prompt = SYSTEM_BASE + "\n\nEXEMPLES DE BASE DU DATASET :\n" + self._exemples_de_base()
        print(f"✅ Mistral prêt — {MISTRAL_MODEL} | {len(self.db.df)} exemples")

    def _exemples_de_base(self) -> str:
        exemples = self.db.df[self.db.df["dioula"].str.split().str.len().between(2, 5)].sample(min(25, len(self.db.df)))
        return "\n".join([f"  • {r['dioula']} = {r['francais']}" for _, r in exemples.iterrows()])

    def _appeler_mistral(self, messages: list) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=MISTRAL_MODEL, messages=messages,
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                extra_headers={"HTTP-Referer": APP_URL, "X-Title": APP_NAME},
            )
            return resp.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "401" in err: return "❌ Clé API invalide — vérifiez votre .env"
            if "429" in err: return "⏳ Limite atteinte — attendez quelques secondes"
            return f"❌ Erreur : {err}"

    def _contexte_dataset(self, requete: str, niveau: str, k: int = NB_EXEMPLES_CTX) -> str:
        exemples = self.db.chercher(requete, k=k)
        if not exemples: return ""
        lignes = "\n".join([f"  • {e['dioula']} = {e['francais']}" for e in exemples if e.get("score", 1) > 0])
        note_niveau = {
            "debutant":      "⚡ NIVEAU DÉBUTANT : mots très simples, beaucoup d'encouragements, explique tout.",
            "intermediaire": "📚 NIVEAU INTERMÉDIAIRE : vocabulaire enrichi, quelques règles grammaticales.",
            "avance":        "🎓 NIVEAU AVANCÉ : précision linguistique, nuances dialectales, grammaire complète.",
        }.get(niveau, "")
        return f"\n\n--- EXEMPLES PERTINENTS DU DATASET ---\n{lignes}\n--- FIN ---\n\n{note_niveau}\n"

    # --------------------------------------------------------
    # CHAT GÉNÉRAL
    # --------------------------------------------------------
    def chat(self, message: str, historique: list = None, niveau: str = "debutant") -> str:
        contexte = self._contexte_dataset(message, niveau)
        messages = [{"role": "system", "content": self.system_prompt + contexte}]
        if historique: messages.extend(historique[-10:])
        messages.append({"role": "user", "content": message})
        return self._appeler_mistral(messages)

    # --------------------------------------------------------
    # TRADUCTION OPTIMISÉE
    # --------------------------------------------------------
    def traduire(self, texte: str, direction: str = "fr_to_dyu", niveau: str = "debutant") -> str:
        """Traduction enrichie avec exemples dataset. Signature alignée avec api.py."""
        exemples = self.db.chercher(texte, k=6)
        ex_str = "\n".join([f"  • {e['francais']} ↔ {e['dioula']}" for e in exemples[:5]])

        if direction == "fr_to_dyu":
            prompt = (
                f"TRADUCTION FRANÇAIS → DIOULA\n"
                f"Texte : « {texte} »\n\n"
                f"Exemples similaires du dataset :\n{ex_str}\n\n"
                f"Réponds avec ce format EXACT :\n"
                f"**Traduction :** [traduction dioula ici]\n"
                f"**Prononciation :** [phonétique ex: 'ah-bi JI min NA']\n"
                f"**Exemple en contexte :** [phrase complète]\n"
                f"**Note :** [variante régionale ou conseil]"
            )
        else:
            prompt = (
                f"TRADUCTION DIOULA → FRANÇAIS\n"
                f"Texte Dioula : « {texte} »\n\n"
                f"Exemples similaires du dataset :\n{ex_str}\n\n"
                f"Réponds avec ce format EXACT :\n"
                f"**Traduction :** [traduction française ici]\n"
                f"**Sens littéral :** [mot à mot si utile]\n"
                f"**Contexte d'usage :** [quand on dit ça]\n"
                f"**Note culturelle :** [contexte ivoirien/burkinabè si pertinent]"
            )
        return self.chat(prompt, niveau=niveau)

    def traduire_simple(self, texte: str, direction: str = "fr_to_dyu") -> str:
        """
        Traduction rapide — retourne UNIQUEMENT le texte traduit, sans formatage.
        Utilisée par VoiceEngine pour générer l'audio de la traduction.
        """
        if direction == "fr_to_dyu":
            prompt = f"Traduis en Dioula UNIQUEMENT ce texte (réponds avec la traduction seule, sans explication ni ponctuation supplémentaire) : {texte}"
        else:
            prompt = f"Traduis en Français UNIQUEMENT ce texte (réponds avec la traduction seule, sans explication) : {texte}"

        messages = [
            {"role": "system", "content": "Tu es un traducteur Dioula/Français. Réponds avec la traduction uniquement, rien d'autre."},
            {"role": "user",   "content": prompt},
        ]
        try:
            resp = self.client.chat.completions.create(
                model=MISTRAL_MODEL, messages=messages,
                temperature=0.3, max_tokens=100,
                extra_headers={"HTTP-Referer": APP_URL, "X-Title": APP_NAME},
            )
            return resp.choices[0].message.content.strip().strip("«»\"'*")
        except Exception:
            return texte  # Retourner le texte original en cas d'erreur

    # --------------------------------------------------------
    # QUIZ — signature alignée avec api.py
    # --------------------------------------------------------
    def generer_quiz(self, theme: str = "vocabulaire", nb: int = 5,
                     niveau: str = "debutant", avec_reponses: bool = True) -> str:
        exemples = self.db.exemple_aleatoire(n=nb * 3)
        if niveau == "debutant":
            exemples = [e for e in exemples if len(e["dioula"].split()) <= 4]
        ex_str = "\n".join([f"  {i+1}. {e['dioula']} = {e['francais']}" for i, e in enumerate(exemples[:nb*2])])
        reponses_note = "\nInclus à la fin une section **RÉPONSES** avec la bonne lettre et une explication courte." \
                        if avec_reponses else "\nN'inclus PAS les réponses (mode entraînement)."
        prompt = (
            f"Génère {nb} questions de quiz sur le thème : {theme} (niveau {niveau})\n"
            f"Base-toi sur ces exemples RÉELS du dataset :\n{ex_str}\n\n"
            f"FORMAT OBLIGATOIRE pour chaque question :\n"
            f"**Q1/{nb}** : Que signifie « [phrase dioula] » ?\n"
            f"🅐 [bonne réponse]\n🅑 [mauvaise réponse]\n🅒 [mauvaise réponse]\n🅓 [mauvaise réponse]\n"
            f"{reponses_note}"
        )
        return self.chat(prompt, niveau=niveau)

    # --------------------------------------------------------
    # DIALOGUE — signature alignée avec api.py
    # --------------------------------------------------------
    def generer_dialogue(self, theme: str = "salutations", nb_tours: int = 4,
                         niveau: str = "debutant") -> str:
        exemples = self.db.chercher(theme, k=12)
        ex_str = "\n".join([f"  • {e['dioula']} ({e['francais']})" for e in exemples[:8]])
        prompt = (
            f"Génère un dialogue naturel en Dioula sur le thème : **{theme}** ({nb_tours} échanges, niveau {niveau})\n"
            f"Phrases du dataset à intégrer :\n{ex_str}\n\n"
            f"FORMAT EXACT :\n"
            f"**Amara :** [phrase dioula]\n*(= traduction française)*\n*[Prononciation : ...]*\n\n"
            f"**Fatoumata :** [réponse dioula]\n*(= traduction française)*\n*[Prononciation : ...]*\n\n"
            f"[continuer {nb_tours} échanges...]\n\n"
            f"---\n**Vocabulaire clé :** [5 mots importants]\n"
            f"**Note culturelle :** [contexte ivoirien/burkinabè]"
        )
        return self.chat(prompt, niveau=niveau)

    # --------------------------------------------------------
    # CORRECTION — signature alignée avec api.py
    # --------------------------------------------------------
    def corriger(self, phrase: str, niveau: str = "debutant", detaille: bool = True) -> str:
        exemples = self.db.chercher(phrase, k=6)
        ex_str = "\n".join([f"  • {e['dioula']} = {e['francais']}" for e in exemples[:5]])
        detail = "Sois TRÈS DÉTAILLÉ : explique la règle grammaticale, compare avec le dataset." \
                 if detaille else "Sois concis."
        prompt = (
            f"Un apprenant (niveau {niveau}) a écrit en Dioula : « {phrase} »\n\n"
            f"Phrases correctes similaires dans le dataset :\n{ex_str}\n\n"
            f"## Analyse\n"
            f"**Statut :** [✅ Correct / ❌ À corriger]\n"
            f"**Correction :** [version corrigée si nécessaire]\n\n"
            f"## Ce qui est bien ✅\n[Points positifs]\n\n"
            f"## Ce qu'on améliore 🔧\n[Erreurs précises]\n\n"
            f"## Règle à retenir 📚\n[Règle grammaticale simple]\n\n"
            f"## Encouragement 💪\n[Message motivant]\n\n"
            f"{detail}"
        )
        return self.chat(prompt, niveau=niveau)

    # --------------------------------------------------------
    # VOCABULAIRE — signature alignée avec api.py
    # --------------------------------------------------------
    def vocabulaire_du_jour(self, theme: str | None = None,
                             nb_mots: int = 5, niveau: str = "debutant") -> str:
        if theme:
            mots = self.db.chercher(theme, k=nb_mots * 2)
            if niveau == "debutant":
                mots = [m for m in mots if len(m["dioula"].split()) <= 3]
            mots = mots[:nb_mots]
        else:
            mots = self.db.vocabulaire_du_jour()[:nb_mots]
        if not mots:
            mots = self.db.exemple_aleatoire(nb_mots)

        liste = "\n".join([f"  {i+1}. **{m['dioula']}** = {m['francais']}" for i, m in enumerate(mots)])
        theme_str = f"sur le thème **{theme}**" if theme else "du jour"
        prompt = (
            f"Présente ces {nb_mots} mots Dioula {theme_str} (niveau {niveau}) :\n{liste}\n\n"
            f"Pour CHAQUE mot :\n"
            f"### [num]. [dioula] = [français]\n"
            f"🔊 **Prononciation :** [phonétique ex: 'AH-bi']\n"
            f"📝 **En contexte :** [phrase exemple = traduction]\n"
            f"🧠 **Astuce :** [truc mnémotechnique ou anecdote]\n\n"
            f"---\n**Pratique rapide :** [2 mini-questions pour mémoriser]"
        )
        return self.chat(prompt, niveau=niveau)

    # --------------------------------------------------------
    # LEÇON COMPLÈTE
    # --------------------------------------------------------
    def generer_lecon(self, theme: str, niveau: str = "debutant") -> str:
        exemples = self.db.chercher(theme, k=15)
        ex_str = "\n".join([f"  • {e['dioula']} = {e['francais']}" for e in exemples[:10]])
        prompt = (
            f"Crée une LEÇON COMPLÈTE de Dioula sur le thème : {theme}\n"
            f"Niveau : {niveau}\n"
            f"Exemples du dataset à utiliser :\n{ex_str}\n\n"
            f"STRUCTURE DE LA LEÇON :\n"
            f"## 🎯 Objectif\n[ce qu'on apprend]\n\n"
            f"## 📚 Vocabulaire essentiel\n[5-8 mots clés avec prononciation]\n\n"
            f"## 💬 Expressions utiles\n[4-6 phrases du dataset]\n\n"
            f"## 🗣️ Mini-dialogue\n[dialogue de 3-4 échanges]\n\n"
            f"## ✏️ Exercice\n[2-3 questions pratiques]\n\n"
            f"## 🌟 Point culturel\n[info culturelle sur la Côte d'Ivoire/Burkina/Mali]"
        )
        return self.chat(prompt, niveau=niveau)

    # --------------------------------------------------------
    # EXPLICATION GRAMMATICALE
    # --------------------------------------------------------
    def expliquer_grammaire(self, concept: str) -> str:
        prompt = (
            f"Explique ce concept grammatical du Dioula : {concept}\n\n"
            f"Structure :\n"
            f"1. Règle principale (simple et claire)\n"
            f"2. Exemples avec les phrases du dataset\n"
            f"3. Exceptions à connaître\n"
            f"4. Comparaison avec le français pour mieux comprendre\n"
            f"5. Exercice pratique"
        )
        return self.chat(prompt)