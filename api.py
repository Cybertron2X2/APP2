"""
api.py — Serveur FastAPI Koumankan v5 (uniquement texte, sans aucune partie vocale)
Intègre : Mistral + Dataset + LessonEngine + ProgressManager
Supprime tous les endpoints audio : /speak, /translate/voice, /audio/...
"""

import os
import io, base64, json, re, unicodedata
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from dataset_loader    import DioulaDataset
from mistral_client    import MistralDioula
from lesson_engine     import LessonEngine
from progress_manager  import ProgressManager

load_dotenv()

# ============================================================
# MODÈLES PYDANTIC
# ============================================================

class RequeteChat(BaseModel):
    message:    str       = Field(..., min_length=1, max_length=2000)
    historique: list[dict]= Field(default=[], max_length=50)
    niveau:     str       = Field(default="debutant", pattern="^(debutant|intermediaire|avance)$")
    theme:      Optional[str] = Field(default=None, max_length=100)

class RequeteTraduction(BaseModel):
    texte:     str  = Field(..., min_length=1, max_length=500)
    direction: str  = Field(default="fr_to_dyu", pattern="^(fr_to_dyu|dyu_to_fr)$")
    niveau:    str  = Field(default="debutant",  pattern="^(debutant|intermediaire|avance)$")

class RequeteQuiz(BaseModel):
    theme:         str  = Field(default="vocabulaire de base", max_length=100)
    nb_questions:  int  = Field(default=5, ge=1, le=20)
    niveau:        str  = Field(default="debutant", pattern="^(debutant|intermediaire|avance)$")
    avec_reponses: bool = Field(default=True)

class RequeteDialogue(BaseModel):
    theme:    str = Field(default="salutations", max_length=100)
    nb_tours: int = Field(default=4, ge=2, le=10)
    niveau:   str = Field(default="debutant", pattern="^(debutant|intermediaire|avance)$")

class RequeteCorrection(BaseModel):
    phrase:   str  = Field(..., min_length=1, max_length=500)
    niveau:   str  = Field(default="debutant", pattern="^(debutant|intermediaire|avance)$")
    detaille: bool = Field(default=True)

class RequeteVocabulaire(BaseModel):
    theme:   Optional[str] = Field(default=None, max_length=100)
    nb_mots: int = Field(default=5, ge=3, le=20)
    niveau:  str = Field(default="debutant", pattern="^(debutant|intermediaire|avance)$")

class RequeteReponse(BaseModel):
    user_id:   str
    phrase_id: int
    reussi:    bool
    contexte:  str = ""
    niveau:    str = "debutant"

class ReponseTexte(BaseModel):
    reponse: str
    timestamp: str
    modele: str

# ============================================================
# ÉTAT GLOBAL
# ============================================================
ia:       MistralDioula  | None = None
db:       DioulaDataset  | None = None
lecons:   LessonEngine   | None = None
progress: ProgressManager| None = None

# ============================================================
# LIFESPAN
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ia, db, lecons, progress

    print("\n" + "="*60)
    print("🇨🇮 KOUMANKAN v5 — Texte uniquement (sans audio)")
    print("="*60)

    print("\n1️⃣  Dataset HuggingFace...")
    db = DioulaDataset()
    db.charger()

    print("\n2️⃣  Mistral (OpenRouter)...")
    try:
        ia = MistralDioula(dataset=db)
    except ValueError as e:
        print(f"⚠️  {e}")
        ia = None

    print("\n3️⃣  LessonEngine...")
    lecons = LessonEngine(db.df)

    print("\n4️⃣  ProgressManager...")
    progress = ProgressManager()

    port = os.getenv("PORT", 8000)
    print(f"\n✅ Prêt | {len(db.df)} phrases | Mistral:{'✅' if ia else '❌'}")
    print(f"   http://localhost:{port}  —  docs: http://localhost:{port}/docs")
    yield
    print("\n🛑 Arrêt Koumankan")

# ============================================================
# APP
# ============================================================
app = FastAPI(
    title="🇨🇮 Koumankan — API Dioula v5 (texte uniquement)",
    description="Professeur virtuel Dioula — sans aucune fonction vocale",
    version="5.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/", tags=["Interface"], include_in_schema=False)
async def interface():
    """Sert l'interface HTML directement depuis http://localhost:8000"""
    return FileResponse("index.html")

# ============================================================
# SANTÉ
# ============================================================
@app.get("/health", tags=["Système"])
async def health():
    return {
        "status": "ok",
        "mistral": ia is not None,
        "dataset": len(db.df) if db else 0,
        "lecons": len(lecons.df) if lecons else 0,
        "utilisateurs": len(progress._data) if progress else 0,
        "modele": os.getenv("MISTRAL_MODEL", "N/A"),
        "audio": False,
        "version": "5.0.0",
        "timestamp": datetime.now().isoformat()
    }

# ============================================================
# CHAT
# ============================================================
@app.post("/chat", response_model=ReponseTexte, tags=["IA"])
async def chat(req: RequeteChat):
    if not ia:
        raise HTTPException(503, "Mistral non disponible")
    msg = f"[Contexte : {req.theme}]\n{req.message}" if req.theme else req.message
    return ReponseTexte(
        reponse=ia.chat(msg, req.historique, req.niveau),
        timestamp=datetime.now().isoformat(),
        modele=os.getenv("MISTRAL_MODEL", "mistral")
    )

# ============================================================
# TRADUCTION ÉCRITE
# ============================================================
@app.post("/translate", tags=["Traduction"])
async def traduire(req: RequeteTraduction):
    if not ia:
        raise HTTPException(503, "Mistral non disponible")

    exemples = db.chercher(req.texte, k=6)
    reponse_ia = ia.traduire(req.texte, req.direction, req.niveau)
    traduction_propre = _extraire_traduction(reponse_ia, req.direction)

    result = {
        "reponse": reponse_ia,
        "traduction": traduction_propre,
        "exemples": exemples[:3],
        "direction": req.direction,
        "timestamp": datetime.now().isoformat(),
        "modele": os.getenv("MISTRAL_MODEL", "mistral"),
    }
    return JSONResponse(result)

# ============================================================
# QUIZ CLASSIQUE
# ============================================================
@app.post("/quiz", tags=["Pédagogie"])
async def quiz(req: RequeteQuiz):
    import random
    exemples_bruts = db.exemple_aleatoire(n=req.nb_questions * 3)
    if req.niveau == "debutant":
        exemples_bruts = [e for e in exemples_bruts if len(e["dioula"].split()) <= 4]
    exemples_bruts = exemples_bruts[:req.nb_questions]

    questions = []
    for i, bonne in enumerate(exemples_bruts):
        autres = [e for j, e in enumerate(exemples_bruts) if j != i][:3]
        options = [bonne["francais"]] + [m["francais"] for m in autres]
        random.shuffle(options)
        questions.append({
            "id": i + 1,
            "dioula": bonne["dioula"],
            "question": f"Que signifie « {bonne['dioula']} » ?",
            "bonne_reponse": bonne["francais"],
            "options": options,
            "explication": f"« {bonne['dioula']} » = « {bonne['francais']} »",
        })
    return {
        "theme": req.theme,
        "nb": len(questions),
        "questions": questions,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/quiz/ai", response_model=ReponseTexte, tags=["Pédagogie"])
async def quiz_ai(req: RequeteQuiz):
    if not ia:
        raise HTTPException(503, "Mistral non disponible")
    return ReponseTexte(
        reponse=ia.generer_quiz(req.theme, req.nb_questions, req.niveau, req.avec_reponses),
        timestamp=datetime.now().isoformat(),
        modele=os.getenv("MISTRAL_MODEL", "mistral")
    )

# ============================================================
# DIALOGUE
# ============================================================
@app.post("/dialogue", response_model=ReponseTexte, tags=["Pédagogie"])
async def dialogue(req: RequeteDialogue):
    if not ia:
        raise HTTPException(503, "Mistral non disponible")
    return ReponseTexte(
        reponse=ia.generer_dialogue(req.theme, req.nb_tours, req.niveau),
        timestamp=datetime.now().isoformat(),
        modele=os.getenv("MISTRAL_MODEL", "mistral")
    )

# ============================================================
# CORRECTION
# ============================================================
@app.post("/correct", response_model=ReponseTexte, tags=["Pédagogie"])
async def corriger(req: RequeteCorrection):
    if not ia:
        raise HTTPException(503, "Mistral non disponible")
    return ReponseTexte(
        reponse=ia.corriger(req.phrase, req.niveau, req.detaille),
        timestamp=datetime.now().isoformat(),
        modele=os.getenv("MISTRAL_MODEL", "mistral")
    )

# ============================================================
# VOCABULAIRE
# ============================================================
@app.post("/vocabulary", response_model=ReponseTexte, tags=["Pédagogie"])
async def vocabulaire(req: RequeteVocabulaire = RequeteVocabulaire()):
    if not ia:
        raise HTTPException(503, "Mistral non disponible")
    return ReponseTexte(
        reponse=ia.vocabulaire_du_jour(req.theme, req.nb_mots, req.niveau),
        timestamp=datetime.now().isoformat(),
        modele=os.getenv("MISTRAL_MODEL", "mistral")
    )

@app.get("/vocabulary/raw", tags=["Pédagogie"])
async def vocabulaire_raw(nb: int = 8, theme: Optional[str] = None):
    if not db:
        raise HTTPException(503, "Dataset non chargé")
    if theme and lecons:
        mots = lecons.phrases_par_contexte(theme, nb=nb)
    else:
        mots = db.vocabulaire_du_jour()
    return {"mots": mots, "date": datetime.now().strftime("%Y-%m-%d")}

# ============================================================
# LEÇONS
# ============================================================
@app.get("/lessons/contexts", tags=["Leçons"])
async def contextes_disponibles():
    if not lecons:
        raise HTTPException(503, "LessonEngine non disponible")
    return {"contextes": lecons.liste_contextes()}

@app.get("/lessons/next", tags=["Leçons"])
async def prochaine_phrase(
    user_id:  str = Query(...),
    contexte: Optional[str] = Query(default=None),
    niveau:   Optional[str] = Query(default=None),
):
    if not lecons or not progress:
        raise HTTPException(503, "Service non disponible")
    profil = progress.obtenir_ou_creer(user_id)
    phrase = lecons.prochaine_phrase(
        niveau=niveau or profil["niveau"],
        contexte=contexte,
        phrases_vues=profil["phrases_vues"],
        phrases_ratees=profil["phrases_ratees"],
    )
    return phrase

@app.get("/lessons/random", tags=["Leçons"])
async def phrase_aleatoire(contexte: Optional[str] = None, niveau: Optional[str] = None):
    if not lecons:
        raise HTTPException(503, "LessonEngine non disponible")
    return lecons.phrase_aleatoire(niveau, contexte)

@app.get("/lessons/context/{contexte}", tags=["Leçons"])
async def phrases_par_contexte(contexte: str, niveau: Optional[str] = None, nb: int = 10):
    if not lecons:
        raise HTTPException(503, "LessonEngine non disponible")
    return {
        "phrases": lecons.phrases_par_contexte(contexte, niveau, nb),
        "contexte": contexte
    }

# ============================================================
# PROGRESSION
# ============================================================
@app.post("/progress/user", tags=["Progression"])
async def creer_utilisateur(nom: str = "Apprenant", niveau: str = "debutant"):
    if not progress:
        raise HTTPException(503, "ProgressManager non disponible")
    return progress.creer_utilisateur(nom, niveau)

@app.post("/progress/answer", tags=["Progression"])
async def enregistrer_reponse(req: RequeteReponse):
    if not progress:
        raise HTTPException(503, "ProgressManager non disponible")
    return progress.enregistrer_reponse(req.user_id, req.phrase_id, req.reussi, req.contexte, req.niveau)

@app.get("/progress/{user_id}", tags=["Progression"])
async def statistiques_utilisateur(user_id: str):
    if not progress:
        raise HTTPException(503, "ProgressManager non disponible")
    return progress.statistiques(user_id)

@app.delete("/progress/{user_id}", tags=["Progression"])
async def reinitialiser_progression(user_id: str):
    if not progress:
        raise HTTPException(503, "ProgressManager non disponible")
    return progress.reinitialiser(user_id)

# ============================================================
# DATASET
# ============================================================
@app.get("/examples", tags=["Dataset"])
async def exemples(nb: int = Query(default=10, ge=1, le=50)):
    if not db:
        raise HTTPException(503, "Dataset non chargé")
    return {"exemples": db.exemple_aleatoire(nb), "total": nb}

@app.get("/search", tags=["Dataset"])
async def rechercher(q: str = Query(..., min_length=1), k: int = Query(default=5, ge=1, le=20)):
    if not db:
        raise HTTPException(503, "Dataset non chargé")
    return {"requete": q, "resultats": db.chercher(q, k), "nb": k}

@app.get("/stats", tags=["Dataset"])
async def statistiques():
    if not db:
        raise HTTPException(503, "Dataset non chargé")
    return db.statistiques()

# ============================================================
# UTILITAIRES
# ============================================================
def _extraire_traduction(reponse_ia: str, direction: str) -> str:
    m = re.search(r"\*\*Traduction\s*:\*\*\s*(.+?)(?:\n|$)", reponse_ia)
    if m:
        return m.group(1).strip().strip("*«»\"'")
    m = re.search(r"[Tt]raduction\s*:\s*(.+?)(?:\n|$)", reponse_ia)
    if m:
        return m.group(1).strip().strip("*«»\"'")
    lignes = [l.strip() for l in reponse_ia.split("\n") if l.strip()]
    return lignes[0].strip("*#«»") if lignes else reponse_ia[:100]

# ============================================================
# LANCEMENT
# ============================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"\n🚀 http://{host}:{port}  |  docs: http://localhost:{port}/docs")
    uvicorn.run("api:app", host=host, port=port, reload=False, log_level="info")
