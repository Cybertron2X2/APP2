"""
progress_manager.py — Gestion de la progression utilisateur v1
Stocke la progression dans un fichier JSON local.
"""

import os, json, uuid
from datetime import datetime
from typing import Optional

PROGRESS_FILE = "./progression_utilisateurs.json"


class ProgressManager:
    def __init__(self):
        self._data = self._charger()
        print(f"✅ ProgressManager — {len(self._data)} utilisateurs")

    def _charger(self):
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _sauvegarder(self):
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def creer_utilisateur(self, nom="Apprenant", niveau="debutant") -> dict:
        uid   = str(uuid.uuid4())[:8]
        profil = self._profil_vide(uid, nom, niveau)
        self._data[uid] = profil
        self._sauvegarder()
        return profil

    def obtenir_ou_creer(self, user_id: str) -> dict:
        if user_id not in self._data:
            self._data[user_id] = self._profil_vide(user_id)
            self._sauvegarder()
        return self._data[user_id]

    def enregistrer_reponse(self, user_id: str, phrase_id: int,
                             reussi: bool, contexte: str = "", niveau: str = "") -> dict:
        u   = self.obtenir_ou_creer(user_id)
        pid = str(phrase_id)

        if phrase_id not in u["phrases_vues"]:
            u["phrases_vues"].append(phrase_id)

        if pid not in u["phrases_details"]:
            u["phrases_details"][pid] = {"nb_vu":0,"nb_reussi":0,"nb_rate":0,"derniere_vue":None,"maitrisee":False}

        d = u["phrases_details"][pid]
        d["nb_vu"] += 1; d["derniere_vue"] = datetime.now().isoformat()

        if reussi:
            d["nb_reussi"] += 1
            if d["nb_reussi"] >= 3 and d["nb_rate"] == 0:
                d["maitrisee"] = True
                if phrase_id not in u["phrases_maitrisees"]: u["phrases_maitrisees"].append(phrase_id)
            if phrase_id in u["phrases_ratees"]: u["phrases_ratees"].remove(phrase_id)
            u["score"] += 10
        else:
            d["nb_rate"] += 1; d["maitrisee"] = False
            if phrase_id not in u["phrases_ratees"]: u["phrases_ratees"].append(phrase_id)
            if phrase_id in u["phrases_maitrisees"]: u["phrases_maitrisees"].remove(phrase_id)

        u["historique"].append({"phrase_id":phrase_id,"reussi":reussi,
                                 "contexte":contexte,"timestamp":datetime.now().isoformat()})
        u["historique"] = u["historique"][-100:]

        montee = self._verifier_montee(u)
        u["derniere_activite"] = datetime.now().isoformat()
        u["total_reponses"] += 1
        u["total_reussites"] += 1 if reussi else 0
        self._data[user_id] = u
        self._sauvegarder()

        return {"profil": u, "montee_niveau": montee,
                "phrases_maitrisees": len(u["phrases_maitrisees"]),
                "message": self._message(u, reussi, montee)}

    def statistiques(self, user_id: str) -> dict:
        u   = self.obtenir_ou_creer(user_id)
        tot = u["total_reponses"]
        return {
            "user_id":              user_id,
            "nom":                  u["nom"],
            "niveau":               u["niveau"],
            "score":                u["score"],
            "total_reponses":       tot,
            "taux_reussite":        round(u["total_reussites"]/tot*100 if tot else 0, 1),
            "phrases_vues":         len(u["phrases_vues"]),
            "phrases_maitrisees":   len(u["phrases_maitrisees"]),
            "phrases_a_retravailler": len(u["phrases_ratees"]),
            "progression_globale":  round(len(u["phrases_maitrisees"])/9536*100, 2),
            "serie_jours":          len({datetime.fromisoformat(h["timestamp"]).date().isoformat()
                                         for h in u["historique"] if "timestamp" in h}),
            "derniere_activite":    u["derniere_activite"],
        }

    def _verifier_montee(self, u) -> bool:
        seuils = {"debutant": 50, "intermediaire": 150}
        niveau = u["niveau"]
        if niveau in seuils and len(u["phrases_maitrisees"]) >= seuils[niveau]:
            niveaux = ["debutant","intermediaire","avance"]
            idx = niveaux.index(niveau)
            if idx < 2: u["niveau"] = niveaux[idx+1]; u["score"] += 100; return True
        return False

    def _message(self, u, reussi, montee) -> str:
        if montee: return f"🎉 Félicitations ! Niveau {u['niveau']} atteint !"
        return f"✅ Bravo ! {len(u['phrases_maitrisees'])} phrases maîtrisées." if reussi \
               else "❌ Pas encore… Cette phrase reviendra bientôt !"

    def _profil_vide(self, uid, nom="Apprenant", niveau="debutant") -> dict:
        now = datetime.now().isoformat()
        return {"user_id":uid,"nom":nom,"niveau":niveau,"score":0,
                "total_reponses":0,"total_reussites":0,"phrases_vues":[],
                "phrases_maitrisees":[],"phrases_ratees":[],"phrases_details":{},
                "historique":[],"cree_le":now,"derniere_activite":now}

    def reinitialiser(self, user_id: str) -> dict:
        if user_id in self._data:
            nom = self._data[user_id]["nom"]
            self._data[user_id] = self._profil_vide(user_id, nom)
            self._sauvegarder()
        return self._data.get(user_id, {})