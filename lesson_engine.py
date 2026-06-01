"""
lesson_engine.py — Moteur de leçons adaptatif v1
Catégorise les phrases du dataset par contexte/niveau
et sélectionne les phrases selon la progression de l'apprenant.
"""

import random
import pandas as pd
from typing import Optional

CONTEXTES_MOTS_CLES = {
    "salutations":    ["sogoma","wula","tile","su","ni ce","i ni","aw ni","bonjour","bonsoir","bonne nuit","salut","au revoir","hello","good morning","kɛnɛya","santé"],
    "famille":        ["fa","ba","denmuso","denke","dɔgɔ","kɔrɔ","muso","cɛ","père","mère","fils","fille","frère","sœur","famille","enfant","mari","femme","parent","father","mother","brother","sister"],
    "nombres":        ["kelen","fila","saba","naani","duuru","wɔrɔ","wolonfila","segin","kɔnɔntɔn","tan","mugan","un","deux","trois","quatre","cinq","six","sept","huit","neuf","dix","one","two","three"],
    "temps":          ["tile","sini","kunu","bi","wula","sogoma","san","kalo","miniti","lɛrɛ","aujourd'hui","demain","hier","heure","minute","mois","année","matin","soir","nuit","today","tomorrow","yesterday"],
    "nourriture":     ["dumu","dumuni","ji","dege","tɔ","jaba","sogo","kɔngɔ","min","manger","boire","eau","nourriture","repas","faim","eat","drink","food","water","hungry"],
    "marche":         ["san","fɛn","wari","sɔrɔ","joli","kɛnɛ","dama","bagan","acheter","vendre","prix","argent","marché","boutique","buy","sell","price","money","market"],
    "ecole":          ["kalan","kalanso","kalanden","sɛbɛn","lɔn","école","étudiant","professeur","apprendre","lire","livre","school","student","teacher","learn","read","book"],
    "travail":        ["bara","baara","bolora","kɛ","travail","métier","profession","faire","travailler","work","job","do"],
    "voyage":         ["taga","na","bɔ","sira","mooto","partir","aller","venir","route","voyage","transport","go","come","travel","road"],
    "sante":          ["bana","dɔgɔtɔrɔ","kɛnɛya","dimi","malade","médecin","santé","douleur","médicament","sick","doctor","health","pain","medicine"],
    "maison":         ["bon","so","blon","kɔnɔ","jɔ","sigi","maison","chambre","porte","fenêtre","cuisine","house","room","door","home"],
    "culture":        ["dɔnkili","cogo","kuma","tɔgɔ","duniya","tradition","culture","langue","histoire","coutume"],
    "vie_quotidienne": [],
}

CONTEXTE_EMOJIS = {
    "salutations":"👋","famille":"👨‍👩‍👧","nombres":"🔢","temps":"⏰",
    "nourriture":"🍽️","marche":"🛒","ecole":"📚","travail":"💼",
    "voyage":"✈️","sante":"🏥","maison":"🏠","culture":"🎭","vie_quotidienne":"🌍"
}


class LessonEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._categoriser()
        print(f"✅ LessonEngine — {len(self.df)} phrases catégorisées")

    def _categoriser(self):
        self.df["contexte"]  = self.df.apply(self._detecter_contexte, axis=1)
        self.df["nb_mots"]   = self.df["dioula"].str.split().str.len()
        self.df["niveau"]    = self.df["nb_mots"].apply(self._detecter_niveau)
        self.df["phrase_id"] = self.df.index

    def _detecter_contexte(self, row) -> str:
        texte = f"{row['dioula']} {row['francais']} {row.get('anglais','')}".lower()
        scores = {}
        for ctx, mots in CONTEXTES_MOTS_CLES.items():
            if ctx == "vie_quotidienne": continue
            score = sum(1 for m in mots if m in texte)
            if score > 0: scores[ctx] = score
        return max(scores, key=scores.get) if scores else "vie_quotidienne"

    def _detecter_niveau(self, nb: int) -> str:
        if nb <= 5:  return "debutant"
        if nb <= 10: return "intermediaire"
        return "avance"

    def prochaine_phrase(self, niveau: str, contexte: Optional[str],
                          phrases_vues: list, phrases_ratees: list) -> Optional[dict]:
        # Priorité aux phrases ratées
        if phrases_ratees:
            c = self.df[self.df["phrase_id"].isin(phrases_ratees)]
            if not c.empty: return self._fmt(c.sample(1).iloc[0])
        pool = self._pool(niveau, contexte)
        if pool.empty: pool = self.df[self.df["niveau"] == niveau]
        if pool.empty: pool = self.df
        non_vues = pool[~pool["phrase_id"].isin(phrases_vues)]
        if len(non_vues) < 3: non_vues = pool
        return self._fmt(non_vues.sample(1).iloc[0])

    def phrase_aleatoire(self, niveau: Optional[str]=None, contexte: Optional[str]=None) -> Optional[dict]:
        pool = self._pool(niveau, contexte)
        if pool.empty: pool = self.df
        return self._fmt(pool.sample(1).iloc[0])

    def phrases_par_contexte(self, contexte: str, niveau: Optional[str]=None, nb: int=10) -> list[dict]:
        pool = self.df[self.df["contexte"] == contexte]
        if pool.empty: return []
        if niveau:
            pn = pool[pool["niveau"] == niveau]
            if not pn.empty: pool = pn
        return [self._fmt(r) for _, r in pool.sample(min(nb, len(pool))).iterrows()]

    def liste_contextes(self) -> list[dict]:
        stats = self.df.groupby("contexte").agg(
            nb_total=("phrase_id","count"),
            nb_debutant=("niveau", lambda x: (x=="debutant").sum()),
            nb_intermediaire=("niveau", lambda x: (x=="intermediaire").sum()),
            nb_avance=("niveau", lambda x: (x=="avance").sum()),
        ).reset_index()
        return [{"nom": r["contexte"], "nb_total": int(r["nb_total"]),
                 "nb_debutant": int(r["nb_debutant"]), "nb_intermediaire": int(r["nb_intermediaire"]),
                 "nb_avance": int(r["nb_avance"]), "emoji": CONTEXTE_EMOJIS.get(r["contexte"],"📝")}
                for _, r in stats.iterrows()]

    def _pool(self, niveau, contexte):
        pool = self.df.copy()
        if contexte and contexte in self.df["contexte"].values:
            pool = pool[pool["contexte"] == contexte]
        if niveau: pool = pool[pool["niveau"] == niveau]
        return pool

    def _fmt(self, row) -> dict:
        return {"id": int(row["phrase_id"]), "dioula": str(row["dioula"]),
                "francais": str(row["francais"]), "anglais": str(row.get("anglais","")),
                "contexte": str(row["contexte"]), "niveau": str(row["niveau"]),
                "nb_mots": int(row["nb_mots"]), "audio_url": f"/audio/{int(row['phrase_id'])}"}