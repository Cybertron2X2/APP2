"""
============================================================
dataset_loader.py — Chargement du dataset avec Pandas
============================================================
Charge uvci/koumankan4dyula depuis HuggingFace,
le transforme en DataFrame Pandas, et construit
la base de connaissance que Mistral utilisera.

Utilisation :
    from dataset_loader import DioulaDataset
    db = DioulaDataset()
    db.charger()
    contexte = db.chercher("bonjour comment ça va", k=5)
"""
import re
import os
import pickle
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


class DioulaDataset:
    """
    Charge et indexe le dataset Dioula pour la recherche contextuelle.
    Utilise Pandas pour la manipulation des données.
    """

    def __init__(self, cache_path: str = "./cache_dataset.pkl"):
        self.cache_path = cache_path
        self.df: pd.DataFrame | None = None      # DataFrame principal
        self._vecteurs: np.ndarray | None = None # Vecteurs TF-IDF simples

    # --------------------------------------------------------
    # CHARGEMENT
    # --------------------------------------------------------

    def charger(self, forcer_reload: bool = False) -> pd.DataFrame:
        """
        Charge le dataset depuis HuggingFace et le convertit en DataFrame.
        Utilise un cache local pour éviter de re-télécharger.
        """
        # Vérifier le cache
        if not forcer_reload and os.path.exists(self.cache_path):
            print(f"📂 Cache trouvé → chargement depuis {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                self.df = pickle.load(f)
            print(f"✅ {len(self.df)} exemples chargés depuis le cache")
            self._construire_index_simple()
            return self.df

        # Télécharger depuis HuggingFace
        print("📦 Téléchargement du dataset uvci/koumankan4dyula...")
        dataset_hf = load_dataset("uvci/koumankan4dyula")

        # ---- Convertir chaque split en DataFrame Pandas ----
        frames = []
        for split_name in ["train", "dev", "test"]:
            split = dataset_hf[split_name]
            df_split = pd.DataFrame({
                "dioula":   split["dyu"],
                "francais": split["fr"],
                "anglais":  split["en"],
                "genre":    split["gender"],
                "age":      split["age_group"],
                "pays":     split["country"],
                "split":    split_name,
            })
            frames.append(df_split)
            print(f"   {split_name:5s} : {len(df_split):5d} lignes")

        # ---- Concaténer tous les splits ----
        self.df = pd.concat(frames, ignore_index=True)

        # ---- Nettoyage Pandas ----
        # Supprimer les lignes avec valeurs manquantes
        avant = len(self.df)
        self.df.dropna(subset=["dioula", "francais"], inplace=True)

        # Supprimer les phrases vides
        self.df = self.df[
            (self.df["dioula"].str.strip() != "") &
            (self.df["francais"].str.strip() != "")
        ]
        apres = len(self.df)
        print(f"\n🧹 Nettoyage : {avant - apres} lignes supprimées")

        # Réinitialiser l'index
        self.df.reset_index(drop=True, inplace=True)

        # ---- Statistiques avec Pandas ----
        print(f"\n📊 Statistiques du dataset :")
        print(f"   Total exemples    : {len(self.df)}")
        print(f"   Pays              : {self.df['pays'].value_counts().to_dict()}")
        print(f"   Genre             : {self.df['genre'].value_counts().to_dict()}")
        print(f"   Longueur moy. Dioula   : {self.df['dioula'].str.split().str.len().mean():.1f} mots")
        print(f"   Longueur moy. Français : {self.df['francais'].str.split().str.len().mean():.1f} mots")

        # ---- Sauvegarder le cache ----
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.df, f)
        print(f"\n💾 Cache sauvegardé → {self.cache_path}")

        # Construire l'index de recherche
        self._construire_index_simple()

        return self.df

    # --------------------------------------------------------
    # INDEX DE RECHERCHE SIMPLE (sans dépendances lourdes)
    # --------------------------------------------------------

    def _construire_index_simple(self):
        """
        Construit un index de recherche par mots-clés simple avec Pandas.
        Pas besoin de FAISS ou sentence-transformers.
        """
        print("🔧 Construction de l'index de recherche...")
        # Colonne de recherche combinée (minuscules)
        self.df["_search"] = (
            self.df["dioula"].str.lower() + " " +
            self.df["francais"].str.lower() + " " +
            self.df["anglais"].str.lower()
        )
        print("✅ Index prêt")

    # --------------------------------------------------------
    # RECHERCHE
    # --------------------------------------------------------

    def chercher(self, requete: str, k: int = 8) -> list[dict]:
        """
        Recherche les exemples les plus pertinents pour une requête.
        Utilise la recherche par mots-clés Pandas (rapide, sans GPU).

        Args:
            requete : texte à chercher (français ou dioula)
            k       : nombre d'exemples à retourner

        Returns:
            Liste de dicts {dioula, francais, anglais, score}
        """
        if self.df is None:
            raise RuntimeError("Dataset non chargé. Appelez charger() d'abord.")

        requete_lower = requete.lower().strip()
        mots = requete_lower.split()

        if not mots:
            # Retourner des exemples aléatoires
            echantillon = self.df.sample(k)
            return echantillon[["dioula", "francais", "anglais"]].to_dict("records")

        # ---- Score par correspondance de mots ----
        scores = pd.Series(0.0, index=self.df.index)

        for mot in mots:
            if len(mot) < 2:
                continue
            # Correspondance exacte du mot
            mask_exact = self.df["_search"].str.contains(
                r'\b' + re.escape(mot) + r'\b', regex=True, na=False
            )
            # Correspondance partielle (préfixe)
            mask_partiel = self.df["_search"].str.contains(mot, regex=False, na=False)

            scores[mask_exact]   += 2.0   # Exact : score double
            scores[mask_partiel] += 1.0   # Partiel : score simple

        # Boost : si la requête complète est dans le texte
        mask_phrase = self.df["_search"].str.contains(
            requete_lower[:30], regex=False, na=False
        )
        scores[mask_phrase] += 5.0

        # Trier et retourner les k meilleurs
        top_indices = scores.nlargest(k).index
        resultats = self.df.loc[top_indices, ["dioula", "francais", "anglais"]].copy()
        resultats["score"] = scores[top_indices].values

        # Filtrer les résultats avec score > 0
        resultats = resultats[resultats["score"] > 0]

        if len(resultats) < k:
            # Compléter avec des exemples aléatoires si pas assez de résultats
            manquants = k - len(resultats)
            extra = self.df.sample(manquants)[["dioula", "francais", "anglais"]].copy()
            extra["score"] = 0.0
            resultats = pd.concat([resultats, extra])

        return resultats.head(k).to_dict("records")

    def exemple_aleatoire(self, n: int = 5) -> list[dict]:
        """Retourne n exemples aléatoires du dataset."""
        return self.df.sample(n)[["dioula", "francais", "anglais"]].to_dict("records")

    def vocabulaire_du_jour(self) -> list[dict]:
        """Sélectionne 5 mots/phrases courts pour la pratique quotidienne."""
        # Filtrer les phrases courtes (max 4 mots)
        courts = self.df[self.df["dioula"].str.split().str.len() <= 4]
        return courts.sample(min(5, len(courts)))[["dioula", "francais"]].to_dict("records")

    def statistiques(self) -> dict:
        """Retourne des statistiques complètes sur le dataset."""
        if self.df is None:
            return {}
        return {
            "total": len(self.df),
            "par_split": self.df["split"].value_counts().to_dict(),
            "par_pays":  self.df["pays"].value_counts().to_dict(),
            "par_genre": self.df["genre"].value_counts().to_dict(),
            "longueur_moy_dioula":   round(self.df["dioula"].str.split().str.len().mean(), 1),
            "longueur_moy_francais": round(self.df["francais"].str.split().str.len().mean(), 1),
        }

    def construire_prompt_systeme(self) -> str:
        """
        Construit le prompt système pour Mistral en incluant
        des exemples représentatifs du dataset.
        """
        # Sélectionner 20 exemples variés et courts
        exemples = self.df[
            self.df["dioula"].str.split().str.len().between(2, 6)
        ].sample(min(20, len(self.df)))[["dioula", "francais"]]

        exemples_str = "\n".join([
            f"  • {row['dioula']} = {row['francais']}"
            for _, row in exemples.iterrows()
        ])

        return f"""Tu es Koumankan, un professeur expert en langue Dioula (Dyula).
Le Dioula est une langue mandingue parlée principalement en Côte d'Ivoire, au Burkina Faso et au Mali.
Tu aides les apprenants à maîtriser le Dioula avec patience et pédagogie.

EXEMPLES DE PHRASES DIOULA (extraits du dataset d'entraînement) :
{exemples_str}

TES RÈGLES :
1. Toujours donner la traduction Dioula quand on te demande
2. Expliquer simplement, de façon encourageante
3. Donner des exemples concrets tirés du dataset
4. Adapter ton niveau à l'apprenant (débutant par défaut)
5. Pour les quiz, proposer 4 choix A/B/C/D avec la bonne réponse
6. Pour les dialogues, écrire en Dioula avec la traduction française entre parenthèses

Réponds toujours en français sauf pour les mots/phrases en Dioula."""


# ============================================================
# TEST DIRECT
# ============================================================
if __name__ == "__main__":
    db = DioulaDataset()
    df = db.charger()

    print("\n" + "="*50)
    print("🔍 TEST DE RECHERCHE")
    print("="*50)

    tests = ["bonjour", "eau", "maison", "A bi ji min na"]
    for t in tests:
        resultats = db.chercher(t, k=3)
        print(f"\n→ '{t}' :")
        for r in resultats:
            print(f"   🇨🇮 {r['dioula']} = 🇫🇷 {r['francais']}")

    print("\n📅 Vocabulaire du jour :")
    for v in db.vocabulaire_du_jour():
        print(f"   {v['dioula']} = {v['francais']}")