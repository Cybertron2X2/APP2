"""
voice_engine.py — Moteur vocal FINAL CORRIGÉ
Résout les 3 bugs audio :
  1. Index _dyu_lower corrigé pour la recherche par phrase_id
  2. Recherche fuzzy (normalisation + partielle) pour matcher les phrases
  3. Fallback gTTS si audio natif introuvable (plus jamais de 404)
"""

import io, os, re, hashlib, wave, unicodedata
import pandas as pd
from gtts import gTTS

CACHE_DIR = "./audio_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

HF_CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--uvci--koumankan4dyula"
)

def _normaliser(s: str) -> str:
    """Minuscules + sans accents + sans ponctuation — pour comparaison robuste."""
    s = s.lower().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class VoiceEngine:

    def __init__(self, utiliser_cache=True, vitesse_lente=False):
        self.utiliser_cache = utiliser_cache
        self.vitesse_lente  = vitesse_lente
        self._df_audio      = None   # DataFrame des audios natifs
        self._index_norm    = {}     # dict normalise -> row index pour recherche rapide
        self._charger_audios()
        print("✅ VoiceEngine FINAL initialisé")

    # ─────────────────────────────────────────────────────────
    # CHARGEMENT
    # ─────────────────────────────────────────────────────────

    def _charger_audios(self):
        """Charge tous les fichiers .parquet du cache HuggingFace."""
        try:
            parquet_files = []
            if os.path.exists(HF_CACHE):
                for root, _, files in os.walk(HF_CACHE):
                    for f in files:
                        if f.endswith(".parquet"):
                            parquet_files.append(os.path.join(root, f))

            if not parquet_files:
                print(f"   ⚠️  Aucun parquet dans {HF_CACHE} — mode gTTS seul")
                return

            rows = []
            total_avec_audio = 0

            for pq in parquet_files:
                try:
                    # Lire sans sampling_rate (colonne absente dans certains splits)
                    df = pd.read_parquet(pq, columns=["dyu", "fr", "audio"])
                    df_ok = df[df["audio"].notna()]
                    total_avec_audio += len(df_ok)

                    for _, row in df_ok.iterrows():
                        dyu = row["dyu"]
                        fr  = row["fr"]
                        if not isinstance(dyu, str) or not isinstance(fr, str):
                            continue

                        audio_info = row["audio"]
                        if not isinstance(audio_info, dict):
                            continue

                        raw = audio_info.get("bytes")
                        if not isinstance(raw, bytes) or len(raw) < 1000:
                            # Certains fichiers ont des bytes trop courts — ignorer
                            continue

                        # Sampling rate : essayer dans le dict audio, sinon 16000
                        sr = audio_info.get("sampling_rate") or 16000
                        try:
                            sr = int(sr)
                        except Exception:
                            sr = 16000
                        if sr <= 0:
                            sr = 16000

                        wav = self._pcm_to_wav(raw, sr)
                        rows.append({
                            "dyu":         dyu,
                            "fr":          fr,
                            "audio_bytes": wav,
                            "dyu_norm":    _normaliser(dyu),
                            "fr_norm":     _normaliser(fr),
                        })

                except Exception as e:
                    print(f"   ⚠️  {os.path.basename(pq)} : {e}")
                    continue

            if not rows:
                print(f"   ⚠️  Aucun audio valide trouvé sur {total_avec_audio} lignes — mode gTTS")
                return

            self._df_audio = pd.DataFrame(rows)
            # Index rapide : normalised_dyu -> position dans le DataFrame
            self._index_norm = {
                row["dyu_norm"]: i
                for i, row in self._df_audio.iterrows()
            }
            print(f"   🎵 {len(self._df_audio)} audios natifs chargés"
                  f" (sur {total_avec_audio} lignes audio dans les parquets)")

        except Exception as e:
            print(f"   ❌ Erreur chargement audio : {e} — mode gTTS seul")
            self._df_audio = None

    # ─────────────────────────────────────────────────────────
    # RECHERCHE — ROBUSTE
    # ─────────────────────────────────────────────────────────

    def _chercher_ligne(self, texte: str, colonne: str = "dyu") -> pd.Series | None:
        """
        Cherche une ligne par texte dans la colonne indiquée.
        3 passes : exacte normalisée → préfixe → contient.
        Retourne la Series pandas de la ligne trouvée, ou None.
        """
        if self._df_audio is None or self._df_audio.empty:
            return None

        norm_col = "dyu_norm" if colonne == "dyu" else "fr_norm"
        t = _normaliser(texte)

        # Passe 1 : correspondance exacte normalisée (O(1) via dict)
        if colonne == "dyu" and t in self._index_norm:
            return self._df_audio.loc[self._index_norm[t]]

        # Passe 1b : exact sur DataFrame pour la colonne fr
        mask = self._df_audio[norm_col] == t
        if mask.any():
            return self._df_audio[mask].iloc[0]

        # Passe 2 : commence par les 25 premiers chars
        prefix = t[:25]
        mask = self._df_audio[norm_col].str.startswith(prefix, na=False)
        if mask.any():
            return self._df_audio[mask].iloc[0]

        # Passe 3 : contient les 15 premiers chars
        prefix = t[:15]
        if len(prefix) >= 4:
            mask = self._df_audio[norm_col].str.contains(
                re.escape(prefix), regex=True, na=False
            )
            if mask.any():
                return self._df_audio[mask].iloc[0]

        return None

    def _audio_pour_texte(self, texte: str, colonne: str = "dyu") -> bytes | None:
        """Retourne les bytes WAV pour un texte, ou None si non trouvé."""
        ligne = self._chercher_ligne(texte, colonne)
        if ligne is not None and "audio_bytes" in ligne.index:
            return ligne["audio_bytes"]
        return None

    # ─────────────────────────────────────────────────────────
    # API PUBLIQUE
    # ─────────────────────────────────────────────────────────

    def synthetiser(self, texte: str, langue: str = "dioula") -> tuple[bytes, str]:
        """
        Synthétise du texte en audio.
        1. Cherche dans le dataset (audio natif WAV)
        2. Fallback gTTS si non trouvé
        Retourne (bytes, mime_type).
        """
        if not texte or not texte.strip():
            raise ValueError("Texte vide")

        texte_propre = self._nettoyer(texte)
        langue_lc    = langue.lower()
        cle          = hashlib.md5(f"{langue_lc}:{texte_propre}".encode()).hexdigest()

        # Cache WAV
        chemin_wav = os.path.join(CACHE_DIR, f"{cle}.wav")
        if self.utiliser_cache and os.path.exists(chemin_wav):
            with open(chemin_wav, "rb") as f:
                return f.read(), "audio/wav"

        # Cache MP3
        chemin_mp3 = os.path.join(CACHE_DIR, f"{cle}.mp3")
        if self.utiliser_cache and os.path.exists(chemin_mp3):
            with open(chemin_mp3, "rb") as f:
                return f.read(), "audio/mpeg"

        # Audio natif (Dioula uniquement)
        if langue_lc in ("dioula", "dyu") and self._df_audio is not None:
            wav = self._audio_pour_texte(texte_propre, "dyu")
            if wav:
                if self.utiliser_cache:
                    with open(chemin_wav, "wb") as f:
                        f.write(wav)
                return wav, "audio/wav"

        # Fallback gTTS (toujours disponible)
        lang_gtts = "fr" if langue_lc in ("dioula", "dyu") else langue_lc
        try:
            mp3 = self._gtts(texte_propre, lang_gtts)
        except Exception:
            mp3 = self._gtts(texte_propre[:100], "fr")   # dernier recours
        if self.utiliser_cache:
            with open(chemin_mp3, "wb") as f:
                f.write(mp3)
        return mp3, "audio/mpeg"

    def obtenir_audio_par_id(
        self, phrase_id: int, df_principal: pd.DataFrame
    ) -> tuple[bytes, str] | None:
        """
        Retourne l'audio d'une phrase identifiée par son index dans df_principal.
        Cherche le texte Dioula dans le dataset audio.
        Fallback gTTS si introuvable.
        """
        # Cache disque
        chemin = os.path.join(CACHE_DIR, f"id_{phrase_id}.wav")
        if self.utiliser_cache and os.path.exists(chemin):
            with open(chemin, "rb") as f:
                return f.read(), "audio/wav"

        # Récupérer le texte Dioula depuis le DataFrame principal
        if phrase_id not in df_principal.index:
            return None

        dioula = str(df_principal.loc[phrase_id, "dioula"])

        # Chercher l'audio natif
        if self._df_audio is not None:
            wav = self._audio_pour_texte(dioula, "dyu")
            if wav:
                if self.utiliser_cache:
                    with open(chemin, "wb") as f:
                        f.write(wav)
                return wav, "audio/wav"

        # Fallback gTTS
        try:
            mp3 = self._gtts(self._nettoyer(dioula), "fr")
            chemin_mp3 = os.path.join(CACHE_DIR, f"id_{phrase_id}.mp3")
            if self.utiliser_cache:
                with open(chemin_mp3, "wb") as f:
                    f.write(mp3)
            return mp3, "audio/mpeg"
        except Exception:
            return None

    def synthetiser_avec_traduction(
        self, texte: str, direction: str = "fr_to_dyu", lent: bool = True
    ) -> dict:
        """
        Cherche la paire dans le dataset et retourne source + traduction audio.
        Si non trouvé → gTTS pour les deux.
        """
        texte_propre = self._nettoyer(texte)
        col_src  = "fr"  if direction == "fr_to_dyu" else "dyu"
        col_trad = "dyu" if direction == "fr_to_dyu" else "fr"

        ligne = self._chercher_ligne(texte_propre, col_src)

        if ligne is not None:
            src_txt  = str(ligne.get(col_src,  texte_propre))
            trad_txt = str(ligne.get(col_trad, ""))
            # Audio source
            if col_src == "dyu":
                src_audio = ligne.get("audio_bytes")
                src_mime  = "audio/wav" if src_audio else "audio/mpeg"
                if not src_audio:
                    src_audio, src_mime = self.synthetiser(src_txt, "dioula")
            else:
                src_audio, src_mime = self.synthetiser(src_txt, "fr")
            # Audio traduction
            if col_trad == "dyu":
                trad_audio = ligne.get("audio_bytes")
                trad_mime  = "audio/wav" if trad_audio else "audio/mpeg"
                if not trad_audio:
                    trad_audio, trad_mime = self.synthetiser(trad_txt, "dioula")
            else:
                trad_audio, trad_mime = self.synthetiser(trad_txt, "fr")

            return {
                "source_texte":        src_txt,
                "source_audio":        src_audio,
                "source_mime":         src_mime,
                "traduction_texte":    trad_txt,
                "traduction_audio":    trad_audio,
                "traduction_mime":     trad_mime,
                "trouve_dans_dataset": True,
                "direction":           direction,
            }
        else:
            # Non trouvé → gTTS pour source, l'IA fournira la traduction
            src_audio, src_mime = self.synthetiser(texte_propre,
                                                    "fr" if col_src == "fr" else "dioula")
            return {
                "source_texte":        texte_propre,
                "source_audio":        src_audio,
                "source_mime":         src_mime,
                "traduction_texte":    None,
                "traduction_audio":    None,
                "traduction_mime":     None,
                "trouve_dans_dataset": False,
                "direction":           direction,
            }

    def lister_audios_disponibles(self, query: str = "", limit: int = 20) -> list[dict]:
        if self._df_audio is None:
            return []
        df = self._df_audio[["dyu", "fr"]].copy()
        if query.strip():
            q = _normaliser(query)
            mask = (
                df["dyu"].str.lower().str.contains(q[:20], na=False) |
                df["fr"].str.lower().str.contains(q[:20], na=False)
            )
            df = df[mask]
        return df.head(limit).rename(
            columns={"dyu": "dioula", "fr": "francais"}
        ).to_dict("records")

    # ─────────────────────────────────────────────────────────
    # UTILITAIRES AUDIO
    # ─────────────────────────────────────────────────────────

    def _pcm_to_wav(self, pcm: bytes, sr: int = 16000) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm)
        buf.seek(0)
        return buf.read()

    def _gtts(self, texte: str, langue: str = "fr") -> bytes:
        tts = gTTS(text=texte, lang=langue, slow=self.vitesse_lente, lang_check=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()

    def _nettoyer(self, texte: str) -> str:
        texte = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", texte)
        texte = re.sub(r"#{1,6}\s", "", texte)
        texte = re.sub(r"`[^`]+`", "", texte)
        texte = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", texte)
        texte = re.sub(
            r"[\U0001F300-\U0001FFFF\U00002600-\U000026FF\U00002700-\U000027BF]",
            "", texte
        )
        texte = re.sub(r"\s+", " ", texte).strip()
        if len(texte) > 500:
            p = max(texte[:500].rfind("."),
                    texte[:500].rfind("!"),
                    texte[:500].rfind("?"))
            texte = texte[:p + 1] if p > 200 else texte[:500]
        return texte

    # Compatibilité ancienne API
    def synthetiser_phrase_dioula(self, texte: str, lent: bool = True):
        prev = self.vitesse_lente
        self.vitesse_lente = lent
        r = self.synthetiser(texte, "dioula")
        self.vitesse_lente = prev
        return r

    def vider_cache(self):
        n = 0
        for f in os.listdir(CACHE_DIR):
            if f.endswith((".mp3", ".wav")):
                os.remove(os.path.join(CACHE_DIR, f))
                n += 1
        print(f"🗑️  {n} fichiers supprimés")

    def taille_cache(self):
        fls = [f for f in os.listdir(CACHE_DIR) if f.endswith((".mp3", ".wav"))]
        total = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in fls)
        return {"nb_fichiers": len(fls), "taille_mb": round(total / 1e6, 2)}