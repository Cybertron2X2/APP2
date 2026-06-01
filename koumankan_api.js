/**
 * koumankan_api.js — Client JavaScript v4 (sans gTTS, avec audio natif)
 * Compatible avec api.py v4
 */

const API_BASE_URL = "http://localhost:8000";

const KoumankanAPI = {

  // ------------------------------------------------------------
  // UTILITAIRES
  // ------------------------------------------------------------
  async _post(endpoint, body) {
    const res = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Erreur ${res.status}`);
    }
    return res.json();
  },

  async _get(endpoint, params = {}) {
    const url = new URL(`${API_BASE_URL}${endpoint}`);
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Erreur ${res.status}`);
    return res.json();
  },

  // ------------------------------------------------------------
  // SANTÉ
  // ------------------------------------------------------------
  async ping() {
    try {
      const d = await this._get("/health");
      return d.status === "ok";
    } catch {
      return false;
    }
  },

  async healthDetails() {
    return this._get("/health");
  },

  // ------------------------------------------------------------
  // CHAT
  // ------------------------------------------------------------
  async chat(message, historique = [], niveau = "debutant", theme = null) {
    const d = await this._post("/chat", { message, historique, niveau, theme });
    return d.reponse;
  },

  // ------------------------------------------------------------
  // TRADUCTION ÉCRITE
  // ------------------------------------------------------------
  async traduire(texte, direction = "fr_to_dyu", niveau = "debutant", avecAudio = false) {
    return this._post("/translate", { texte, direction, niveau, avec_audio: avecAudio });
  },

  // ------------------------------------------------------------
  // TRADUCTION ORALE (microphone → texte → traduction)
  // ------------------------------------------------------------
  async traduireVoix(audioBlob, direction = "fr_to_dyu", niveau = "debutant") {
    const fd = new FormData();
    fd.append("audio", audioBlob, "recording.webm");
    fd.append("direction", direction);
    fd.append("niveau", niveau);
    const res = await fetch(`${API_BASE_URL}/translate/voice`, { method: "POST", body: fd });
    if (!res.ok) {
      const e = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(e.detail);
    }
    return res.json();
  },

  // ------------------------------------------------------------
  // SYNTHÈSE VOCALE (audio natif uniquement)
  // ------------------------------------------------------------
  async parler(texte, lent = true, langue = "dioula") {
    const res = await fetch(`${API_BASE_URL}/speak`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texte, direction: langue, lent }),
    });
    if (!res.ok) throw new Error(`Erreur audio ${res.status}`);
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("audio")) {
      const blob = await res.blob();
      return this._jouerBlob(blob);
    } else {
      const data = await res.json();
      if (data.traduction_audio_b64) {
        return this._jouerBase64(data.traduction_audio_b64, data.traduction_mime || "audio/wav");
      } else if (data.source_audio_b64) {
        return this._jouerBase64(data.source_audio_b64, data.source_mime || "audio/wav");
      }
    }
  },

  async parlerAvecTraduction(texte, direction = "fr_to_dyu", lent = true) {
    return this._post("/speak", { texte, direction, lent });
  },

  async jouerAudioParId(phraseId) {
    const res = await fetch(`${API_BASE_URL}/audio/${phraseId}`);
    if (!res.ok) throw new Error("Audio non disponible");
    const blob = await res.blob();
    return this._jouerBlob(blob);
  },

  audioUrl(texte, langue = "dioula", lent = true) {
    return `${API_BASE_URL}/speak/${encodeURIComponent(texte)}?langue=${langue}&lent=${lent}`;
  },

  _jouerBlob(blob) {
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.onended = () => URL.revokeObjectURL(url);
    return audio.play().then(() => audio);
  },

  _jouerBase64(b64, mime = "audio/wav") {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length).map((_, i) => bin.charCodeAt(i));
    const blob = new Blob([bytes], { type: mime });
    return this._jouerBlob(blob);
  },

  // ------------------------------------------------------------
  // QUIZ CLASSIQUE
  // ------------------------------------------------------------
  async quiz(theme = "vocabulaire de base", nbQuestions = 5, niveau = "debutant") {
    return this._post("/quiz", { theme, nb_questions: nbQuestions, niveau });
  },

  async quizIA(theme = "vocabulaire de base", nbQuestions = 5, niveau = "debutant") {
    const d = await this._post("/quiz/ai", { theme, nb_questions: nbQuestions, niveau });
    return d.reponse;
  },

  // ------------------------------------------------------------
  // DIALOGUE
  // ------------------------------------------------------------
  async dialogue(theme = "salutations", nbTours = 4, niveau = "debutant") {
    const d = await this._post("/dialogue", { theme, nb_tours: nbTours, niveau });
    return d.reponse;
  },

  // ------------------------------------------------------------
  // CORRECTION
  // ------------------------------------------------------------
  async corriger(phrase, niveau = "debutant") {
    const d = await this._post("/correct", { phrase, niveau, detaille: true });
    return d.reponse;
  },

  // ------------------------------------------------------------
  // VOCABULAIRE
  // ------------------------------------------------------------
  async vocabDuJour(theme = null, nbMots = 5, niveau = "debutant") {
    const d = await this._post("/vocabulary", { theme, nb_mots: nbMots, niveau });
    return d.reponse;
  },

  async vocabRaw(nb = 8, theme = null) {
    const params = { nb };
    if (theme) params.theme = theme;
    return this._get("/vocabulary/raw", params);
  },

  // ------------------------------------------------------------
  // LEÇONS ADAPTATIVES
  // ------------------------------------------------------------
  async contextes() {
    return this._get("/lessons/contexts");
  },

  async prochainePhraseLecon(userId, contexte = null, niveau = null) {
    const params = { user_id: userId };
    if (contexte) params.contexte = contexte;
    if (niveau) params.niveau = niveau;
    return this._get("/lessons/next", params);
  },

  async phraseAleatoire(contexte = null, niveau = null) {
    const p = {};
    if (contexte) p.contexte = contexte;
    if (niveau) p.niveau = niveau;
    return this._get("/lessons/random", p);
  },

  async phrasesContexte(contexte, niveau = null, nb = 10) {
    const p = { nb };
    if (niveau) p.niveau = niveau;
    return this._get(`/lessons/context/${contexte}`, p);
  },

  // ------------------------------------------------------------
  // PROGRESSION
  // ------------------------------------------------------------
  async creerUtilisateur(nom = "Apprenant", niveau = "debutant") {
    const url = new URL(`${API_BASE_URL}/progress/user`);
    url.searchParams.set("nom", nom);
    url.searchParams.set("niveau", niveau);
    const res = await fetch(url, { method: "POST" });
    return res.json();
  },

  async enregistrerReponse(userId, phraseId, reussi, contexte = "") {
    return this._post("/progress/answer", { user_id: userId, phrase_id: phraseId, reussi, contexte });
  },

  async progression(userId) {
    return this._get(`/progress/${userId}`);
  },

  async reinitialiserProgression(userId) {
    const res = await fetch(`${API_BASE_URL}/progress/${userId}`, { method: "DELETE" });
    return res.json();
  },

  // ------------------------------------------------------------
  // DATASET
  // ------------------------------------------------------------
  async exemples(nb = 10) {
    return this._get("/examples", { nb });
  },
  async rechercher(q, k = 5) {
    return this._get("/search", { q, k });
  },
  async stats() {
    return this._get("/stats");
  },
  async audioBrowse(q = "", limit = 20) {
    return this._get("/speak/browse", { q, limit });
  },

  // ------------------------------------------------------------
  // TEST AUDIO (quiz audio)
  // ------------------------------------------------------------
  async audioQuizRandom(niveau = "debutant", contexte = null) {
    let url = `${API_BASE_URL}/audio/quiz/random?niveau=${niveau}`;
    if (contexte) url += `&contexte=${encodeURIComponent(contexte)}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Erreur ${res.status}`);
    return res.json();
  },

  async audioQuizVerify(phraseId, userAnswer) {
    const fd = new FormData();
    fd.append("phrase_id", phraseId);
    fd.append("user_answer", userAnswer);
    const res = await fetch(`${API_BASE_URL}/audio/quiz/verify`, { method: "POST", body: fd });
    if (!res.ok) throw new Error(`Erreur ${res.status}`);
    return res.json();
  },
};

// ------------------------------------------------------------
// CLASSES UTILITAIRES (conversation, micro)
// ------------------------------------------------------------
class ConversationDioula {
  constructor(niveau = "debutant") {
    this.niveau = niveau;
    this.historique = [];
  }
  async envoyer(message, theme = null) {
    const reponse = await KoumankanAPI.chat(message, this.historique, this.niveau, theme);
    this.historique.push({ role: "user", content: message });
    this.historique.push({ role: "assistant", content: reponse });
    if (this.historique.length > 20) this.historique = this.historique.slice(-20);
    return reponse;
  }
  reinitialiser() {
    this.historique = [];
  }
}

class MicRecorder {
  constructor() {
    this.mediaRecorder = null;
    this.chunks = [];
    this.isRecording = false;
  }
  async start() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    this.chunks = [];
    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) this.chunks.push(e.data);
    };
    this.mediaRecorder.start();
    this.isRecording = true;
  }
  stop() {
    return new Promise((resolve) => {
      this.mediaRecorder.onstop = () => {
        const blob = new Blob(this.chunks, { type: "audio/webm" });
        this.mediaRecorder.stream.getTracks().forEach((t) => t.stop());
        this.isRecording = false;
        resolve(blob);
      };
      this.mediaRecorder.stop();
    });
  }
  async stopAndTranslate(direction = "fr_to_dyu", niveau = "debutant") {
    const blob = await this.stop();
    return KoumankanAPI.traduireVoix(blob, direction, niveau);
  }
}