import os
import pandas as pd
from glob import glob

HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub/datasets--uvci--koumankan4dyula")
parquet_files = glob(os.path.join(HF_CACHE, "**/*.parquet"), recursive=True)

print(f"📁 {len(parquet_files)} fichiers parquet trouvés\n")

total_audios_valides = 0

for pq in parquet_files:
    print(f"\n🔍 Lecture de {os.path.basename(pq)}...")
    try:
        # Lecture comme dans VoiceEngine (sans sampling_rate)
        df = pd.read_parquet(pq, columns=["dyu", "fr", "audio"])
        df_audio = df[df["audio"].notna()]
        print(f"   Lignes avec audio non nul : {len(df_audio)}")
        
        for idx, row in df_audio.iterrows():
            if not isinstance(row["dyu"], str) or not isinstance(row["fr"], str):
                continue
            audio_dict = row["audio"]
            if not isinstance(audio_dict, dict):
                continue
            bytes_data = audio_dict.get("bytes")
            if bytes_data is not None and isinstance(bytes_data, bytes) and len(bytes_data) > 5000:
                total_audios_valides += 1
                # On s'arrête après le premier trouvé pour l'exemple
                print(f"   ✅ Premier audio valide trouvé !")
                print(f"      dyu : {row['dyu']}")
                print(f"      fr  : {row['fr']}")
                print(f"      taille bytes : {len(bytes_data)}")
                # Optionnel : tester la conversion WAV
                import io, wave
                try:
                    buf = io.BytesIO()
                    with wave.open(buf, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(bytes_data)
                    print(f"      ✅ Conversion WAV réussie ({buf.tell()} octets)")
                except Exception as e:
                    print(f"      ⚠️ Erreur conversion WAV : {e}")
                break  # on sort après le premier pour ne pas saturer
        else:
            print(f"   ⚠️ Aucun audio valide dans ce fichier")
    except Exception as e:
        print(f"   ❌ Erreur : {e}")

print(f"\n\n🎯 TOTAL AUDIOS VALIDES DANS TOUS LES PARQUETS : {total_audios_valides}")