import os
import pandas as pd
from glob import glob

cache_path = os.path.expanduser("~/.cache/huggingface/hub/datasets--uvci--koumankan4dyula")
parquets = glob(os.path.join(cache_path, "**/*.parquet"), recursive=True)

if not parquets:
    print("❌ Aucun fichier .parquet trouvé dans le cache.")
    exit()

for pq in parquets:
    print(f"\n📁 {pq}")
    try:
        df = pd.read_parquet(pq, columns=["dyu", "fr", "audio", "sampling_rate"])
        non_null = df["audio"].notna().sum()
        print(f"   Lignes avec audio non nul : {non_null} / {len(df)}")
        
        if non_null > 0:
            # Examiner la première ligne avec audio non nul
            sample = df[df["audio"].notna()].iloc[0]["audio"]
            print(f"   Type de 'audio' : {type(sample)}")
            if isinstance(sample, dict):
                print(f"   Clés du dictionnaire : {list(sample.keys())}")
                if "bytes" in sample:
                    bytes_len = len(sample["bytes"]) if sample["bytes"] else 0
                    print(f"   Longueur des bytes : {bytes_len}")
                    if bytes_len > 5000:
                        print("   ✅ Audio valide (taille > 5KB)")
                    else:
                        print("   ⚠️ Audio trop petit ou vide")
                else:
                    print("   ❌ Pas de clé 'bytes'")
            else:
                print(f"   Contenu de 'audio' : {sample}")
    except Exception as e:
        print(f"   ⚠️ Erreur : {e}")