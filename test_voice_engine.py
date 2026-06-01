from voice_engine import VoiceEngine
ve = VoiceEngine()
print(f"Audios chargés : {len(ve._df_audio) if ve._df_audio is not None else 0}")
if ve._df_audio is not None:
    print("Exemple :", ve._df_audio.iloc[0]["dyu"])