from pipeline import compare_tracks
r = compare_tracks('data/mippia_full/3_original.wav', 'data/mippia_full/3_similar.wav', verbose=True)
print(f"Score: {r['attribution_score']:.4f}")
print(f"MERT: {r.get('mert_similarity', 'MISSING')}")
print(f"Verdict: {r['verdict']}")
