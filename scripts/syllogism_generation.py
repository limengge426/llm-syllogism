import os
import json
import random
from pathlib import Path
import nltk
import nltk.data
from tqdm import tqdm

# --- NLTK Path Setup (Preserved) ---
ROOT = Path(__file__).resolve().parents[1]
NLTK_BASE = ROOT / "vendor"
WORDNET_DIR = NLTK_BASE / "corpora" / "wordnet"

if not (WORDNET_DIR / "index.noun").exists():
    raise SystemExit(
        f"WordNet not found：{WORDNET_DIR}/index.noun\n"
    )

os.environ["NLTK_DATA"] = str(NLTK_BASE)
if str(NLTK_BASE) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_BASE))

from nltk.corpus import wordnet as wn

try:
    _ = wn.synsets("dog")
    print("WordNet loaded from:", nltk.data.find("corpora/wordnet"))
except LookupError as e:
    print("Could not load WordNet. Check NLTK_DATA setup.")
    print("NLTK_BASE path:", NLTK_BASE)
    print("NLTK full path:", nltk.data.path)
    raise
# --- End NLTK Setup ---

def find_smp_chain(start_word: str):
    """Finds a valid (Subject, Middle, Predicate) hyponym chain."""
    s_synsets = wn.synsets(start_word, pos=wn.NOUN)[:1]
    if not s_synsets:
        return None
    
    s_synset = s_synsets[0]
    
    m_hypernyms = s_synset.hypernyms()
    if not m_hypernyms:
        return None
    m_synset = random.choice(m_hypernyms)
    
    p_hypernyms = m_synset.hypernyms()
    if not p_hypernyms:
        return None
    p_synset = random.choice(p_hypernyms)
    
    s_name = s_synset.lemmas()[0].name().replace('_', ' ')
    m_name = m_synset.lemmas()[0].name().replace('_', ' ')
    p_name = p_synset.lemmas()[0].name().replace('_', ' ')
    
    if s_name != m_name and m_name != p_name and s_name != p_name:
        if p_synset.lexname() != "noun.Tops": # Avoid generic root 'entity'
            return (s_name, m_name, p_name)
    
    return None

# Base terms to start mining from
START_WORDS = [
    "dog", "cat", "lion", "tiger", "horse", "cow", "sheep", "ant", "bee", "spider", "eagle", "shark", "salmon",
    "oak", "pine", "rose", "daisy", "apple", "banana", "carrot", "potato",
    "chair", "table", "car", "bicycle", "computer", "book", "hammer", "guitar",
    "triangle", "square", "rectangle", "pentagon", "hexagon", "circle", "ellipse",
    "doctor", "lawyer", "engineer", "artist", "king", "president",
    "love", "justice", "anger", "science", "mathematics", "history"
]

# 24 Valid Forms
VALID_MOODS = {
    1: ["AAA","AAI","AII","EAE","EAO","EIO"],
    2: ["AEE","AEO","AOO","EAE","EAO","EIO"],
    3: ["AAI","AII","EAO","EIO","IAI","OAO"],
    4: ["AAI","AEE","AEO","EAO","EIO","IAI"]
}

# 20 Invalid Forms
INVALID_MOODS = {
    1: ["AEE", "AEO", "AOO", "IAI", "OAO"],
    2: ["AAA", "AAI", "AII", "IAI", "OAO"],
    3: ["AAA", "AEE", "AEO", "AOO", "EAE"],
    4: ["AAA", "AII", "AOO", "EAE", "OAO"]
}

def realize(m, X, Y):
    """Realizes a logical mood (A,E,I,O) as a natural language sentence."""
    return {
        "A": f"All {X} are {Y}.",
        "E": f"No {X} are {Y}.",
        "I": f"Some {X} are {Y}.",
        "O": f"Some {X} are not {Y}."
    }[m]

def wire(figure):
    """Returns (premise1, premise2, conclusion) term patterns for a figure."""
    if figure == 1:   return (("M","P"), ("S","M"), ("S","P"))
    if figure == 2:   return (("P","M"), ("S","M"), ("S","P"))
    if figure == 3:   return (("M","P"), ("M","S"), ("S","P"))
    if figure == 4:   return (("P","M"), ("M","S"), ("S","P"))
    raise ValueError("figure must be 1-4")

def fill(tag, S,M,P): 
    """Helper to fill a pattern tag (e.g., 'S' or 'M') with its term."""
    return {"S":S,"M":M,"P":P}[tag]

def gen_valid_sample(fig, mood, S, M, P):
    """Generates a single 'validity' test item (Expected: Yes)."""
    m1,m2,mc_valid = mood
    patt1,patt2,pattC = wire(fig)
    p1 = realize(m1, fill(patt1[0],S,M,P), fill(patt1[1],S,M,P))
    p2 = realize(m2, fill(patt2[0],S,M,P), fill(patt2[1],S,M,P))
    context = [p1, p2]
    c_valid = realize(mc_valid, fill(pattC[0],S,M,P), fill(pattC[1],S,M,P))
    rand_id = random.randint(100000, 999999)
    base_data = {
        "figure": fig,
        "mood": mood,
        "domain": "WordNet_Hypernym",
        "context": context,
        "placeholders": {"S":S,"M":M,"P":P}
    }
    ex_valid = {
        **base_data,
        "id": f"{mood}-{fig}_WordNet_{rand_id}_validity",
        "type": "validity",
        "fact": c_valid,
        "expected_answer": "Yes"
    }
    return ex_valid

def gen_fallacy_sample(fig, mood, S, M, P):
    """Generates a single 'invalidity' (fallacy) test item (Expected: No)."""
    m1, m2, mc = mood
    patt1,patt2,pattC = wire(fig)
    p1 = realize(m1, fill(patt1[0],S,M,P), fill(patt1[1],S,M,P))
    p2 = realize(m2, fill(patt2[0],S,M,P), fill(patt2[1],S,M,P))
    context = [p1, p2]
    c_fallacy = realize(mc, fill(pattC[0],S,M,P), fill(pattC[1],S,M,P))
    rand_id = random.randint(100000, 999999)
    ex_fallacy = {
        "figure": fig,
        "mood": mood,
        "domain": "WordNet_Hypernym",
        "context": context,
        "placeholders": {"S":S,"M":M,"P":P},
        "id": f"{mood}-{fig}_WordNet_{rand_id}_fallacy",
        "type": "fallacy", # This corresponds to the paper's "Invalidity Test"
        "fact": c_fallacy,
        "expected_answer": "No"
    }
    return ex_fallacy

def main(n_per_combo=5, out="data/samples_wordnet.jsonl"):
    Path("data").mkdir(exist_ok=True, parents=True)
    random.seed(7)

    print(f"Mining {len(START_WORDS)} start words from WordNet to find (S,M,P) chains...")
    smp_chains = []
    for word in tqdm(START_WORDS):
        chain = find_smp_chain(word)
        if chain:
            smp_chains.append(chain)

    if not smp_chains:
        print("Fatal: Could not find any (S,M,P) chains in WordNet. Exiting.")
        return

    print(f"Successfully mined {len(smp_chains)} valid (S,M,P) chains.")
    lines = []
    # Counters now match the paper's two test types
    counters = {"validity": 0, "invalidity": 0}

    print("Generating Validity Test (Type 1) samples...")
    for fig, moods in VALID_MOODS.items():
        for mood in moods:
            for _ in range(n_per_combo):
                S, M, P = random.choice(smp_chains)
                # Call the new function that only generates the valid sample
                valid_sample = gen_valid_sample(fig, mood, S, M, P)
                lines.append(json.dumps(valid_sample, ensure_ascii=False))
                counters["validity"] += 1

    print("Generating Invalidity Test (Type 2 / Fallacy) samples...")
    for fig, moods in INVALID_MOODS.items():
        for mood in moods:
            for _ in range(n_per_combo):
                S, M, P = random.choice(smp_chains)
                fallacy = gen_fallacy_sample(fig, mood, S, M, P)
                lines.append(json.dumps(fallacy, ensure_ascii=False))
                # Renamed "fallacy" to "invalidity" for clarity
                counters["invalidity"] += 1

    Path(out).write_text("\n".join(lines), encoding="utf-8")

    total_valid = len(VALID_MOODS.items()) * len(VALID_MOODS[1]) * n_per_combo
    total_invalid = len(INVALID_MOODS.items()) * len(INVALID_MOODS[1]) * n_per_combo

    print("\n--- Generation Complete ---")
    print(f"Total lines written: {len(lines)} -> {out}")
    print(f"  Validity Test:     {counters['validity']} (Expected: {6000 if n_per_combo == 250 else 'N/A'})")
    print(f"  Invalidity Test:   {counters['invalidity']} (Expected: {5000 if n_per_combo == 250 else 'N/A'})")

if __name__ == "__main__":
    # Set n_per_combo to 250 as requested by the paper
    main(n_per_combo=250)
