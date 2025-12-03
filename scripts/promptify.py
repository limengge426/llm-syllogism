import json
from pathlib import Path

TEMPLATE_RAG = """Consider the context:
- {p1}
- {p2}

Is the conclusion correct GIVEN the context?
Conclusion: {c}

Return ONLY the following JSON object (no code fences, no extra text):
{{"valid":"Yes" or "No","why":"One short sentence"}}
"""

def main(inp="data/samples_wordnet.jsonl", outp="data/prompts_wordnet.jsonl"):
    lines = Path(inp).read_text(encoding="utf-8").strip().splitlines()
    outs = []
    for line in lines:
        ex = json.loads(line)
        
        prompt = TEMPLATE_RAG.format(
            p1=ex["context"][0], 
            p2=ex["context"][1], 
            c=ex["fact"]
        )
        
        output_data = {
            "id": ex["id"],
            "prompt": prompt,
            "expected_answer": ex["expected_answer"],
            "type": ex["type"], 
            "meta": {
                "figure": ex["figure"],
                "mood": ex["mood"],
                "domain": ex["domain"],
                "placeholders": ex["placeholders"]
            }
        }
        outs.append(output_data)

        
    Path(outp).write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in outs), encoding="utf-8")
    print(f"wrote {len(outs)} -> {outp}")

if __name__ == "__main__":
    main()
