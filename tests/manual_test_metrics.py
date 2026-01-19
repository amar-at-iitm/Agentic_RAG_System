
from evaluation.metrics import calculate_rouge
import sys

def test_rouge():
    ref = "The quick brown fox jumps over the dog."
    cand = "The quick brown fox jumps over the lazy dog."
    
    scores = calculate_rouge(ref, cand)
    print("Scores:", scores)
    
    if scores['rouge1'] > 0 and scores['rougeL'] > 0:
        print("ROUGE metric test PASSED")
    else:
        print("ROUGE metric test FAILED")
        sys.exit(1)

if __name__ == "__main__":
    test_rouge()
