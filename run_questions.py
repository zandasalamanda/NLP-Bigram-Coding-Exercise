from main import *

print("=" * 80)
print("QUESTION 1: Querying the LM")
print("=" * 80)
(train, test) = read_data()
lm = estimate_bigram_lm(train)

print()
query_lm(lm, "I like to")
print()
query_lm(lm, "I want to")

print()
print("=" * 80)
print("QUESTION 2: get_best_sentence with various prefixes")
print("=" * 80)

prefixes_q2 = ["I", "The", "We", "She", "They", "He", "It"]
for prefix in prefixes_q2:
    sentence = get_best_sentence(lm, prefix)
    print(f"Prefix '{prefix}': {' '.join(sentence)}")

print()
print("=" * 80)
print("QUESTION 3: sample_sentence with various prefixes (repeated)")
print("=" * 80)

prefixes_q3 = ["I", "The", "We", "She", "They", "He", "It"]
for prefix in prefixes_q3:
    print(f"\n--- Prefix '{prefix}' (3 samples) ---")
    for trial in range(3):
        sentence = sample_sentence(lm, prefix)
        print(f"  Sample {trial+1}: {' '.join(sentence)}")
