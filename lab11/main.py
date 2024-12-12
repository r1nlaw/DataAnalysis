from collections import defaultdict
import random
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt_tab")


def ngram(text: str, n: int = 3) -> str:
    """
    Function for calculation ngram
    """
    words = word_tokenize(text)
    ngrams = zip(*[words[i:] for i in range(n)])
    trans = defaultdict(list)
    init = []
    for i in ngrams:
        if i[0] == ".":
            init.append(i[1 : n - 1])
        trans[i[0 : n - 1]].append(i[-1])
    result = random.choice(init)
    fr = ["."]
    res_true = [result[0]]
    while True:
        candidates = trans[tuple(fr + list(result))]
        next_word = random.choice(candidates)
        fr = [result[0]]
        result = list(result[1:]) + [next_word]
        res_true.append(result[0])
        if result[0] == ".":
            return " ".join(res_true)


def main() -> None:
    """
    main function
    """
    with open("data.txt", "r+", encoding="UTF8") as f:
        text = f.read()
        print(f"Trigram {ngram(text, 3)}")
        print(f"Fourgram {ngram(text, 4)}")


if __name__ == "__main__":
    main()
