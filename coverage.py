from nltk.tokenize import sent_tokenize
import llm_utils

def detect_coverage(x, a=None, r=None):
    # Given input CQA, best answer so far, and rating, return an answer with the same rating with minimal coverage.
    # now only c and q are needed
    (c, q, _a) = x
    # split into sentences
    sents = [s.replace("\n", " ") for s in sent_tokenize(c)]
    # query coverage
    coverage = llm_utils.select_relevant_sents(q, sents)
    total = 0
    covered = 0
    for i, s in enumerate(sents):
        wc = llm_utils.word_cnt(s)
        total += wc
        if i in coverage:
            covered += wc
    records = {"sents": sents, "coverage": coverage, "total": total}
    return covered / total, len(coverage) / len(sents), records

if __name__ == "__main__":
    c = """\
In an article about 'Royton', section 'Overview', subsection 'nan', paragraph 'nan', it mentioned: 
Royton ( pop . 21,284 ( 2011 ) ) is a town within the Metropolitan Borough of Oldham , in Greater Manchester , England . It is situated close to the source of the River Irk , near undulating land at the foothills of the South Pennines , 1.7 miles ( 2.7 km ) north-northwest of Oldham , 3.2 miles ( 5.1 km ) south-southeast of Rochdale and 7.6 miles ( 12.2 km ) northeast of the city of Manchester .
"""
    q = """\
What is the population of Royton as recorded in 2011 and how is it geographically situated within Greater Manchester, England?
"""
    from pprint import pprint 
    ratio, records = detect_coverage((c, q, None))
    print(ratio)
    print("Coverage:", records["coverage"])
    print("Sentences:")
    for i, s in enumerate(records["sents"]):
        print(i, f"wc={llm_utils.word_cnt(s)}", s)