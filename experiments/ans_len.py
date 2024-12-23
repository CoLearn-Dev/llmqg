from utils.llm_utils import check_ans_star, word_cnt, generate_ans

def detect_ans_len_req(x, a, r=None):
    # Given input CQA, best answer so far, and rating, return an answer with the same rating with minimal length.
    current = word_cnt(a)
    if r is None:
        r = check_ans_star((x, a))
    trials = [{"l": -1, "aa": a, "rr": r, "wc": current}]
    if r[0] > 0:
        for l in [0, 1, 2, 3, 4, 8]:
            trial = {"l": l}
            aa = trial["aa"] = generate_ans(x, enforce_short=l)
            wc = trial["wc"] = word_cnt(aa)
            if wc < current:
                rr = trial['rr'] = check_ans_star((x, aa))
                trials.append(trial)
                if rr >= r:
                    current = wc
                    if l > 0:  # already enforcing within `l` words
                        break
    return current, trials


def packed_detect_ans_len_req(p):
    return detect_ans_len_req(*p)

if __name__ == "__main__":
    c = """\
In an article about 'Royton', section 'Overview', subsection 'nan', paragraph 'nan', it mentioned: 
Royton ( pop . 21,284 ( 2011 ) ) is a town within the Metropolitan Borough of Oldham , in Greater Manchester , England . It is situated close to the source of the River Irk , near undulating land at the foothills of the South Pennines , 1.7 miles ( 2.7 km ) north-northwest of Oldham , 3.2 miles ( 5.1 km ) south-southeast of Rochdale and 7.6 miles ( 12.2 km ) northeast of the city of Manchester .
"""
    q = """\
What is the population of Royton as recorded in 2011 and how is it geographically situated within Greater Manchester, England?
"""
    aa = """\
In 2011, the population of Royton was 21,284. Geographically, it is located within the Metropolitan Borough of Oldham in Greater Manchester, England, and is situated near the source of the River Irk, at the foothills of the South Pennines. It lies 1.7 miles north-northwest of Oldham, 3.2 miles south-southeast of Rochdale, and 7.6 miles northeast of the city of Manchester.
"""
    from pprint import pprint
    pprint(detect_ans_len_req((c, q, None), aa))