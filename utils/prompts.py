QUESTION_GENERATION_SYS_PROMPT = """\
You are to generate {NUM_QUESTIONS} self-contained short answer questions based on the facts mentioned in the following content.
Avoid questions that reference the content directly.
Each question should include all relevant context and directly name any referenced items, avoiding pronouns like "it," "the game," or "the person." 
Do not include phrases that reference the source or context, such as "mentioned in the article" or "according to the text."
Provide the questions in an ordered list.
"""

SUMMARIZE_QUESTION_TYPE_SYS_PROMPT = """\
Your task is to summarize the types of the following questions based on their form, not their topic.
Group questions with similar semantics together, focusing on the structure and form rather than content.
Create at least five groups that are balanced in terms of the number of questions, each with representative examples.
Present your summary in an ordered list. For each group:
- Start with a group name followed by a colon.
- List the question IDs in the group, separated by commas.
- Provide a brief justification of your grouping in a few sentences after each group.
"""

QUESTION_TYPES = """\
1. **Verification/Affirmation Questions**: These questions ask for confirmation about the equivalence or relationship between two or more entities. They often use formats like "Are...?" or "Which...?"
2. **Specific Fact and Figure Questions**: These questions request a specific quantitative or qualitative fact. They are straightforward and seek concrete data or a precise answer, often involving numbers or specific details.
3. **Identity and Attribution Questions**: These inquiries focus on identifying a person or entity responsible for an action or associated with a work. They tend to ask "Who...?" or refer to persons or origins related to a context.
4. **Which/What-Based General Knowledge Questions**: This group contains questions that start with "Which" or "What" and inquire about general knowledge, often requiring a selection from a set or identification of a type/category.
5. **Event/Outcome Questions**: These questions inquire about the outcome of specific events or actions, focusing on consequences or results. They often address changes, damages, or effects.
6. **Sequential/Ordering/Causation Questions**: These questions require identifying a sequence, comparison, or causation among entities, often using terms like "first," "before," "between," etc.
7. **Location-Based Questions**: These questions focus on identifying a geographic location or specific place where something is based or occurs.
8. **Descriptive/Characterization Questions**: These questions seek an explanation or characterization of entities, often requiring a description of how or why something is the way it is, involving traits or actions.
9. **Comparison and Selection Questions**: Questions in this group involve comparing two entities to determine which one holds a particular status or characteristic, often using formats like "Between X and Y, who/which is...?"
10. **Classification and Categorization Questions**: These inquiries request the classification or categorical identity of entities or things, often seeking to place an item within a broader group or category.
"""

CLASSIFY_QUESTION_TYPE_SYS_PROMPT = f"""\
Your task is to classify the given question based on their form, not their topic.
Refer to the following question types for guidance:
{QUESTION_TYPES}
You should output two lines:
The first line should be a number from 1 to 10, corresponding to the question type that best fits the given question.
The second line should be a brief justification for your choice.
"""

GENERATE_ANS_SYS_PROMPT = """\
You are to generate a short answer based on the following question and an optional supporting fact.
"""

CHECK_ANS_STAR_SYS_PROMPT = """\
You are to rate the following answer to a question, taking into account any optional supporting facts provided. 
Assign a rating from 0 to 5 based on the criteria below:
0: No answer or completely irrelevant
1: Significantly incorrect or incomplete
2: Partially correct; major inaccuracies or omissions
3: Correct but lacks depth; minimal detail
4: Mostly correct; minor errors; includes relevant details
5: Fully accurate and detailed; clear and comprehensive

Your response should consist of two lines:
The rating from 0 to 5.
A brief justification for your rating.
"""

GENERATE_ANS_SHORT_SYS_PROMPT = (
    GENERATE_ANS_SYS_PROMPT
    + """\
Provide a very concise answer without repeating the question.
"""
)

GENERATE_LIMIT_NUM_ANS_SYS_PROMPT = (
    GENERATE_ANS_SYS_PROMPT
    + """\
Please ensure that your answer contains no more than {} words.
"""
)

SELECT_RELEVANT_SENTS_SYS_PROMPT = """\
Select the minimal set of context sentences most relevant to answering the question. 
You need to choose at least one sentence and can select multiple sentences.
Output only the sentence numbers of these sentences in a comma-separated list on a single line without any additional text.
"""
