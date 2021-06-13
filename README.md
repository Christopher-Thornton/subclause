# SubClause

Automatic subordinate clause extractor

This is a non-identical Python port of the Java library [AutoSubclause](https://github.com/ctapweb/AutoSubClause).

The NLP library *SpaCy* is used for document annotation.

Extract subordinate clauses in English text and related information:
#### General
   - Clause Text
   - Clause Type
   - Clause Finiteness
   - Clause Subordinator
   - Level of Embededness 
#### Complement
   - Type (subject/object)
#### Adverbial
   - Semantic Function of the Adjunct
     - Time, Place, Condition, Reason, Concession, Purppose, Comparison, Manner, Results
#### Relative
   - Restrictiveness
   - Head Noun
   - Head Noun Animacy
   - Head Noun Role in Main Clause
   - Head Noun Role in Subordinate Clause

```python
import spacy
nlp = spacy.load('en_core_web_sm')

from subclause import SubClauseFinder
subclausefinder = SubClauseFinder()

text = "The door opened because the man pushed it. " \
       "I wondered whether the homework was necessary. " \
       "They will visit you before they go to the airport. " \
       "Before they go to the airport, they will visit you. " \
       "I went to the show that was very popular."

doc = nlp(text)

for sent in doc.sents:
    print('\nSentence:', sent)
    subclauses = subclausefinder.get_subclauses(sent)
    for sc in subclauses:
        print("\tclause text:", sc.get_clause_span())
        print("\tis finite:", sc.get_is_finite())
        print("\tsubordinator:", sc.get_subordinator())
        print("\tembeddedness:", sc.get_embeddedness())
        print("\tclause type:", sc.get_clause_type())
        # for complement clauses
        if sc.get_clause_type() == 'COMPLEMENT':
            print("\t\tcomplement type:", sc.get_complement_type())
        # for adverbial clauses
        elif sc.get_clause_type() == 'ADJUNCT':
            print("\t\tadjunct function:", sc.get_adjunct_function())
        # for relative clauses
        elif sc.get_clause_type() == 'RELATIVE':
            print("\t\tis restrictive:", sc.get_is_restrictive())
            print("\t\thead noun:", sc.get_head_noun())
            print("\t\tis head noun animate:", sc.get_head_noun_animacy())
            print("\t\thead noun role in main clause:", sc.get_head_noun_role_in_main_clause())
            print("\t\thead noun role in subordinate clause:", sc.get_head_noun_role_in_sub_clause())
```
### See also

Xiaobin Chen et al., Automatic extraction of subordinate clauses and its application in second language acquisition research [[1]](https://www.researchgate.net/publication/344039283_Automatic_extraction_of_subordinate_clauses_and_its_application_in_second_language_acquisition_research)
