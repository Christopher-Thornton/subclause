import itertools
import networkx as nx


def get_animate_words():
    animate_words = set(line.strip() for line in open('../animacy/animate.unigrams.txt', 'r', encoding='utf8'))
    animate_words.update({"i", "me", "myself", "mine", "my", "we", "us", "ourself", "ourselves", "ours", "our",
                          "you", "yourself", "yours", "your", "yourselves", "he", "him", "himself", "his", "she",
                          "her", "herself", "hers", "her", "one", "oneself", "one's", "they", "them", "themself",
                          "themselves", "theirs", "their", "they", "them", "'em", "themselves", "who", "whom",
                          "whose"})
    return animate_words


def get_inanimate_words():
    inanimate_words = set(line.strip() for line in open('../animacy/inanimate.unigrams.txt', 'r', encoding='utf8'))
    inanimate_words.update({"it", "itself", "its", "where", "when"})
    return inanimate_words


ANIMATE = get_animate_words()
INANIMATE = get_inanimate_words()


class SubClauseFinder:
    def __init__(self):
        # target relations of dependency parsing
        self.TARGET_RELATIONS = {'relcl', 'advcl', 'ccomp', 'csubj', 'csubjpass', 'xcomp'}

    def get_dependency_tree(self, root):
        # SpaCy dependency parse doesnt return a tree, start from the root token and
        # navigate down the tree via .children
        dependency_tree = [root]
        while sum([len(list(tok.children)) for tok in dependency_tree[-1]]) > 0:
            dependency_tree.append(list(itertools.chain.from_iterable(
                [list(tok.children) for tok in dependency_tree[-1]])))
        dependency_tree = list(itertools.chain.from_iterable(dependency_tree))
        return dependency_tree

    def get_subclauses(self, annotated_sent):
        root = [token for token in annotated_sent if token.dep_ == 'ROOT']
        dependency_tree = self.get_dependency_tree(root)

        # iterate the edges to find dependent clauses relations
        subordinate_edges = []
        for clause_root in dependency_tree:
            if clause_root.dep_ in self.TARGET_RELATIONS:
                subordinate_edges.append(clause_root)

        subclauses = []
        for clause_root in subordinate_edges:
            clause_type = self.identify_clause_type(clause_root.dep_)
            # extract information of specific clause type
            if clause_type == 'RELATIVE':
                clause = RelativeClause(annotated_sent, clause_root)
            elif clause_type == 'ADJUNCT':
                clause = AdjunctClause(annotated_sent, clause_root)
            elif clause_type == 'COMPLEMENT':
                clause = ComplementClause(annotated_sent, clause_root)
            else:
                raise ValueError
            subclauses.append(clause)
        return subclauses

    def identify_clause_type(self, clause_root_dep):
        if clause_root_dep == 'relcl':
            return 'RELATIVE'
        elif clause_root_dep == 'advcl':
            return 'ADJUNCT'
        elif clause_root_dep in {'ccomp', 'csubj', 'csubjpass', 'xcomp'}:
            return 'COMPLEMENT'
        else:
            raise ValueError


class SubordinateClause:
    # Abstract class for storing subordinate clause information.
    # Actual subordinate clauses extend this class.
    def __init__(self, annotated_sent, clause_root):
        self.annotated_sent = annotated_sent
        self.clause_root = clause_root
        self.clause_root_dep = clause_root.dep_
        # identify clause finiteness
        self.is_finite = None
        # type of subordinate clause
        self.clause_type = None
        self.clause_span = None
        # subordinator is'mark' in adverbial clauses and complement clause but 'ref' in relative clause
        self.subordinator = None
        # level of embeddedness, main clause at level 0
        self.embeddedness = None

    def get_is_finite(self):
        if self.is_finite is None:
            self.identify_finiteness()
        return self.is_finite

    def get_clause_type(self):
        if self.clause_type is None:
            self.identify_clause_type()
        return self.clause_type

    def get_clause_span(self):
        if self.clause_span is None:
            self.set_clause_span()
        return self.clause_span

    def get_subordinator(self):
        if self.subordinator is None:
            self.identify_subordinator()
        return self.subordinator

    def get_embeddedness(self):
        if self.embeddedness is None:
            self.count_embeddedness()
        return self.embeddedness

    def referent_dependency(self, outedge):
        """
        https://www.mathcs.emory.edu/~choi/doc/cu-2012-choi.pdf
        A referent is the relation between a wh-complementizer in a relative clause and its referential head. In
        Referent relations are represented as secondary dependencies because integrating them with
        other dependencies breaks the single-head tree property (e.g., which would have multiple heads in Figure 28).
        """
        # TODO: Not Implemented
        return False

    def identify_subordinator(self):
        # for relative clauses, find the "referent"
        if self.get_clause_type() == 'RELATIVE':
            head_noun = self.clause_root.head
            for child in head_noun.children:
                if self.referent_dependency(child):
                    self.subordinator = child
        else:
            # for adverbial and complement clauses, find the "mark"
            for child in self.clause_root.children:
                child_dep = child.dep_
                if child_dep == 'mark':  # MARKER
                    self.subordinator = child

    def set_clause_span(self):
        self.clause_span = self.make_span(self.clause_root)

    def count_embeddedness(self):
        sent_root = [token for token in self.annotated_sent if token.dep_ == 'ROOT'][0]
        if sent_root.head.i == sent_root.i or sent_root.i == self.clause_root.i:
            self.embeddedness = 1
            return
        # find number of edges to go from clause root to sent root
        # Load spaCy's dependency tree into a networkx graph
        edges = []
        for token in self.annotated_sent:
            for child in token.children:
                edges.append(('{0}'.format(token.i),
                              '{0}'.format(child.i)))
        graph = nx.Graph(edges)
        # Get the length and path
        levels = nx.shortest_path_length(graph, source=str(self.clause_root.i), target=str(sent_root.i))
        return levels

    def identify_clause_type(self):
        if self.clause_root_dep == 'relcl':
            self.clause_type = 'RELATIVE'
        elif self.clause_root_dep == 'advcl':
            self.clause_type = 'ADJUNCT'
        elif self.clause_root_dep in {'ccomp', 'csubj', 'csubjpass', 'xcomp'}:
            self.clause_type = 'COMPLEMENT'

    def identify_finiteness(self):
        """
        check if the sub clause finite
        Finite clauses are clauses that contain verbs which show tense. Otherwise they are nonfinite.
        some examples:
        I had something to eat [before leaving].
        [After having spent six hours at the hospital], they eventually came home.
        [Helped by local volunteers], staff at the museum have spent many years cataloguing photographs.
        He left the party and went home, [not having anyone to talk to].
        The person to ask [about going to New Zealand] is Beck.
        You have to look at the picture really carefully [in order to see all the detail].
        """
        # xcomp is nonfinite by definition
        if self.clause_root_dep == 'xcomp':
            self.is_finite = False
            return
        # the verb is the root of the clause
        idx_word_before_verb = self.clause_root.i - 1
        verb_pos = self.clause_root.pos_
        if idx_word_before_verb < self.annotated_sent.start:
            if verb_pos in {"VBG"  "VBN"}:
                self.is_finite = False
                return
            else:
                # not VBG or VBN, then finite
                self.is_finite = True
                return
        wordBeforeVerb = self.annotated_sent[idx_word_before_verb - self.annotated_sent.start]
        #  if the verb follows TO or a preposition, it is nonfinite
        posWordBeforeVerb = wordBeforeVerb.pos_
        if posWordBeforeVerb in {"IN", "TO"}:
            self.is_finite = False
            return
        # if verb is gerund (VBG), it must have an aux, otherwise nonfinite
        if verb_pos == "VBG":
            hasAux = False
            # check if there is aux
            for child in self.clause_root.children_:  # childIterable(self.clause_root)
                rel = child.dep_
                if rel == "aux":
                    hasAux = True
            if not hasAux:
                self.is_finite = False
                return
        # if verb is past participle (VBN), it must have aux/auxpass which is not VBGs, otherwise non-finite
        if verb_pos == "VBN":
            vbg_aux = False
            # check if there is aux that is not in gerund form
            for child in self.clause_root.children_:  # childIterable
                if child.dep_ in {"aux" "auxpass"}:
                    # get pos of aux
                    aux = child  # child.getDependent()
                    auxPOS = aux.pos_
                    if auxPOS == "VBG":
                        vbg_aux = True
                    if vbg_aux:
                        self.is_finite = False
                        return
        self.is_finite = True

    def make_span(self, word):
        i = word.i - self.annotated_sent.start
        span = self.annotated_sent[
               self.annotated_sent[i].left_edge.i - self.annotated_sent.start:
               self.annotated_sent[i].right_edge.i + 1 - self.annotated_sent.start]

        return span


class RelativeClause(SubordinateClause):
    def __init__(self, annotated_sent, clause_root):
        super().__init__(annotated_sent, clause_root)
        self.clause_type = "RELATIVE"
        # further information related to relative clause
        self.is_restrictive = None
        # head noun
        # TODO: Decide if noun chunking should be used
        self.head_noun = self.clause_root.head
        self.is_head_noun_animate = None
        # head noun role in main clause
        self.head_noun_role_in_main_clause = None
        self.head_noun_role_in_sub_clause = None
        # relative clauses's embeddedness is different from the other two types of clause
        self.embeddedness = max(self.get_embeddedness() - 1, 1)

    def get_head_noun(self):
        return self.head_noun

    def get_head_noun_animacy(self):
        if self.get_head_noun() is not None:
            if self.is_head_noun_animate is None:
                self.set_head_noun_animacy()
            return self.is_head_noun_animate

    def get_head_noun_role_in_main_clause(self):
        if self.get_head_noun() is not None:
            if self.head_noun_role_in_main_clause is None:
                self.set_head_noun_roles()
            return self.head_noun_role_in_main_clause

    def get_head_noun_role_in_sub_clause(self):
        if self.get_head_noun() is not None:
            if self.head_noun_role_in_sub_clause is None:
                self.set_head_noun_roles()
            return self.head_noun_role_in_sub_clause

    def get_is_restrictive(self):
        if self.is_restrictive is None:
            self.set_restrictiveness()
        return self.is_restrictive

    def set_head_noun_animacy(self):
        # TODO: use alternate method to detect animacy (Language Models)
        if self.get_head_noun() in ANIMATE:
            self.is_head_noun_animate = True
        else:
            self.is_head_noun_animate = False

    def set_head_noun_roles(self):
        # TODO: Check function
        # https://www.brighthubeducation.com/english-homework-help/32754-the-functions-of-nouns-and-noun-phrases/
        is_from_inside_rc = False
        edge = self.get_head_noun()
        relation = edge.dep_
        head_idx = edge.head.i

        # see if it is from inside or outside of the RC
        span = self.get_clause_span()
        if span.start <= head_idx <= span.end-1:
            is_from_inside_rc = True

        if relation in {'nsubj', 'nsubjpass'}:
            self.set_role('SUBJECT', is_from_inside_rc)
        elif relation == 'dobj':
            self.set_role('DIRECT_OBJECT', is_from_inside_rc)
        elif relation == 'pobj':  # 'iobj'
            self.set_role('INDIRECT_OBJECT', is_from_inside_rc)
        elif relation == 'nmod':
            self.set_role('PREPOSITION_COMPLEMENT', is_from_inside_rc)
        elif relation == 'appos':
            self.set_role('APPOSITIVE', is_from_inside_rc)

    def set_role(self, role, is_from_inside_rc):
        if is_from_inside_rc:
            self.head_noun_role_in_sub_clause = role
        else:
            self.head_noun_role_in_main_clause = role

    def set_restrictiveness(self):
        # if zero relativizer or "that", restrictive
        subordinator = self.get_subordinator()
        if subordinator is None or subordinator.text.lower() == "that":
            self.is_restrictive = True
            return

        head_noun = self.get_head_noun()
        if head_noun is not None:
            # if the head noun is personal pronoun or proper noun(s), the clause is nonrestrictive
            head_noun_pos = head_noun.pos_
            if head_noun_pos in {"NNP", "NNPS", "PRP"}:
                self.is_restrictive = False
                return
            # if the head noun is modified by an indefinite determiner like 'a', 'some', or 'any', restrictive
            for child in head_noun.children:
                relation = child.dep_
                if relation == 'det':  # DETERMINER
                    determiner = child.text.lower()
                    if determiner in {"a", "an", "some", "any"}:
                        self.is_restrictive = True
                        return
        self.is_restrictive = True


class AdjunctClause(SubordinateClause):
    # function of clause, e.g. temporal, modal, instrumental...
    def __init__(self, annotated_sent, clause_root):
        super().__init__(annotated_sent, clause_root)

        self.clause_type = "ADJUNCT"

        self.TIME_SUBORDINATORS = {"when", "before", "after", "since", "while", "as", "till", "until"}
        self.PLACE_SUBORDINATORS = {"where", "wherever", "anywhere", "everywhere"}
        self.CONDITION_SUBORDINATORS = {"if", "unless", "lest", "provided"}
        self.REASON_SUBORDINATORS = {"because", "since", "as", "given"}
        self.CONCESSION_SUBORDINATORS = {"although", "though"}
        self.PURPOSE_SUBORDINATORS = {"so", "to"}
        self.COMPARISON_SUBORDINATORS = {"than"}
        self.MANNER_SUBORDINATORS = {"like", "way"}
        self.RESULTS_SUBORDINATORS = {"so", "such"}

        self.adjunct_function = None

    def get_adjunct_function(self):
        if self.adjunct_function is None:
            self.assign_function()
        return self.adjunct_function

    def assign_function(self):
        subordinator = self.get_subordinator()
        if subordinator is None:
            self.adjunct_function = None
            return
        subordinator = subordinator.text.lower()
        if subordinator in self.TIME_SUBORDINATORS:
            self.adjunct_function = 'TIME'
        elif subordinator in self.PLACE_SUBORDINATORS:
            self.adjunct_function = 'PLACE'
        elif subordinator in self.CONDITION_SUBORDINATORS:
            self.adjunct_function = 'CONDITION'
        elif subordinator in self.REASON_SUBORDINATORS:
            self.adjunct_function = 'REASON'
        elif subordinator in self.CONCESSION_SUBORDINATORS:
            self.adjunct_function = 'CONCESSION'
        elif subordinator in self.PURPOSE_SUBORDINATORS:
            self.adjunct_function = 'PURPOSE'
        elif subordinator in self.COMPARISON_SUBORDINATORS:
            self.adjunct_function = 'COMPARISION'
        elif subordinator in self.MANNER_SUBORDINATORS:
            self.adjunct_function = 'MANNER'
        elif subordinator in self.RESULTS_SUBORDINATORS:
            self.adjunct_function = 'RESULTS'
        else:
            self.adjunct_function = 'OTHER'


class ComplementClause(SubordinateClause):
    def __init__(self, annotated_sent, clause_root):
        super().__init__(annotated_sent, clause_root)

        self.clause_type = "COMPLEMENT"
        # set complement type (subject or object)
        self.complement_type = None

    def get_complement_type(self):
        if self.complement_type is None:
            self.identify_complement_type()
        return self.complement_type

    def identify_complement_type(self):
        # ccomp is always object complement by definition
        if self.clause_root.dep_ == "ccomp":
            self.complement_type = 'OBJECT_COMPLEMENT'
            return
        # check governor/head of edge.
        # If it is outside the clause, it is an object complement, otherwise subject,
        # because English is an SVO language
        head_idx = self.clause_root.head.i

        span = self.get_clause_span()
        if span.start <= head_idx <= span.end-1:
            self.complement_type = 'OBJECT_COMPLEMENT'
        else:
            self.complement_type = 'SUBJECT_COMPLEMENT'


if __name__ == '__main__':
    import spacy
    nlp = spacy.load('en_core_web_sm')

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
