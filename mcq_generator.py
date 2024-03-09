class MCQGenerator:
    def __init__(self):
        self.nlp = None
        self.spacy = None
        self.stoplist = None
        self.requests = None
        self.wordnet = None
        self.cache = {}

    @staticmethod
    def get_re():  # Define the get_re method
        import re
        return re

    def get_spacy(self):
        if not self.nlp:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        return self.nlp

    def get_stoplist(self):
        if not self.stoplist:
            from nltk.corpus import stopwords
            import string
            stop_words = set(stopwords.words())
            punctuation = set(string.punctuation)
            additional_words = {'-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'} | {'example', 'examples', 'task',
                                                                                         'entity', 'data', 'use',
                                                                                         'type', 'concepts', 'concept',
                                                                                         'learn', 'function', 'method',
                                                                                         'unit', 'fontionality',
                                                                                         'behavior', 'simple', 'ways',
                                                                                         'capsule', 'capsules',
                                                                                         'medicines', 'details'}
            self.stoplist = stop_words | punctuation | additional_words
        return self.stoplist

    def get_wordnet(self):
        if not self.wordnet:
            import nltk
            nltk.download('wordnet')
            from nltk.corpus import wordnet as wn
            self.wordnet = wn
        return self.wordnet

    def get_requests(self):
        if not self.requests:
            import requests
            self.requests = requests
        return self.requests

    def tokenize_sentences(self, text):
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)

    def get_nouns_multipartite(self, text):
        out = set()
        nlp = self.get_spacy()
        stoplist = self.get_stoplist()  # Reuse stoplist

        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in {'PERSON', 'ORG', 'GPE'}:
                out.add(ent.text)

        for token in doc:
            if token.text.lower() not in stoplist and token.pos_ in {'NOUN', 'PROPN'}:
                if any(it_term in token.text.lower() for it_term in
                       {'programming', 'software', 'algorithm', 'database', 'network', 'security', 'server', 'cloud',
                        'internet', 'protocol', 'web', 'application', 'interface', 'compiler', 'framework', 'year'}):
                    out.add(token.text)
                elif token.ent_type_ != '':
                    out.add(token.text)

        return out

    def get_wordsense(self, sent, word, preprocessed_word):
        wn = self.get_wordnet()  # Ensure wordnet is imported before using wn

        # Check if result is in cache
        if (sent, word, preprocessed_word) in self.cache:
            return self.cache[(sent, word, preprocessed_word)]

        # Your existing code for obtaining synsets and calculating lowest_index
        synsets = wn.synsets(preprocessed_word, 'n')
        if synsets:
            from pywsd.similarity import max_similarity
            from pywsd.lesk import adapted_lesk
            wup = max_similarity(sent, preprocessed_word, 'wup', pos='n')
            adapted_lesk_output = adapted_lesk(sent, preprocessed_word, pos='n')

            if wup is None or adapted_lesk_output is None:
                result = None
            else:
                try:
                    lowest_index = min(synsets.index(wup), synsets.index(adapted_lesk_output))
                    result = synsets[lowest_index]
                except ValueError:
                    result = None
        else:
            result = None

        # Store result in cache
        self.cache[(sent, word, preprocessed_word)] = result
        return result

    def get_distractors_wordnet(self, syn, preprocessed_word):
        wn = self.get_wordnet()

        if syn.hypernyms():
            for item in syn.hypernyms()[0].hyponyms():
                name = item.lemmas()[0].name().replace("_", " ")
                if name != preprocessed_word:
                    yield name.capitalize()

    def get_distractors_conceptnet(self, word):
        import requests  # Import requests module here
        if not self.requests:
            self.requests = requests

        preprocessed_word = word.lower().replace(" ", "_") if len(word.split()) > 0 else word.lower()
        original_word = preprocessed_word

        url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5" % (
            preprocessed_word, preprocessed_word)
        obj = self.requests.get(url).json()
        for edge in obj['edges']:
            link = edge['end']['term']
            url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10" % (link, link)
            obj2 = self.requests.get(url2).json()
            for edge in obj2['edges']:
                word2 = edge['start']['label']
                if word2.lower() != original_word:
                    yield word2

    def get_distractors_from_csv(self, input_file, keyword):
        import csv  # Moved import to the function level
        encodings = ['latin-1']
        for encoding in encodings:
            try:
                with open(input_file, 'r', newline='', encoding=encoding) as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        key_concept = row['Key Concept']
                        distractors = row['Distractors'].split(', ')
                        if keyword.lower() in key_concept.lower() or keyword.lower() in distractors:
                            for distractor in distractors:
                                if distractor.lower() != keyword.lower():
                                    yield distractor
            except UnicodeDecodeError:
                print(f"Error decoding file with encoding {encoding}. Trying another encoding...")
        return []

    def get_sentences_for_keyword(self, keywords, sentences):
        keyword_sentence_mapping = {}
        for keyword in keywords:
            keyword_sentence_mapping[keyword] = []

        for sentence in sentences:
            for keyword in keywords:
                if keyword in sentence:
                    keyword_sentence_mapping[keyword].append(sentence)

        return keyword_sentence_mapping

    def get_distractors(self, keyword_sentence_mapping):
        key_distractor_list = {}
        for keyword in keyword_sentence_mapping:
            sentences = keyword_sentence_mapping[keyword]
            if sentences:
                csv_distractors = list(self.get_distractors_from_csv('JAVA.csv', keyword))

                # Check if there are enough distractors from the CSV file
                if len(csv_distractors) >= 4:
                    key_distractor_list[keyword] = csv_distractors
                else:
                    wordsense = self.get_wordsense(sentences[0], keyword, preprocessed_word=keyword)
                    if wordsense:
                        distractors = list(self.get_distractors_wordnet(wordsense, preprocessed_word=keyword))
                        if not distractors:
                            distractors = list(self.get_distractors_conceptnet(keyword))
                        if distractors:
                            # Combine distractors from CSV and other sources
                            combined_distractors = csv_distractors + distractors
                            key_distractor_list[keyword] = combined_distractors[:4]  # Take the first 4 distractors

        return key_distractor_list

    def generate_mcqs(self, text_data):
        sentences = self.tokenize_sentences(text_data)
        keywords = self.get_nouns_multipartite(text_data)
        keyword_sentence_mapping = self.get_sentences_for_keyword(keywords, sentences)
        key_distractor_list = self.get_distractors(keyword_sentence_mapping)
        return self.generate_mcqs_from_data(keyword_sentence_mapping, key_distractor_list)

    def generate_mcqs_from_data(self, keyword_sentence_mapping, key_distractor_list):
        import random
        option_choices = ['a', 'b', 'c', 'd']

        for keyword in key_distractor_list:
            sentences = keyword_sentence_mapping[keyword]
            if sentences:
                sentence = sentences[0]
                pattern = self.get_re().compile(keyword, self.get_re().IGNORECASE)
                output = pattern.sub(" _______ ", sentence)

                distractors = list(key_distractor_list[keyword])

                # Add the keyword as a distractor if not already present
                distractors.append(keyword)

                # Shuffle distractors and select 3 unique distractors
                distractors = random.sample(distractors, min(4, len(distractors)))

                # Ensure distractors list has at least 3 elements
                distractors += [''] * (4 - len(distractors))

                # Shuffle the distractors again to randomize their order
                random.shuffle(distractors)

                mcq = {"question": output, "answer": keyword, "options": dict(zip(option_choices, distractors))}
                yield mcq









