import nltk
from nltk import pos_tag
from nltk import tokenize

def extract_phrases(my_tree, phrase):
    my_phrases = []
    if my_tree.label() == phrase:
        my_phrases.append(my_tree.copy(True))
    for child in my_tree:
        if type(child) is nltk.Tree:
            list_of_phrases = extract_phrases(child, phrase)
            if len(list_of_phrases) > 0:
                my_phrases.extend(list_of_phrases)
    return my_phrases



def main():
    sentences = ["The little yellow dog barked at the cat",
                 "He studies Information Technology",
                 "When a message from Enrico Hernandez arrives, get the necklace price; Convert it from Chilean Pesos to Euro; If it costs less than 100 EUR, send to him a message asking him to buy it; If not, write saying I am not interested.".lower(),
                 "search for an image of Saturn in Flickr and attach it to an email. Send this mail to my son saying I love you.".lower(),
                 "make a payment using my bank account when I receive a message from Good Sports Store. Send the receipt back to the store.",
                 "pay in the bank my purchases that I will soon receive from contact@gs-store.de. Send the proof of payment back to the store.",
                 "pay the invoice that I will receive from contact@gs-store.de with my cred card, then send the payment receipt to them using the same address"]

    grammar = "NP: {(<VB>|<VBP>)+(<IN>|<PRP>|<TO>|<DT>|<JJ>|<NN>|<IN>)*<NN>+|<NNP>*<VB>}"
    # grammar = "NP: {<VB>+<IN>*<DT>*<JJ>*<NN>*<IN>*<NN>+|<NNP>*<VB>}"
    # grammar = "NP: {<DT>?<JJ>*<NN>|<NNP>*}"
    cp = nltk.RegexpParser(grammar)

    for x in sentences:
        sentence = pos_tag(tokenize.word_tokenize(x))
        # print sentence
        tree = cp.parse(sentence)
        print "\nNoun phrases:"
        list_of_noun_phrases = extract_phrases(tree, 'NP')
        for phrase in list_of_noun_phrases:
            print phrase, "_".join([x[0] for x in phrase.leaves()])












