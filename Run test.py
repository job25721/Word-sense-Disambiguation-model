from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import csv
import deepcut

def cut_words(words,targetLength,windowSize):
    count = 0
    for sen in words:
        target = 0
        i = 0
        if len(sen) > targetLength:
            for word in sen:
                for char in range(len(word)):
                    if word[char] == "เ" and word[char + 1] == "ข" and word[char + 2] == "า":
                        target = i
                        break
                    else:
                        pass
                if target != 0:
                    break
                else:
                    pass
                i = i + 1
            if (targetLength-1) - (target + windowSize) > 0:
                words[count] = sen[:targetLength]
            else:
                start = (target + windowSize + 1) - targetLength
                words[count] = sen[start:target + windowSize + 1]
        else:
            pass
        count=count+1
    return words


def read_input(path):
    f = open(path, 'r', encoding='utf-8-sig')
    data = list(f)
    sentences = [d.split(':')[2].split('\n')[0] for d in data]
    print(sentences)
    print("tokenizing...")
    words = [[w for w in deepcut.tokenize(s) if w != ' '] for s in sentences]
    print("tokenize complete")
    max_sentence_length = max([len(s) for s in words])
    if max_sentence_length > 93:
        print("Cuttingword...")
        words = cut_words(words=words, targetLength=93, windowSize=2)
        print("cutted")
    return words

def extract_word_vectors(words,max_sentence_length):
    print("extracting word2vec...")
    vocab = set([w for s in words for w in s])

    pretrained_word_vec_file = open('cc.th.300.vec', 'r', encoding='utf-8-sig')
    count = 0
    vocab_vec = {}
    for line in pretrained_word_vec_file:
        if count > 0:
            line = line.split()
            if (line[0] in vocab):
                vocab_vec[line[0]] = line[1:]
        count = count + 1

    word_vector_length = 300
    word_vectors = np.zeros((len(words), max_sentence_length, word_vector_length))
    sample_count = 0
    for s in words:
        word_count = 0
        for w in s:
            try:
                word_vectors[sample_count, max_sentence_length - word_count - 1, :] = vocab_vec[w]
                word_count = word_count + 1
            except:
                pass
        sample_count = sample_count + 1

    print("extract word2vec complete")
    return word_vectors


def write_output(y_pred,path,name):
    # write output
    open(path, 'r+').truncate(0)  # clear output.txt
    count = 1
    for predict in y_pred:
        if predict == 0:
            predict = "H\n"
        elif predict == 1:
            predict = "P\n"
        elif predict == 2:
            predict = "M\n"
        write_pattern = str(count) + "::" + predict
        f = open(path, mode='a')
        f.write(write_pattern)
        count = count+1


def accuracy(model, word_vectors, labels, name):
    # test with ans
    score = model.evaluate(word_vectors, to_categorical(labels))  # with ans
    print(name,end=" ")
    print("Test Accuracy : ", score[1])


def loadCalc(path,wordVec,labels):
    model = load_model(path)
    accuracy(model=model,word_vectors=wordVec,labels=labels,name=path)

def main():
    # ------------------------- Read label to compare -------------------------------
    # d = open('testset_real/real_ans.txt', 'r')
    # l = list(d)
    # labels = [d.split('\n')[0] for d in l]

    # ------------------------- Read test data from txt ------------------------------
    words = read_input(path='testset_real/input.txt')
    # ------------------- Extract word vectors -------------------------
    word_vectors = extract_word_vectors(words=words,max_sentence_length=93)
    # -------------------------- Evaluation-----------------------------
    model = load_model('Saved_model/my_model.h5')
    model.summary()
    write_output(y_pred=model.predict(word_vectors).argmax(axis=1),path='Test set/ans.txt')



if __name__ == "__main__":
    main()