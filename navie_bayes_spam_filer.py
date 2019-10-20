import email
import math
import os
from queue import PriorityQueue as PQ

def load_tokens(email_path):
    token_list = []
    email_obj = email.message_from_file(open(email_path))
    for sentence in email.iterators.body_line_iterator(email_obj):
        token_list = token_list + sentence.split()
    return token_list


def log_probs(email_paths, smoothing):
    prob_dict = {}
    for email_path in email_paths:
        bag_of_word = load_tokens(email_path)
        for word in bag_of_word:
            if word not in prob_dict:
                prob_dict[word] = 1
            else:
                prob_dict[word] += 1
    total_words, num_of_word = sum(prob_dict.values()), len(prob_dict)
    for word, frequency in prob_dict.items():
        prob_dict[word] = math.log((frequency+smoothing)/(total_words+smoothing*(num_of_word+1)))
    prob_dict.update({'<UNK>': math.log(smoothing/(total_words + smoothing * (num_of_word+1)))})
    return prob_dict


class SpamFilter(object):

    def __init__(self, spam_dir, ham_dir, smoothing):
        self.spam_dir = spam_dir
        self.ham_dir = ham_dir
        self.smoothing = smoothing
        self.spam_paths = []
        self.ham_paths = []
        for dirpath, dirnames, filenames in os.walk(spam_dir):
            for filename in filenames:
                self.spam_paths.append(dirpath+'/'+filename)
        for dirpath, dirnames, filenames in os.walk(ham_dir):
            for filename in filenames:
                self.ham_paths.append(dirpath+'/'+filename)
        self.spam_log_prob = log_probs(self.spam_paths, self.smoothing)
        self.ham_log_prob = log_probs(self.ham_paths, self.smoothing)
        self.ham_prob = math.log(len(self.ham_paths)/(len(self.spam_paths)+len(self.ham_paths)))
        self.spam_prob = math.log(len(self.spam_paths)/(len(self.spam_paths)+len(self.ham_paths)))

    def is_spam(self, email_path):
        word_dict = {}
        spam_log_prob, ham_log_prob = 0, 0
        for word in load_tokens(email_path):
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
        for word, count in word_dict.items():
            if word not in self.ham_log_prob:
                ham_log_prob += self.ham_log_prob["<UNK>"]
            else:
                ham_log_prob += self.ham_log_prob[word]
            if word not in self.spam_log_prob:
                spam_log_prob += self.spam_log_prob["<UNK>"]
            else:
                spam_log_prob += self.spam_log_prob[word]
        ham_prob = self.ham_prob + ham_log_prob
        spam_prob = self.spam_prob + spam_log_prob
        if spam_prob > ham_prob:
            return True
        return False

    def most_indicative_spam(self, n):
        spam_queue = PQ()
        for word, p_w_c in self.ham_log_prob.items():
            #the least indicative ham word is the most indicattive spam word
            if word not in self.spam_log_prob:
                #p_w = p(spam)+p(ham) not p(spam)*p(ham) so => e^(log(spam)) + e^(log(ham))
                p_w = math.exp(self.ham_log_prob[word]) + math.exp(self.spam_log_prob["<UNK>"])
            else:
                p_w = math.exp(self.spam_log_prob[word]) + math.exp(self.ham_log_prob[word])
            spam_queue.put((p_w_c - math.log(p_w), word))
        return [spam_queue.get()[1] for x in range(n)]

    def most_indicative_ham(self, n):
        ham_queue = PQ()
        for word, p_w_c in self.spam_log_prob.items():
            if word not in self.ham_log_prob:
                p_w = math.exp(self.spam_log_prob[word]) + math.exp(self.ham_log_prob["<UNK>"])
            else:
                p_w = math.exp(self.spam_log_prob[word]) + math.exp(self.ham_log_prob[word])
            ham_queue.put((p_w_c - math.log(p_w), word))
        return [ham_queue.get()[1] for x in range(n)]
