from tqdm import tqdm
import json
import numpy as np
from collections import defaultdict
import csv
import random
import os


class RelationEntityBatcher():
    def __init__(self, input_dir, batch_size, path_length, entity_vocab, relation_vocab, mode = "train"):
        self.input_dir = input_dir
        self.input_file = input_dir+'/{0}.txt'.format(mode)
        self.batch_size = batch_size
        self.path_length = path_length
        print('Reading vocab...')
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.mode = mode
        self.create_triple_store(self.input_file)
        print("batcher loaded")


    def get_next_batch(self):
        if self.mode == 'train':
            yield self.yield_next_batch_train()
        else:
            yield self.yield_next_batch_test()


    def create_triple_store(self, input_file):
        self.paths = defaultdict(set)
        self.store_all_correct = defaultdict(set)
        triples = defaultdict(list)
        self.store = []
        if self.mode == 'train':
            with open(input_file) as raw_input_file:
                csv_file = csv.reader(raw_input_file, delimiter = '\t' )
                for line in csv_file:
                    e1 = self.entity_vocab[line[0]]
                    r = self.relation_vocab[line[1]]
                    e2 = self.entity_vocab[line[2]]
                    self.store.append([e1,r,e2])
                    self.store_all_correct[(e1, r)].add(e2)
        else:
            with open(input_file) as raw_input_file:
                csv_file = csv.reader(raw_input_file, delimiter = '\t' )
                for line in csv_file:
                    e1 = line[0]
                    r = line[1]
                    e2 = line[2]
                    if e1 in self.entity_vocab and e2 in self.entity_vocab:
                        e1 = self.entity_vocab[e1]
                        r = self.relation_vocab[r]
                        e2 = self.entity_vocab[e2]
                        self.store.append([e1,r,e2])
            fact_files = ['train.txt', 'test.txt', 'dev.txt', 'graph.txt']
            if os.path.isfile(self.input_dir+'/'+'full_graph.txt'):
                fact_files = ['full_graph.txt']
                print("Contains full graph")

            for f in fact_files:
            # for f in ['graph.txt']:
                with open(self.input_dir + '/' + f) as raw_input_file:
                    csv_file = csv.reader(raw_input_file, delimiter='\t')
                    for line in csv_file:
                        e1 = line[0]
                        r = line[1]
                        e2 = line[2]
                        if e1 in self.entity_vocab and e2 in self.entity_vocab:
                            e1 = self.entity_vocab[e1]
                            r = self.relation_vocab[r]
                            e2 = self.entity_vocab[e2]
                            self.store_all_correct[(e1, r)].add(e2)

        for e1, r, e2 in self.store:
            if e1 not in triples:
                triples[e1] = []
            triples[e1].append((r, e2))

        for e1 in list(triples):
            for r, e2 in triples[e1]:
                all_paths = self.get_paths(e1, e2, triples)
                path = all_paths[0] if all_paths else [e1, e2]
                self.paths[(e1, r, e2)].add(tuple(path))

        self.store = np.array(self.store)


    def path_batch(self, e1, r, e2):
        paths, lengths = [], []
        for i in range(e1.shape[0]):
            trail = self.paths[(e1[i], r[i], e2[i])]
            trail = [list(p) for p in trail if p]
            lengths.append([len(p) for p in trail])

            trail = [p if len(p) == self.path_length else \
                    p + [p[-1]] * (self.path_length - len(p)) for p in trail]
            paths.append(trail)

        all_paths = np.asarray(paths)
        lengths = np.asarray(lengths)
        all_paths = np.reshape(all_paths, [self.batch_size, -1])
        lengths = np.reshape(lengths, [self.batch_size])
        return all_paths, lengths


    def yield_next_batch_train(self):
        while True:
            batch_idx = np.random.randint(0, self.store.shape[0], size=self.batch_size)
            batch = self.store[batch_idx, :]
            e1 = batch[:,0]
            r = batch[:, 1]
            e2 = batch[:, 2]

            all_e2s = []
            for i in range(e1.shape[0]):
                all_e2s.append(self.store_all_correct[(e1[i], r[i])])

            all_paths, lengths = self.path_batch(e1, r, e2)
            assert e1.shape[0] == e2.shape[0] == r.shape[0] == len(all_e2s)
            yield e1, r, e2, all_e2s, all_paths, lengths


    def yield_next_batch_test(self):
        remaining_triples = self.store.shape[0]
        current_idx = 0
        while True:
            if remaining_triples == 0:
                return

            if remaining_triples - self.batch_size > 0:
                batch_idx = np.arange(current_idx, current_idx+self.batch_size)
                current_idx += self.batch_size
                remaining_triples -= self.batch_size
            else:
                current_idx = self.store.shape[0] - self.batch_size # cover full batch
                batch_idx = np.arange(current_idx, self.store.shape[0])
                remaining_triples = 0

            batch = self.store[batch_idx, :]
            e1 = batch[:,0]
            r = batch[:, 1]
            e2 = batch[:, 2]

            all_e2s = []
            for i in range(e1.shape[0]):
                all_e2s.append(self.store_all_correct[(e1[i], r[i])])

            all_paths, lengths = self.path_batch(e1, r, e2)
            assert e1.shape[0] == e2.shape[0] == r.shape[0] == len(all_e2s)
            yield e1, r, e2, all_e2s, all_paths, lengths


    def get_paths(self, src, dst, triples):
        all_paths, path = [], []
        num_vertices = len(triples)
        visited = [False]*(np.max(list(triples.keys())) + 1)

        def search(u, d, visited, path, all_paths):
            visited[u] = True
            path.append(u)
            if u == d:
                if path:
                    all_paths.append(path.copy())
            else:
                if triples[u]:
                    neighbors = list(zip(*triples[u]))[1]
                    for i in neighbors:
                        if not visited[i]:
                            search(i, d, visited, path, all_paths)
            path.pop()

        search(src, dst, visited, path, all_paths)
        return [p for p in all_paths if p and len(p) <= self.path_length]


    def find_paths(self, src, dst, triples):
        def search(start, end):
            path, paths = [], []
            queue = [(start, end, path)]
            while queue:
                start, end, path = queue.pop()
                path = path + [start]
                if start == end:
                    paths.append(path)
                neighbors = list(zip(*triples[start]))[1]
                for node in set(neighbors).difference(path):
                    queue.append((node, end, path))
            return paths

        paths = search(src, dst)
        return [p for p in paths if p and len(p) <= self.path_length]
