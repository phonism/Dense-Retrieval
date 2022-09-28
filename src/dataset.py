"""
dataset
"""

import copy

class Collections(object):
    """
    collection
    """
    def __init__(self):
        self.path = "./msmarco/collection.tsv"
        self.collection_dict = {}
        self._load()

    def _load(self):
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                line_sp = line.split("\t")
                docid = line_sp[0]
                collection = line_sp[1]
                self.collection_dict[docid] = collection

    def __getitem__(self, collectionid):
        return self.collection_dict[collectionid]

class Queries(object):
    """
    queries
    """
    def __init__(self, name="train"):
        self.path = "./msmarco/queries." + name + ".tsv"
        self.query_dict = {}
        self._load()

    def _load(self):
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                line_sp = line.split("\t")
                qid = line_sp[0]
                query = line_sp[1].strip()
                self.query_dict[qid] = query

    def __getitem__(self, qid):
        return self.query_dict[qid]

class Triples(object):
    """
    triples
    """
    def __init__(self):
        self.path = "./msmarco/qidpidtriples.train.full.tsv"
        # self.path = "./msmarco/qidpidtriples.train.small.tsv"
        self.triple = []
        self._load()

    def _load(self):
        with open(self.path) as f:
            for line in f:
                line = line.strip("\n\r")
                line_sp = line.split("\t")
                if len(line_sp) != 3:
                    continue
                qid = line_sp[0]
                posid = line_sp[1]
                negid = line_sp[2]
                self.triple.append([qid, posid, negid])

    def __getitem__(self, idx):
        return self.triple[idx]
    
    def __len__(self):
        return len(self.triple)

    def __iter__(self):
        return self.triple.__iter__()

class SmallTriples(object):
    """
    small triples
    """
    def __init__(self, limit=None):
        self.path = "./msmarco/triples.train.small.tsv"
        self.limit = limit
        self.triple = []
        self._load()

    def _load(self):
        step = 0
        with open(self.path) as f:
            for line in f:
                line = line.strip("\n\r")
                line_sp = line.split("\t")
                if len(line_sp) != 3:
                    continue
                query = line_sp[0]
                pos = line_sp[1]
                neg = line_sp[2]
                self.triple.append([query, pos, neg])
                step += 1
                if step == self.limit:
                    break

    def __getitem__(self, idx):
        return self.triple[idx]
    
    def __len__(self):
        return len(self.triple)

    def __iter__(self):
        return self.triple.__iter__()

class EvalQueries(object):
    """
    small triples
    """
    def __init__(self, limit=None):
        self.path = "./msmarco/qrels.dev.tsv"
        self.limit = limit
        self.dedup = {}
        self.eval_queries = []
        self._load()

    def _load(self):
        step = 0
        with open(self.path) as f:
            for line in f:
                line = line.strip("\n\r")
                line_sp = line.split("\t")
                qid = line_sp[0]
                if qid in self.dedup:
                    continue
                self.dedup[qid] = 1
                self.eval_queries.append(qid)

    def __len__(self):
        return len(self.eval_queries)

    def __getitem__(self, idx):
        return self.eval_queries[idx]

    def __iter__(self):
        return self.eval_queries.__iter__()

class DataSet(object):
    """
    dataset
    """
    def __init__(self, tokenizer, device, rank=0, limit=None):
        self.tokenizer = tokenizer
        self.device = device
        self.rank = rank
        self.nranks = 4
        #self.queries = Queries()
        #print("load queries done!")
        #self.collections = Collections()
        #print("load collection done!")
        self.triples = SmallTriples(limit=limit)
        print("load triples done in rank:", self.rank)

        self.batch_size = 64

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        start_pos = idx * self.batch_size * self.nranks
        end_pos = min((idx + 1) * self.batch_size * self.nranks, len(self.triples))

        if start_pos >= len(self.triples):
            raise StopIteration

        all_queries, all_pos, all_neg = [], [], []

        for position in range(start_pos, end_pos):
            if position % self.nranks != self.rank:
                continue
            query, pos, neg = copy.deepcopy(self.triples[position])

            all_queries.append(query)
            all_pos.append(pos)
            all_neg.append(neg)
        query_input = self.tokenizer(
                all_queries, return_tensors="pt", max_length=32, padding="max_length", truncation="longest_first")
        pos_input = self.tokenizer(
                all_pos, return_tensors="pt", max_length=256, padding="max_length", truncation="longest_first")
        neg_input = self.tokenizer(
                all_neg, return_tensors="pt", max_length=256, padding="max_length", truncation="longest_first")
        return [query_input, pos_input, neg_input]
    

class EvalDataSet(object):
    """
    dataset
    """
    def __init__(self, tokenizer, device, rank=0, limit=None):
        self.tokenizer = tokenizer
        self.device = device
        self.rank = rank
        self.eval_queries = EvalQueries(limit=limit)
        print("load eval_queries done in rank:", self.rank)
        self.queries_dict = Queries(name="dev")
        print("load queries done in rank:", self.rank)

        self.batch_size = 128

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.eval_queries)

    def __getitem__(self, idx):
        start_pos = idx * self.batch_size
        end_pos = min((idx + 1) * self.batch_size, len(self.eval_queries))

        if start_pos >= len(self.eval_queries):
            raise StopIteration

        all_queries = []
        all_qid = []

        for position in range(start_pos, end_pos):
            qid = self.eval_queries[position]
            query = copy.deepcopy(self.queries_dict[qid])

            all_queries.append(query)
            all_qid.append(qid)
        query_input = self.tokenizer(
                all_queries, return_tensors="pt", max_length=32, padding="max_length", truncation="longest_first")
        return query_input, all_qid

class CollectionDataSet(object):
    """
    dataset
    """
    def __init__(self, tokenizer, device, rank=0, limit=None):
        self.tokenizer = tokenizer
        self.device = device
        self.rank = rank

        self.path = "./msmarco/collection.tsv"
        self.collection_ids = []
        self.collection_dict = []
        self._load()
        print("load collections done have ", len(self.collection_ids), " collections!")

        self.batch_size = 256

    def _load(self):
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                line_sp = line.split("\t")
                docid = line_sp[0]
                collection = line_sp[1]
                self.collection_ids.append(docid)
                self.collection_dict.append(collection)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.collection_ids)

    def __getitem__(self, idx):
        start_pos = idx * self.batch_size
        end_pos = min((idx + 1) * self.batch_size, len(self.collection_ids))

        if start_pos >= len(self.collection_ids):
            raise StopIteration

        all_docs = []
        all_docid = []

        for position in range(start_pos, end_pos):
            docid = self.collection_ids[position]
            doc = self.collection_dict[position]

            all_docs.append(doc)
            all_docid.append(docid)
        doc_input = self.tokenizer(
                all_docs, return_tensors="pt", max_length=256, padding="max_length", truncation="longest_first")
        return doc_input, all_docid
