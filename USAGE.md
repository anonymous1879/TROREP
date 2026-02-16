# Usage of Data Generation/Analysis Tools
We provide detailed CLI tools for the analyses described in the paper.
These tools are designed to showcase the methodological biases such as score inflation, effects of candidate filtering, and entity-based relevance leakage.

All scripts provide detailed descriptions and argument definitions using the `--help` flag.

## Evaluation
### Bulk TREC Evaluation (`score_trec_eval.py`)
- **Requires:** [trec_eval](https://github.com/usnistgov/trec_eval)
- **Refers to:** Section 3.1: Data & Evaluation
- **Example:** `python analysis/score_trec_eval.py --qrels data/qrels.robust04.txt --ranking data/config-5.run Biased --ranking data/config-6.run Unbiased`

**Description:**
Standardizes the evaluation process, utility to directly compare multiple ranking (run-files) using the metrics from the paper (`MAP`, `MRR`, `nDCG@20`, and `P@20`).

```
Usage: score_trec_eval.py [OPTIONS]

  Given a set of rankings, and the ground-truth file, creates a table of
  effectiveness scores. I.e. it simply runs 'trec_eval' on the given rankings,
  outputting the metrics we report in the paper.

Options:
  --ranking <FILE TEXT>...  Ranking file and name (.run, name)
  --qrels FILE              TREC Ground-Truth file (.txt)  [required]
  --help                    Show this message and exit.
```

## Analysis
### Balancing of Evaluation Candidates
- **Refers to:** _Section 3.1.4: Balancing_
- **Example:** `python analysis/analyze_doc_balancing.py --ranking data/title.bm25-rm3.run --qrels data/qrels.robust04.txt`

**Description:**
Quantifies how pruning the majority relevance class to achieve a 1:1 balancing artificially inflates effectiveness.

```
Usage: analyze_doc_balancing.py [OPTIONS]

  This script takes a given document ranking and shows the effect of the
  discussed balancing method. Outputs the amount of documents pruned per query
  and the inflation of effectiveness scores.

Options:
  --ranking FILE  Ranking file (.run)  [required]
  --qrels FILE    TREC Ground-Truth file (.txt)  [required]
  --help          Show this message and exit.
```

### Biased Entity Filtering/Injection
- **Refers to:** _Section 3.5.3: Biased Filtering_
- **Example:** `python analysis/analyze_illegal_entities.py --initial data/title.bm25-rm3.run --dataset data/entity-linked.robust04.jsonl --entity-ranking data/filtered.entity_ranking.run`

**Description:**
Identifies "illegal" entity usage by cross-referencing the entity ranking against the documents from the initial retrieval step. 
Calculates how many scored entities were actually present in the retrieved documents versus those "injected" from external sources like the Entity QRELs.

```
Usage: analyze_illegal_entities.py [OPTIONS]

  This script takes the initial document ranking, the dataset with entity
  links, and an entity ranking. From this data, derives if any (and which)
  entities are impossible to be derived from the initial ranking, given the
  provided links. I.e. "which entities are illegal, and (possibly) derived
  from the QRELs?". Outputs the following statistics per topic:

  - Num. Retrieved: Number of entities which can be found in the initial
  document retrieval/ranking.
  - Num. Scored: Number of entities scored in the given entity ranking.
  - Num. Illegal: Number of entities scored in the given entity ranking, but
  NOT found in the initial retrieval.
  - Num. Legal: Number of entities scored in the given entity ranking, AND
  found in the initial retrieval.

Options:
  --initial FILE         Initial ranking file (.run)  [required]
  --dataset FILE         Dataset with entity links (.jsonl)  [required]
  --entity-ranking FILE  Entity ranking file (.run)  [required]
  --help                 Show this message and exit.
```

### Entity Prevalence Analysis
- **Refers to:** _Section 4.1: Entity Prevalence Analysis_
- **Example:** `python analysis/analyze_entity_prevalence.py --entity-ranking data/heuristic.entity_ranking.txt`

**Description:**
Computes prevalence and average rank of each entity in the given entity ranking to identify entity distribution across queries.

```
Usage: analyze_entity_prevalence.py [OPTIONS]

  Takes an entity ranking and retrieves the most-prevalent entities, including
  their average rank. That is, the most common entities across queries with
  the average ranks they are encountered at.

Options:
  --entity-ranking FILE  Entity ranking file (.run)  [required]
  --k INTEGER            Threshold to limit prevalence statistics to the top-k
                         entities per query/topic (values < 0 = all)
                         [default: 20]
  --help                 Show this message and exit.
```

### Entity Pool Statistics
- **Refers to:** _Section 4.2: Heuristic Entity Ranking_
- **Example:** `python analysis/analyze_entity_ranking.py --entity-ranking data/filtered.entity_ranking.run`

**Description:**
Generates descriptive statistics for the provided entity ranking, including unique entity counts and entities per query.

```
Usage: analyze_entity_ranking.py [OPTIONS]

  This script takes the entity ranking and outputs statistics on this ranking.
  Shows the num. of queries, num. of total query-entity pairs, num. of unique
  entities, and entities per query/topic.

Options:
  --entity-ranking FILE  Entity ranking file (.run)  [required]
  --help                 Show this message and exit.
```

### Entity Overlap Biased
- **Refers to:** _Section 4.3: Entity Overlap Bias_
- **Example:** `python analysis/analyze_shared_entities.py --dataset data/entity-linked.robust04.jsonl --qrels data/qrels.robust04.txt`

**Description:**
Identifies "shared" entities that appear across both positive and negative document classes.
Showcasing how many, and which, entities are removed from the candidate entities in both QDER and DREQ.

```
Usage: analyze_shared_entities.py [OPTIONS]

  Takes the dataset with entity links and the document-level ground-truth
  file. Mimics the QDER and DREQ Entity QREL creation logic
  ('make_entity_qrels.py') to classify entities as either positive, negative,
  or shared. As identified in *Sec 3.5.1*, this is the first relevance-based
  filtering applied to the entity candidates. Outputs the number of shared,
  positive, and negative entities averaged per query. Additionally, shows the
  most prevalent shared/filtered out entities.

Options:
  --dataset FILE  Dataset with entity links (.jsonl)  [required]
  --qrels FILE    TREC Ground-Truth file (.txt)  [required]
  --help          Show this message and exit.
```

### Candidate Pool Decomposition
- **Refers to:** _Figure 2 & 3: Candidate Pool Decomposition_
- **Example:** `python analysis/analyze_ranking_decomposition.py --qrels data/qrels.robust04.txt  --ranking data/title.bm25-rm3.run Initial --ranking data/title.run DREQ`

**Description:**
Decomposes the documents in the given rankings into the three main document relevance classes: relevant, irrelevant, and unknown.
Allowing for a direct comparison on how the observed pruning in DREQ (and QDER) affect the distribution of these classes.

```
Usage: analyze_ranking_decomposition.py [OPTIONS]

  Given a set of rankings, and the ground-truth file, creates a ranking
  (de)composition plot as done in the paper. Additionally, outputs the mean
  amount of positive, negative, and unknown documents. E.g. --ranking
  <ranking1> <name1> --ranking <ranking2> <name2> allows for direct comparison
  between two rankings.

Options:
  --ranking <FILE TEXT>...  Entity ranking file and name (.run, name)
  --qrels FILE              TREC Ground-Truth file (.txt)  [required]
  --output PATH             Output path for the composition plot.
  --help                    Show this message and exit.
```

### Analysis of DREQ
- **Requires:** The published data of [DREQ](https://github.com/shubham526/ECIR2024-DREQ/wiki/)
- **Refers to:** _Section 5.1: Impact of Filtering_
- **Example:** `python analysis/analyze_dreq_filtering.py --initial data/title.BM25_RM3_TUNED.run --ent-qrels data/entity.qrels --dataset data/robust04.jsonl --save data/filtered_ranking_new.run --reference data/title.run`

**Description:**
Applies the artificial filtering methods identified in DREQ and QDER, applying it directly on the published data of DREQ.
Showcases which topics/queries are omitted, how many documents are filtered out, and the overlap between this filtered and the published ranking.

```
Usage: analyze_dreq_filtering.py [OPTIONS]

  This script takes the initial ranking (BM25+RM3), Entity QRELs, and dataset
  with entity links. Each of these files can be either generated, or taken
  directly from DREQ's published data. By providing a 'reference' ranking, the
  final published ranking of DREQ, it shows the document overlap as discussed
  in the paper.

Options:
  --initial FILE    Initial ranking file (.run)  [required]
  --ent-qrels FILE  Entity Ground-Truth file (.txt)  [required]
  --dataset FILE    Dataset with entity links (.jsonl)  [required]
  --save FILE       Output location where the filtered ranking will be saved
                    (.run)  [required]
  --reference FILE  Reference ranking for computing overlap (.run)
  --help            Show this message and exit.
```
