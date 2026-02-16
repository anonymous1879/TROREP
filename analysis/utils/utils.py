"""
Collects all entities linked per document.
"""
import json
import logging
import os
import subprocess
import tempfile
from collections import defaultdict
from io import StringIO
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
import pytrec_eval
from mmead.data.mappings import Mapping
from tqdm import tqdm

DIVIDER_WIDTH: int = 25

# =============================================================================
# 							DATA COLLECTION
# =============================================================================

def collect_doc_ent_links(dataset: str) -> dict[str, set]:
	"""
	Collects and returns the documents by IDs with the set of entities linked within for the given dataset.
	:param dataset:
	:return: dictionary of <doc_id: set(entities)> for each document and entity links in the given dataset
	"""
	doc_ents = defaultdict(set)
	with open(dataset, "r") as f_corpus:
		for line in tqdm(f_corpus):
			obj: dict = json.loads(line)
			doc_id: str = obj.get("doc_id")
			ents: list = obj.get("entities")
			# Check types
			assert type(doc_id) == str
			assert type(ents) == list
			doc_ents[doc_id] = set(ents)
	logging.info(f"Processed {len(doc_ents.keys())} documents in the dataset.")
	return doc_ents

def collect_ranks_with_stats(file: str, name: str="given", element="elements") -> dict:
	"""
	Collects the ranking data and logs the amount of queries and elements (documents or entities) in the rankings.
	:param file: ranking file (.run)
	:param name: name when logging statistics (default: 'given')
	:param element: element name (document / entity)
	:return: the rankings as a dict
	"""
	with open(file, "r") as f_in:
		rank_dict = pytrec_eval.parse_run(f_in)
	ranked_docs = [len(list(docs.keys())) for qid, docs in rank_dict.items()]

	log_stat(f"Queries in {name} ranking", len(rank_dict.keys()))
	log_stat(f"{element.title()} in {name} ranking (rows)", sum(ranked_docs))
	return rank_dict

def collect_unique_elements(ranking: dict, topic="all", element="elements") -> set:
	"""
	Collects all unqiue elements in the given ranking for the given (or all) topics.
	:param ranking: either document or entity ranking gained from pytrec_eval
	:param topic: target topic, or 'all'
	:return:
	"""
	if topic == "all":
		result = set()
		for t, e_dict in ranking.items():
			result.update(e_dict.keys())
		log_stat(f"Unique {element} (Total)", len(result))
		return result
	elif str(topic) in ranking.keys():
		result = set(ranking.get(str(topic)).keys())
		log_stat(f"Unique {element} (#{topic})", len(result))
		return result
	else:
		logging.error(f"Could not find given query in ranking: {topic}")
		return set()

def collect_entity_prevalence(ents_per_topic: dict, class_name: str | None=None):
	"""
	Collects query-wide prevalence of each entity
	:param ents_per_topic:
	:param class_name:
	:return:
	"""
	topic_count = len(ents_per_topic.keys())
	ent_counts = defaultdict(int)
	for topic, classes in ents_per_topic.items():
		if class_name is not None:
			for ent in set(classes.get(class_name)):
				ent_counts[ent] += 1
		else:
			for ent in set(classes):
				ent_counts[ent] += 1
	ent_counts_avg = {k: v / topic_count for k, v in ent_counts.items()}
	return ent_counts_avg

def collect_entity_rank_prevalence(ents_per_topic: dict, k: int | None=None) -> tuple[dict, dict]:
	"""
	Collects the average rank per entity over all queries and their prevalence.
	:param ents_per_topic:
	:param k: limit prevalence calculation to top-k ('None' for no limit)
	:return:
	"""
	ent_ranks = defaultdict(list)
	ent_prevs = defaultdict(int)
	topic_count = len(ents_per_topic.keys())
	for topic_ents in ents_per_topic.values():
		sorted_ents = sorted(topic_ents.items(), key=lambda x: x[1], reverse=True)
		for rank, (eid, score) in enumerate(sorted_ents, start=1):
			ent_ranks[eid].append(rank)
			if k is None or rank <= k:
				ent_prevs[eid] += 1

	avg_ranks = defaultdict(float)
	for eid, ranks in ent_ranks.items():
		avg_ranks[eid] = mean(ranks)
	avg_prevs = {k: v / topic_count for k, v in ent_prevs.items()}
	return avg_ranks, avg_prevs

def collect_decomposed_ranking(doc_ranking: dict, qrels: dict, name: str="given"):
	topics = set(doc_ranking.keys())
	docs_per_rel = defaultdict(list)
	for topic in topics:
		if topic not in qrels:
			logging.warning(f"Topic #{topic} missing from QRELs, skipping.")
			continue

		qrel_dict: dict = qrels.get(topic)
		docs = doc_ranking.get(topic).keys()

		positive, negative, unknown = [], [], []
		for doc_id in docs:
			rel = qrel_dict.get(doc_id, None)
			if rel is None:
				unknown.append(doc_id)
			elif rel >= 1:
				positive.append(doc_id)
			else:
				negative.append(doc_id)

		docs_per_rel["positive"].append(len(positive))
		docs_per_rel["negative"].append(len(negative))
		docs_per_rel["unknown"].append(len(unknown))
	return docs_per_rel


# =============================================================================
# 								 MMEAD
# =============================================================================

def mmead_titles_from_ids(mapping: Mapping, identifiers: list) -> dict:
	"""
	Bulk operations of the MMEAD 'identifier -> entity title' mapping.
	:param mapping: MMEAD Mapping instance
	:param identifiers: list of entity identifiers
	:return: dictionary of entity ids to their respective titles (None if unknown)
	"""
	identifiers = np.array(identifiers)
	mapping.cursor.register("identifiers", {
		"id": identifiers,
		"ord": np.arange(identifiers.shape[0])
	})
	mapping.cursor.execute(f"""
		SELECT m.id AS eid, m.entity AS title
		FROM entity_id_mapping m, identifiers i
		WHERE m.id = i.id
		ORDER BY i.ord
	""")
	ent_dict = mapping.cursor.fetchdf().set_index("eid")["title"].to_dict()
	mapping.cursor.unregister("identifiers")
	return {k: v for k, v in ent_dict.items()}


# =============================================================================
# 								LOGGING
# =============================================================================

def log_setup():
	logging.basicConfig(level=logging.INFO)

def log_divider(title: str=None):
	if title:
		logging.info('---------- ' + title + ' ----------')
	else:
		logging.info('-' * DIVIDER_WIDTH)

def log_stat(title: str, value):
	logging.info(f" - {title + ':':<40} {value}")

def log_table(title: str, df: pd.DataFrame):
	logging.info(f" - {title}:\n{df}")


# =============================================================================
# 								EVALUATION
# =============================================================================

METRIC_MAPPING = {
	"map": "MAP",
	"ndcg_cut_20": "nDCG@20",
	"P_20": "P@20",
	"recip_rank": "MRR"
}

def evaluate_trec(run: Path, qrels: Path) -> pd.DataFrame:
	"""
	Scores the given ranking (.run) using 'trec_eval' based on the given QRELs.
	:param run:
	:param qrels:
	:return:
	"""
	res = subprocess.run([
		"trec_eval", str(qrels), str(run),
		"-m", "map", "-m", "ndcg_cut.20", "-m", "P.20", "-m", "recip_rank",
	], capture_output=True, text=True)

	# Convert output directly to DataFrame
	df = pd.read_csv(
		StringIO(res.stdout),
		sep=r'[\t ]+',
		header=None,
		names=["metric", "scope", "value"],
		engine="python"
	)
	row = df.set_index("metric")["value"].astype(float).to_frame().T
	row["version"] = run
	row.set_index("version", inplace=True)
	return row

def evaluate_trec_dict(ranking: dict, qrels: Path):
	"""
	Scores the given ranking dictionary using 'trec_eval' based on the given QRELs.
	Temporary creates a physical ranking file (.run) to score using 'trec_eval'.
	:param ranking:
	:param qrels:
	:return:
	"""
	with tempfile.TemporaryDirectory() as tmp_dir:
		tmp_run = os.path.join(tmp_dir, "temp_rank.run")

		with open(tmp_run, "w") as f_run:
			for topic, docs in ranking.items():
				sorted_docs = sorted(docs.items(), key=lambda x: x[1], reverse=True)
				for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
					f_run.write(f"{topic} Q0 {doc_id} {rank} {score} TEMP\n")
			log_stat("Temp. ranking stored to", tmp_run)
			return evaluate_trec(Path(tmp_run), qrels)