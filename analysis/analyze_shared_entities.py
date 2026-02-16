import logging
from collections import defaultdict
from statistics import mean

import click
import mmead
import pandas as pd
import pytrec_eval
from tqdm import tqdm

from utils.utils import collect_doc_ent_links, collect_entity_prevalence, mmead_titles_from_ids, log_setup, \
	log_divider, log_stat, log_table

log_setup()


def get_classified_entities(qrels: dict, links: dict) -> dict:
	"""
	Returns, per topic, entities classified as positive, negative, or shared (similar to Entity QRELs creation).
	:param qrels:
	:param links:
	:return:
	"""
	result = defaultdict(dict)
	for topic, doc_rels in tqdm(qrels.items()):
		pos_entities = set()
		neg_entities = set()
		for doc_id, rel in doc_rels.items():
			if rel >= 1:
				pos_entities.update(links.get(str(doc_id)))
			elif rel < 1:
				neg_entities.update(links.get(str(doc_id)))
			else:
				logging.warning(f"Invalid relevance value found: {rel}")
		shared = pos_entities.intersection(neg_entities)
		pos_entities = pos_entities.difference(shared)
		neg_entities = neg_entities.difference(shared)

		result[topic] = {
			"shared": shared,
			"positive": pos_entities,
			"negative": neg_entities
		}
	return result


def get_class_distributions(class_entities: dict):
	"""
	Logs the number of entities per class with the class percentage.
	:param class_entities:
	:return: the average number of shared entities
	"""
	pos_lens = [len(v.get("positive")) for v in class_entities.values()]
	neg_lens = [len(v.get("negative")) for v in class_entities.values()]
	sha_lens = [len(v.get("shared")) for v in class_entities.values()]

	m_pos, m_neg, m_sha = mean(pos_lens), mean(neg_lens), mean(sha_lens)
	total = m_pos + m_neg + m_sha

	log_stat("Mean positive entities", f"{m_pos:.1f} ({m_pos / total * 100:.2f}%)")
	log_stat("Mean negative entities", f"{m_neg:.1f} ({m_neg / total * 100:.2f}%)")
	log_stat("Mean shared entities", f"{m_sha:.1f} ({m_sha / total * 100:.2f}%)")
	return m_sha


@click.command()
@click.option("--dataset", required=True, type=click.Path(exists=True, dir_okay=False),
			  help="Dataset with entity links (.jsonl)")
@click.option("--qrels", required=True, type=click.Path(exists=True, dir_okay=False),
			  help="TREC Ground-Truth file (.txt)")
def main(dataset, qrels):
	"""
	Takes the dataset with entity links and the document-level ground-truth file.
	Mimics the QDER and DREQ Entity QREL creation logic ('make_entity_qrels.py') to classify entities as either positive, negative, or shared.
	As identified in *Sec 3.5.1*, this is the first relevance-based filtering applied to the entity candidates.
	Outputs the number of shared, positive, and negative entities averaged per query.
	Additionally, shows the most prevalent shared/filtered out entities.
	"""
	log_divider("Phase 1: Loading Data")
	mappings = mmead.get_mappings()
	with open(qrels, "r") as f_qrels:
		qrels_dict = pytrec_eval.parse_qrel(f_qrels)
	doc_ents = collect_doc_ent_links(dataset)

	log_divider("Phase 2: Classifying Entities")
	class_entities = get_classified_entities(qrels_dict, doc_ents)
	m_shared = get_class_distributions(class_entities)

	log_divider("Phase 3: Shared (Removed) Entities")
	shared_prevalence = collect_entity_prevalence(class_entities, "shared")
	title_mapping = mmead_titles_from_ids(mappings, list(shared_prevalence.keys()))

	rows = []
	for ent, prev in shared_prevalence.items():
		rows.append((ent, title_mapping.get(ent), prev))
	df = pd.DataFrame(rows, columns=["Entity ID", "Entity Title", "Prevalence"]).set_index("Entity ID").sort_values(
		"Prevalence", ascending=False)
	log_table("Most common 'shared' entities:", df)
	most_prevalent = df['Prevalence'].idxmax()

	log_divider("Summary")
	logging.info("QDER and DREQ filter out any 'shared' entities from the pool of candidates entities.")
	logging.info(f"In this case, on average, {m_shared:.1f} entities are filtered out per query.")
	logging.info(f"The most common 'shared' entity is '{df.loc[most_prevalent]["Entity Title"]}',"
				 f"which is shared for {df.loc[most_prevalent]["Prevalence"]} queries.")


if __name__ == '__main__':
	main()
