import logging

import click
import pandas as pd

from utils.utils import collect_doc_ent_links, collect_ranks_with_stats, log_setup, log_divider, log_table, log_stat

log_setup()


def validate_topic_entities(links: dict, i_dict: dict, e_dict: dict, topic: str | int) -> dict:
	"""
	Computes the theoretical, scored, illegal, and valid entities for the given topic, with the following meanings:
	- Retrieved: Entities which can be retrieved from the initial ranking, the entity pool (Ɛ) defined by QDER and DREQ
	- Scored: Entities scored in the given entity ranking
	- Illegal: Entities which are scored, but are not in the theoretical pool
	- Legal: Entities which are scored, and are retrieved by the given initial ranking
	:param links:
	:param i_dict:
	:param e_dict:
	:param topic:
	:return:
	"""
	result = dict()

	topic_str = str(topic)
	if topic_str not in i_dict or topic_str not in e_dict:
		logging.warning(f"Topic #{topic} missing from either the initial or entity ranking, skipping.")
		return result

	# We first collect all 'valid' entities, entities found in documents retrieved in the initial retrieval step:
	valid_entities = set()
	docs = i_dict.get(topic_str).keys()
	for doc_id in docs:
		ents_in_doc = {str(e) for e in links.get(str(doc_id), [])}
		valid_entities.update(ents_in_doc)

	# We then collect all entities scored in the entity ranking:
	scored_entities = set(e_dict.get(topic_str).keys())

	# Write stats for the given topic
	result["Topic"] = topic_str
	result["Num. Retrieved"] = len(valid_entities)
	result["Num. Scored"] = len(scored_entities)
	result["Num. Illegal"] = len(scored_entities.difference(valid_entities))
	result["Num. Legal"] = len(scored_entities & valid_entities)
	return result


@click.command()
@click.option("--initial", required=True, type=click.Path(exists=True, dir_okay=False),
			  help="Initial ranking file (.run)")
@click.option("--dataset", required=True, type=click.Path(exists=True, dir_okay=False),
			  help="Dataset with entity links (.jsonl)")
@click.option("--entity-ranking", required=True, type=click.Path(exists=True, dir_okay=False),
			  help="Entity ranking file (.run)")
def main(initial, dataset, entity_ranking):
	"""
	This script takes the initial document ranking, the dataset with entity links, and an entity ranking.
	From this data, derives if any (and which) entities are impossible to be derived from the initial ranking, given the provided links.
	I.e. "which entities are illegal, and (possibly) derived from the QRELs?".
	Outputs the following statistics per topic:\n
	- Num. Retrieved: Number of entities which can be found in the initial document retrieval/ranking.\n
	- Num. Scored: Number of entities scored in the given entity ranking.\n
	- Num. Illegal: Number of entities scored in the given entity ranking, but NOT found in the initial retrieval.\n
	- Num. Legal: Number of entities scored in the given entity ranking, AND found in the initial retrieval.\n
	"""
	log_divider("Phase 1: Loading Rankings")
	doc_ents = collect_doc_ent_links(dataset)
	doc_rank = collect_ranks_with_stats(initial, name="initial", element="documents")
	ent_rank = collect_ranks_with_stats(entity_ranking, name="entity", element="entities")

	rows = []
	for topic in doc_rank.keys():
		topic_res = validate_topic_entities(doc_ents, doc_rank, ent_rank, topic)
		if topic_res:
			rows.append(topic_res)
	df = pd.DataFrame(rows).set_index("Topic").sort_values("Num. Illegal", ascending=False)
	log_divider("Phase 2: Entity Subsets")
	log_table("Entity statistics", df)
	log_stat("Mean entities (retrieved)", f"{df["Num. Retrieved"].mean():.2f}")
	log_stat("Mean entities (scored)", f"{df["Num. Scored"].mean():.2f}")
	log_stat("Mean entities (illegal)", f"{df["Num. Illegal"].mean():.2f}")
	log_stat("Mean entities (legal)", f"{df["Num. Legal"].mean():.2f}")

	log_divider("Summary")
	logging.info(
		f"There are {df["Num. Scored"].sum()} scored query-entity pairs in the entity ranking ({entity_ranking}).")
	logging.info(f"Of which, {df["Num. Legal"].sum()} can be derived from the initial ranking ({initial}).")
	logging.info(
		f"Where, {df["Num. Illegal"].sum()} are 'illegal' and have been injected from some other source, such as the Entity QRELs.")


if __name__ == '__main__':
	main()
