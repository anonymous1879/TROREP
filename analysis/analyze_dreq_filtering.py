from itertools import groupby
from statistics import mean

import click
import pytrec_eval

from utils.utils import collect_doc_ent_links, collect_ranks_with_stats, log_setup, log_divider, log_stat

log_setup()


def collect_filtered_eqrels(entity_qrels: str) -> dict:
	"""
	Collects all entities from the Entity QRELs for each query.
	Applies the mandatory balancing found in both QDER and DREQ.
	This is the **second** artificial filtering.
	:param entity_qrels: Entity QRELs
	:return:
	"""
	with open(entity_qrels, "r") as f_eqrels:
		eqrels_dict = pytrec_eval.parse_qrel(f_eqrels)

	filtered_eqrels = {}
	for qid, ents in eqrels_dict.items():
		# Create subsets
		pos = [ent_id for ent_id, score in ents.items() if score >= 1]
		neg = [ent_id for ent_id, score in ents.items() if score == 0]

		# Balance 1:1
		k = min(len(pos), len(neg))
		if k > 0:
			balanced = {ent_id: 1 for ent_id in pos[:k]}
			balanced.update({ent_id: 0 for ent_id in neg[:k]})
			filtered_eqrels[qid] = balanced
	lost = set(eqrels_dict.keys()) - set(filtered_eqrels.keys())

	log_divider("Phase 1: Candidate Entities Balancing")
	log_stat("Input queries", len(eqrels_dict.keys()))
	log_stat("Queries after balancing", len(filtered_eqrels.keys()))
	log_stat("Queries lost", lost if lost else None)
	return filtered_eqrels


def filter_ranking(ent_qrels: dict, ranking: str, doc_ents: dict, save: str):
	"""
	Filters the given ranking based on scored entity presence.
	Given the filtered Entity QRELs, if a document has NO **scored** entities, the document is filtered out.
	This is the **third** filtering we describe.
	:param ent_qrels: Entity QRELs
	:param ranking: Initial ranking
	:param doc_ents: Dictionary of document -> entities linked within
	:param save: Optionally store the filtered file
	:return:
	"""
	with open(ranking, "r") as f_initial, open(save, "w") as f_out:
		doc_count: int = 0

		# Iterate over the ranking for each query
		for qid, group in groupby(f_initial, key=lambda x: x.split(' ')[0]):
			rank: int = 1
			scored_ents = set(ent_qrels.get(qid).keys()) if qid in ent_qrels else set()
			for line in group:
				doc_count += 1
				parts = line.split(' ')
				doc_id: str = parts[2]
				linked_ents = doc_ents.get(doc_id, set())

				# Get the intersection of the set of scored entities and the set of linked entities
				# If a document has NO scored entities, we filter out the document
				overlap = scored_ents & linked_ents
				if len(overlap) > 0:
					# Artificial rank increase: document retains it original score but receives a higher rank due to pruning
					f_out.write(f"{qid} {parts[1]} {doc_id} {rank} {parts[4]} Entity-Filtered\n")
					rank += 1
		log_divider("Phase 2: Document Filtering")
		log_stat("Total documents scanned", doc_count)


def get_statistics(output: str, reference: str):
	"""
	Computes statistics between the output (filtered) ranking (𝐷𝑓), and a given reference ranking (𝐷𝑝).
	Also computes the overlap:
		 𝑂𝑣𝑒𝑟𝑙𝑎𝑝 (𝐷𝑓) =
		|𝐷𝑓 ∩ 𝐷𝑝| / |𝐷𝑝|
	:param output: Output/filtered ranking
	:param reference: Reference ranking to compare to
	:return:
	"""
	fil_dict = collect_ranks_with_stats(output, "filtered", element="documents")

	if reference:
		ref_dict = collect_ranks_with_stats(reference, "reference", element="documents")
		comparable_queries = fil_dict.keys() & ref_dict.keys()

		overlaps = []
		for qid in comparable_queries:
			fil_docs = fil_dict.get(qid).keys()
			ref_docs = ref_dict.get(qid).keys()
			shared_docs = fil_docs & ref_docs

			overlap = len(shared_docs) / len(ref_docs)
			overlaps.append(overlap)
		log_divider("Phase 3: Overlap Analysis")
		log_stat("Comparable queries", len(comparable_queries))
		log_stat("Mean Overlap (|𝐷𝑓 ∩ 𝐷𝑝| / |𝐷𝑝|)", f"{mean(overlaps) * 100:.2f}%")
		log_stat("Overlap range (min, max)", f"[{min(overlaps) * 100:.2f}%, {max(overlaps) * 100:.2f}%]")


@click.command()
@click.option("--initial", required=True, type=click.Path(exists=True, dir_okay=False),
			  help="Initial ranking file (.run)")
@click.option("--ent-qrels", required=True, type=click.Path(exists=True, dir_okay=False),
			  help="Entity Ground-Truth file (.txt)")
@click.option("--dataset", required=True, type=click.Path(exists=True, dir_okay=False),
			  help="Dataset with entity links (.jsonl)")
@click.option("--save", required=True, type=click.Path(exists=False, dir_okay=False),
			  help="Output location where the filtered ranking will be saved (.run)")
@click.option("--reference", required=False, type=click.Path(exists=True, dir_okay=False),
			  help="Reference ranking for computing overlap (.run)")
def main(initial, ent_qrels, dataset, save, reference):
	"""
	This script takes the initial ranking (BM25+RM3), Entity QRELs, and dataset with entity links.
	Each of these files can be either generated, or taken directly from DREQ's published data.
	By providing a 'reference' ranking, the final published ranking of DREQ, it shows the document overlap as discussed in the paper.
	"""
	filtered_eqrels = collect_filtered_eqrels(ent_qrels)
	doc_ents = collect_doc_ent_links(dataset)
	filter_ranking(filtered_eqrels, initial, doc_ents, save)
	get_statistics(save, reference)


if __name__ == '__main__':
	main()
