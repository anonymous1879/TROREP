import logging

import click
import pandas as pd
import pytrec_eval

from utils.utils import evaluate_trec_dict, collect_ranks_with_stats, log_setup, log_stat, log_table, log_divider, \
	METRIC_MAPPING

log_setup()


def get_balanced_counts(ranking: dict, balanced: dict):
	"""
	Counts number of documents before and after balancing in a ranking.
	:param ranking:
	:param balanced:
	:return:
	"""
	rows = []
	for topic, docs in ranking.items():
		if topic not in balanced:
			continue
		docs_bal = balanced.get(topic)
		rows.append({
			"Query": topic,
			"Num. Documents (Before)": len(docs),
			"Num. Documents (After)": len(docs_bal),
			"Num. Pruned": len(docs) - len(docs_bal)
		})
	return rows

def get_balanced_ranking(ranking: dict, qrels: dict):
	"""
	Implements the balancing logic, partitions documents into positive and negative relevance classes.
	Down-samples/prunes the majority class to create an equal distribution.
	:param ranking:
	:param qrels:
	:return:
	"""
	result = {}
	for topic, docs in ranking.items():
		topic_qrels = qrels.get(topic, {})

		sorted_ids = sorted(docs.keys(), key=lambda x: docs[x], reverse=True)
		pos_docs = [d for d in sorted_ids if topic_qrels.get(d, 0) >= 1]
		neg_docs = [d for d in sorted_ids if topic_qrels.get(d, 0) == 0]

		k = min(len(pos_docs), len(neg_docs))
		if k > 0:
			balanced_ids = set(pos_docs[:k] + neg_docs[:k])
			result[topic] = {doc_id: score for doc_id, score in docs.items() if doc_id in balanced_ids}
		else:
			log_stat(f"Query {topic}", "Skipped (missing positive or negative documents)")
			logging.warning(f"Could not balance query #{topic}: missing positive or negative documents, skipped.")
	return result


@click.command()
@click.option("--ranking", required=True, type=click.Path(exists=True, dir_okay=False),
			  help="Ranking file (.run)")
@click.option("--qrels", required=True, type=click.Path(exists=True, dir_okay=False),
			  help="TREC Ground-Truth file (.txt)")
def main(ranking, qrels):
	"""
	This script takes a given document ranking and shows the effect of the discussed balancing method.
	Outputs the amount of documents pruned per query and the inflation of effectiveness scores.
	"""
	log_divider("Phase 1: Loading Data")
	rank_dict = collect_ranks_with_stats(ranking, element="entities")
	with open(qrels, "r") as f_qrels:
		qrel_dict = pytrec_eval.parse_qrel(f_qrels)

	log_divider("Phase 2: Balancing Ranking")
	rank_balanced = get_balanced_ranking(rank_dict, qrel_dict)
	df = pd.DataFrame(get_balanced_counts(rank_dict, rank_balanced)).set_index("Query").sort_values("Num. Pruned", ascending=False)
	mean_pruned = round(df["Num. Pruned"].mean(), 1)
	log_table("Documents before and after balancing", df)
	log_stat("Mean documents pruned", mean_pruned)

	log_divider("Phase 3: Evaluation")
	df_unb = evaluate_trec_dict(rank_dict, qrels)
	df_unb.index = ["Original"]
	df_bal = evaluate_trec_dict(rank_balanced, qrels)
	df_bal.index = ["Balanced"]

	df_final = pd.concat([df_unb, df_bal]).sort_values("map", ascending=False).rename(columns=METRIC_MAPPING)
	log_table("Effectiveness before and after balancing", df_final)

	log_divider("Summary")
	logging.info("To summarize, if balancing is used on the evaluation candidates:")
	logging.info(f"Prunes out, on average, {mean_pruned} documents per query (min={df["Num. Pruned"].min()}, max={df["Num. Pruned"].max()}).")
	logging.info(f"And causes an artificial inflation of MAP from {df_final.loc['Original']['MAP']} to {df_final.loc['Balanced']['MAP']}, by pruning the majority relevance class.")

if __name__ == '__main__':
	main()
