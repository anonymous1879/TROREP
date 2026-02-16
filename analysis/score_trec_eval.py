import click
import pandas as pd

from utils.utils import evaluate_trec, log_divider, log_setup, log_stat, log_table, METRIC_MAPPING

log_setup()

@click.command()
@click.option("--ranking", multiple=True, nargs=2, type=(click.Path(exists=True, dir_okay=False), str),
			  help="Ranking file and name (.run, name)")
@click.option("--qrels", required=True, type=click.Path(exists=True, dir_okay=False),
			  help="TREC Ground-Truth file (.txt)")
def main(ranking, qrels):
	"""
	Given a set of rankings, and the ground-truth file, creates a table of effectiveness scores.
	I.e. it simply runs 'trec_eval' on the given rankings, outputting the metrics we report in the paper.
	"""
	log_divider("Phase 1: Scoring Ranking(s)")
	results = []
	for rank, name in ranking:
		log_stat("Scoring ranking", name)
		df = evaluate_trec(rank, qrels)
		df.index = [name]
		results.append(df)
	df_final = pd.concat(results).sort_values("map", ascending=False).rename(columns=METRIC_MAPPING)
	log_table("Effectiveness per ranking", df_final)

if __name__ == '__main__':
	main()
