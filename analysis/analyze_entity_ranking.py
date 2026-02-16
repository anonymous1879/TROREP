import click
import pandas as pd

from utils.utils import collect_unique_elements, collect_ranks_with_stats, log_setup, log_stat, log_table

log_setup()


def get_entities_per_topic(ranking: dict):
	"""
	Gets the average number of entities per query
	:param ranking:
	:return:
	"""
	rows = []
	for topic, ranked in ranking.items():
		rows.append((topic, len(ranked.keys())))
	df = pd.DataFrame(rows, columns=["Query", "Num. Entities"]).sort_values("Num. Entities", ascending=False)
	avg = df["Num. Entities"].mean()

	log_table("Entities per query", df)
	log_stat("Mean entities per query", f"{avg:.2f}")


@click.command()
@click.option("--entity-ranking", required=True, type=click.Path(exists=True, dir_okay=False),
			  help="Entity ranking file (.run)")
def main(entity_ranking):
	"""
	This script takes the entity ranking and outputs statistics on this ranking.
	Shows the num. of queries, num. of total query-entity pairs, num. of unique entities, and entities per query/topic.
	"""
	erank_dict = collect_ranks_with_stats(entity_ranking, element="entities")
	_ = collect_unique_elements(erank_dict, element="entities")
	get_entities_per_topic(erank_dict)


if __name__ == '__main__':
	main()
