import click
import mmead
import pandas as pd

from utils.utils import collect_entity_rank_prevalence, mmead_titles_from_ids, log_setup, \
	log_divider, log_table, collect_ranks_with_stats, log_stat

log_setup()


@click.command()
@click.option("--entity-ranking", required=True, type=click.Path(exists=True, dir_okay=False),
			  help="Entity ranking file (.run)")
@click.option("--k", required=False, type=int, default=20, show_default=True,
			  help="Threshold to limit prevalence statistics to the top-k entities per query/topic (values < 0 = all)")
def main(entity_ranking, k):
	"""
	Takes an entity ranking and retrieves the most-prevalent entities, including their average rank.
	That is, the most common entities across queries with the average ranks they are encountered at.
	"""
	log_divider("Phase 1: Loading Data")
	if k < 0: k = None
	mapping = mmead.get_mappings()
	ranking = collect_ranks_with_stats(entity_ranking, "entities")
	log_stat("Limiting prevalence to", "All" if k is None else f"Top-{k}")

	log_divider("Phase 2: Entity Prevalence")
	ent_ranks, ent_prevs = collect_entity_rank_prevalence(ranking, k=k)

	ent_ids = set(ent_ranks.keys())
	ent_titles = mmead_titles_from_ids(mapping, list(ent_ids))

	rows = []
	for eid in ent_ids:
		# Get prevalence, average rank, and title of each entity
		prev = ent_prevs.get(eid)
		rank = ent_ranks.get(eid)
		title = ent_titles.get(int(eid))
		rows.append((eid, title, prev, rank))

	prev_title = "Prevalence" if k is None else f"Prevalence in Top-{k}"
	df = pd.DataFrame(rows, columns=["Entity ID", "Entity Title", prev_title, "Avg. Rank"]).set_index(
		"Entity ID").sort_values([prev_title, "Avg. Rank"], ascending=[False, True]).round(2)
	log_table("Entities from given ranking sorted by prevalence", df)


if __name__ == '__main__':
	main()
