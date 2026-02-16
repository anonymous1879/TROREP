import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean

import click
import matplotlib.pyplot as plt
import pytrec_eval

from utils.utils import collect_ranks_with_stats, collect_decomposed_ranking, log_setup, log_stat, log_divider

log_setup()


def log_decomposition(docs_per_class: dict, class_name="positive"):
	counts = docs_per_class.get(class_name)
	log_stat(f"Mean num. {class_name} documents", f"{mean(counts):.2f} (min={min(counts)}, max={max(counts)})")


def plot_decompositions(rank_names: list[str], rank_data: dict[str, list], output: str):
	"""
	Creates a bar plot of the given ranking compositions
	:param rank_names:
	:param rank_data:
	:param output:
	:return:
	"""
	plt.rcParams.update({'font.size': 12})
	fig, ax = plt.subplots(figsize=(len(rank_names) * 2 + 2, 6))
	x_pos = range(len(rank_names))

	# Configure grid/style
	ax.set_axisbelow(True)
	ax.grid(axis="y", linestyle="--", alpha=0.6)
	cmap = plt.get_cmap('Set2')

	# Plot the different classification layers
	bottom = [0] * len(rank_names)
	for i, (label, values) in enumerate(rank_data.items()):
		ax.bar(x_pos, values, 0.5, label=label, bottom=bottom, color=cmap(i), edgecolor="black")
		bottom = [b + v for b, v in zip(bottom, values)]

	ax.set_xticks(x_pos)
	ax.set_xticklabels(rank_names)

	# Styling
	ax.set_ylabel("Avg. Number of Documents per Topic")
	ax.set_title("Ranking Composition Comparison")
	ax.legend(loc="upper right", frameon=True)
	plt.tight_layout()

	Path(output).parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output, dpi=300)


@click.command()
@click.option("--ranking", multiple=True, nargs=2, type=(click.Path(exists=True, dir_okay=False), str),
			  help="Entity ranking file and name (.run, name)")
@click.option("--qrels", required=True, type=click.Path(exists=True, dir_okay=False),
			  help="TREC Ground-Truth file (.txt)")
@click.option("--output", default="results/decomposition_plot.png", type=click.Path(),
			  help="Output path for the composition plot.")
def main(ranking, qrels, output):
	"""
	Given a set of rankings, and the ground-truth file, creates a ranking (de)composition plot as done in the paper.
	Additionally, outputs the mean amount of positive, negative, and unknown documents.
	E.g. --ranking <ranking1> <name1> --ranking <ranking2> <name2> allows for direct comparison between two rankings.
	"""
	log_divider("Phase 1: Loading Data")
	with open(qrels, "r") as f_qrels:
		qrel_dict = pytrec_eval.parse_qrel(f_qrels)

	log_divider("Phase 2: Decomposing Rankings")
	rank_names = []
	rank_data = defaultdict(list)
	for idx, (path, name) in enumerate(ranking):
		log_divider(f"{name} Ranking")
		log_stat("Loading ranking", path)
		try:
			rank_dict = collect_ranks_with_stats(path, name=name, element="documents")
			docs_per_class = collect_decomposed_ranking(rank_dict, qrel_dict, name=name)
		except ValueError:
			logging.warning(f"Could not parse ranking: {name} ({path}), skipping.")
			continue
		log_decomposition(docs_per_class, class_name="positive")
		log_decomposition(docs_per_class, class_name="negative")
		log_decomposition(docs_per_class, class_name="unknown")

		rank_names.append(f"{name}\n({path})")
		rank_data["Positive"].append(mean(docs_per_class.get("positive")))
		rank_data["Negative"].append(mean(docs_per_class.get("negative")))
		rank_data["Unknown"].append(mean(docs_per_class.get("unknown")))

	log_divider("Phase 3: Plotting Compositions")
	plot_decompositions(rank_names, rank_data, output)
	log_stat("Saved decomposition plot", output)
	click.launch(output)


if __name__ == '__main__':
	main()
