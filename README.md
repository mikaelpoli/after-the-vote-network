# After the Vote: Mapping Online Trans Discourse
This project was created for the Network Science exam at Unipd.

## Overview
Transgender people face stigma and discrimination, leading to minority stress: a chronic strain linked to negative mental and physical health outcomes ([Hunter et al., 2021](https://doi.org/10.1177/13591045211033187)). It is realistic to think this stress can be intensified by political attacks on trans rights, especially efforts to restrict access to gender-affirming care and fuel the demonization of trans people. Donald Trump’s 2024 Agenda 47 campaign included several such proposals ([President Trump’s Plan to Protect Children From Left-Wing Gender Insanity | Donald J. Trump for President 2024](https://www.donaldjtrump.com/agenda47/president-trumps-plan-to-protect-children-from-left-wing-gender-insanity)).
The aim of the project is to capture a snapshot of online discourse within trans-oriented communities in the months following Trump’s re-election by performing topic detection and sentiment analysis on top posts and comments from Reddit r/trans, r/transgender, and r/asktransgender. Posts span from December 6, 2025, to May 17, 2025.

For a full description of the workflow and results:
- [Report](https://github.com/mikaelpoli/ns-reddit-app/blob/main/report.pdf)
- [PPT (.pdf)](https://github.com/mikaelpoli/ns-reddit-app/blob/main/presentation.pdf)

## Essential Project Repo Structure
[root.]
  - data/
	  - comments/
	  - posts/
	  - df_dd.json
	  - docs_dd.json
	  - docs_dd_giant.json
	  - topic_df.csv
  - src/
  - results/
	  - graphs/
	  - models/
	  - plots/
    - .gitignore
- build_network.ipynb
- fetch_comments.py
- fetch_posts.py
- graph_analysis.ipynb
- preprocess_comments.py
- preprocess_posts.py
- presentation.pdf
- report.pdf
- sentiment_graph.ipynb
- topic_detection_bertopic.ipynb
- topic_detection_leiden.ipynb
- topic_detection_louvain.ipynb
- bertagent_analysis.ipynb

### Running the Code
Use **Python 3.11** for compatibility with spaCy (must install spaCy's English language pipeline `en_core_web_sm`), either in a venv or in Google Colab. If using Colab, the path to the directories and files may need to be reset. The order in which the Python files and notebooks were created is:
1. fetch_posts.py
2. fetch_comments.py
3. preprocess_posts.py
4. preprocess_comments.py
5. build_network.ipynb
6. topic_detection_bertopic.ipynb
7. topic_detection_louvain.ipynb
8. topic_detection_leiden.ipynb
9. bertagent_analysis.ipynb

All intermediate results were saved in the `data/` and `results/` directories and loaded in each file and notebook in the pipeline.

Main files:
- **Documents**: in the `data/` directory.
	- `docs_dd_giant.json` contains the original and POS-tagged documents in the final document-level network.
	- `df_dd.json` contains the original documents as well as their classification into topics by all four methods (BERTopic, BERTopic-reduced, Louvain, Leiden).
- **Graphs**: in the `results/graphs/` directory.
	- g_dd.graphml and g_dd.pickle: document-level graph and matrices
	- topic_graph.graphl: topic-level graph
	The dataframe containing the final metrics (sentiment analysis, agency analysis, node centality) for all topics is in `data/topic_df.csv`.
- **Models**: in the `results/models/` directory in each model's respective folder.
	- **Evaluation metrics**: the `model_results.csv` file.
- **Visualization**: the Gephi project for the topic-level network is in `results/plots/graph/topic-graph-visualization.gephi`.

## Credits
The code for the `CleanText` and `BuildNetwork` classes, as well as the code in `metrics.py` and the `plot_topic_network_louvain()` function were adapted from the Network Science lab classes by Sina Tavakoli.
