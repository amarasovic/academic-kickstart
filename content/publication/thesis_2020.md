+++
title = "Deep Learning With Sentiment Inference For Discourse-Oriented Opinion Analysis"

# Date first published.
date = "2018-11-31"

# Authors. Comma separated list, e.g. `["Bob Smith", "David Jones"]`.
authors = ["Marasović Ana"]

# Publication type.
# Legend:
# 0 = Uncategorized
# 1 = Conference proceedings
# 2 = Journal
# 3 = Work in progress
# 4 = Technical report
# 5 = Book
# 6 = Book chapter
publication_types = ["0"]

# Publication name and optional abbreviated version.
#publication = "In *SNAMS*"
#publication_short = "In *SNAMS*"

# Abstract and optional shortened version.
abstract = "Opinions are omnipresent in written and spoken text ranging from editorials, reviews, blogs, guides, and informal conversations to written and broadcast news. However, past research in NLP has mainly addressed explicit opinion expressions, ignoring implicit opinions. As a result, research in opinion analysis has plateaued at a somewhat superficial level, providing methods that only recognize what is explicitly said and do not understand what is implied. In this dissertation, we develop machine learning models for two tasks that presumably support propagation of sentiment in discourse, beyond one sentence. The first task we address is opinion role labeling, i.e.\ the task of detecting who expressed a given attitude toward what or who. The second task is abstract anaphora resolution, i.e.\ the task of finding a (typically) non-nominal antecedent of pronouns and noun phrases that refer to abstract objects like facts, events, actions, or situations in the preceding discourse. We propose a neural model for labeling of opinion holders and targets and circumvent the problems that arise from the limited labeled data. In particular, we extend the baseline model with different multi-task learning frameworks. We obtain clear performance improvements using semantic role labeling as the auxiliary task. We conduct a thorough analysis to demonstrate how multi-task learning helps, what has been solved for the task, and what is next. We show that future developments should improve the ability of the models to capture long-range dependencies and consider other auxiliary tasks such as dependency parsing or recognizing textual entailment. We emphasize that future improvements can be measured more reliably if opinion expressions with missing roles are curated and if the evaluation considers all mentions in opinion role coreference chains as well as discontinuous roles. To the best of our knowledge, we propose the first abstract anaphora resolution model that handles the unrestricted phenomenon in a realistic setting. We cast abstract anaphora resolution as the task of learning attributes of the relation that holds between the sentence with the abstract anaphor and its antecedent. We propose a Mention-Ranking siamese-LSTM model (MR-LSTM) for learning what characterizes the mentioned relation in a data-driven fashion. The current resources for abstract anaphora resolution are quite limited. However, we can train our models without conventional data for abstract anaphora resolution. In particular, we can train our models on many instances of antecedent-anaphoric sentence pairs. Such pairs can be automatically extracted from parsed corpora by searching for a common construction which consists of a verb with an embedded sentence (complement or adverbial), applying a simple transformation that replaces the embedded sentence with an abstract anaphor, and using the cut-off embedded sentence as the antecedent. We refer to the extracted data as silver data. We evaluate our MR-LSTM models in a realistic task setup in which models need to rank embedded sentences and verb phrases from the sentence with the anaphor as well as a few preceding sentences. We report the first benchmark results on an abstract anaphora subset of the ARRAU corpus \citep{uryupina_et_al_2016} which presents a greater challenge due to a mixture of nominal and pronominal anaphors as well as a greater range of confounders. We also use two additional evaluation datasets: a subset of the CoNLL-12 shared task dataset \citep{pradhan_et_al_2012} and a subset of the ASN corpus \citep{kolhatkar_et_al_2013_crowdsourcing}. We show that our MR-LSTM models outperform the baselines in all evaluation datasets, except for events in the CoNLL-12 dataset. We conclude that training on the small-scale gold data works well if we encounter the same type of anaphors at the evaluation time. However, the gold training data contains only six shell nouns and events and thus resolution of anaphors in the ARRAU corpus that covers a variety of anaphor types benefits from the silver data. Our MR-LSTM models for resolution of abstract anaphors outperform the prior work for shell noun resolution \citep{kolhatkar_et_al_2013} in their restricted task setup. Finally, we try to get the best out of the gold and silver training data by mixing them. Moreover, we speculate that we could improve the training on a mixture if we: (i) handle artifacts in the silver data with adversarial training and (ii) use multi-task learning to enable our models to make ranking decisions dependent on the type of anaphor. These proposals give us mixed results and hence a robust mixed training strategy remains a challenge."

#abstract_short = "A short version of the abstract."

# Featured image thumbnail (optional)
image_preview = ""

# Is this a selected publication? (true/false)
selected = true

# Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter the filename (excluding '.md') of your project file in `content/project/`.
#   E.g. `projects = ["deep-learning"]` references `content/project/deep-learning.md`.
#projects = []

# Links (optional).
url_pdf = "http://archiv.ub.uni-heidelberg.de/volltextserver/27699/"
url_preprint = ""
url_code = ""
url_dataset = ""
url_project = ""
url_slides = ""
url_video = ""
url_poster = ""
url_source = ""

# Custom links (optional).
#   Uncomment line below to enable. For multiple links, use the form `[{...}, {...}, {...}]`.
# url_custom = [{name = "Custom Link", url = "http://example.org"}]

# Does the content use math formatting?
math = true

# Does the content use source code highlighting?
highlight = true

# Featured image
# Place your image in the `static/img/` folder and reference its filename below, e.g. `image = "example.jpg"`.
[header]
#image = "headers/bubbles-wide.jpg"
#caption = "My caption 😄"

+++

