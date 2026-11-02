# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,Evaluation dataset — methodology
# MAGIC %md
# MAGIC # Sprint 5 — Evaluation dataset
# MAGIC
# MAGIC **Purpose:** create a small, curated, reproducible set of Civilization IV strategy questions with ground truth for evaluating the Civ4 RAG pipeline (Sprints 3-4).
# MAGIC
# MAGIC ## How questions were selected
# MAGIC
# MAGIC * **2 questions per source document** (24 questions over the 12 indexed CivFanatics strategy guides), so every corpus category is represented.
# MAGIC * Questions were written **from the actual chunk content** of each document (read from `workspace.gcivilization.civ4_chunks`), not from general Civ4 knowledge — they must be answerable from the indexed corpus.
# MAGIC * Categories mirror the corpus metadata: `beginner_strategy`, `economy` (Specialist Economy), `city_specialization`, `combat_mechanics`, `technology_research`, `military_strategy` (Early Rush, Stack of Doom), `civ_specific` (Cyrus, Julius Caesar), `cultural_victory`, `diplomatic_victory`.
# MAGIC
# MAGIC ## Ground truth
# MAGIC
# MAGIC * `expected_document_ids` — the document(s) that contain the answer. **Document-level ground truth**: exact chunk IDs are not specified (chunks of one document overlap heavily).
# MAGIC * `expected_answer` — a short factual summary **written from the source chunks** (verified against `civ4_chunks` content). It is used as `expected_response` by the `answer_correctness` judge and is NOT a full model answer.
# MAGIC * `expected_source_titles` — derived from the corpus (document_id → title).
# MAGIC
# MAGIC ## Limitations (documented)
# MAGIC
# MAGIC * Small dataset (24 questions) — sufficient for methodology demonstration, not for statistically strong conclusions.
# MAGIC * Document-level ground truth means retrieval metrics credit any chunk from the right document.
# MAGIC * `expected_answer` texts are concise summaries; the `answer_correctness` judge therefore measures factual coverage, not stylistic similarity.
# MAGIC
# MAGIC **Output:** `workspace.gcivilization.civ4_eval_questions` (Delta, overwrite — deterministic and idempotent).

# COMMAND ----------

# DBTITLE 1,Evaluation question set (24 questions, 2 per document)
# The curated evaluation question set (v1).
# 24 questions, 2 per document. expected_answer texts are short factual
# summaries written FROM the source documents (verified against
# civ4_chunks content), not invented.

EVAL_QUESTIONS = [
    # --- Beginner strategy (WoundedKnight + Sisiutil) ---
    dict(question_id="q01",
         question="What should a brand-new Civilization IV player focus on first to improve at the game?",
         expected_answer="Learn the rules of the game: the tech trees, improvements, and units. It is much easier to do well with a clear strategy and knowing where you want to go.",
         expected_document_ids=["01_woundedknights_strategy_guide", "02_sisiutils_strategy_guide"],
         category="beginner_strategy", difficulty="easy"),
    dict(question_id="q02",
         question="How does the choice of map type affect your Civilization IV strategy?",
         expected_answer="Map choice has a major impact: continental maps promote early isolation of civs into 2-3 large groups, and archipelago maps can isolate individual civs entirely, which changes the strategy you should adopt.",
         expected_document_ids=["01_woundedknights_strategy_guide"],
         category="beginner_strategy", difficulty="medium"),
    dict(question_id="q03",
         question="How does Sisiutil's beginner guide distinguish tips that only apply to the Warlords and Beyond the Sword expansions from vanilla Civilization IV?",
         expected_answer="Wherever a pointer applies only to expansion content, the term Warlords/BtS or BtS is included in bold. Warlords/BtS applies to both expansion packs; BtS only to Beyond the Sword. Vanilla means non-expansion Civilization IV.",
         expected_document_ids=["02_sisiutils_strategy_guide"],
         category="beginner_strategy", difficulty="medium"),
    dict(question_id="q04",
         question="What does Sisiutil's Civilization IV strategy guide aim to provide, and who is it for?",
         expected_answer="A compilation of tips, tricks, tactics, and strategies gleaned from the Civilization Fanatics Center forums and the author's own experience, intended to provide a foundation for players new to the game. It is explicitly not a definitive collection, and strategies that run counter to its advice can also be effective.",
         expected_document_ids=["02_sisiutils_strategy_guide"],
         category="beginner_strategy", difficulty="easy"),

    # --- Economy: Specialist Economy ---
    dict(question_id="q05",
         question="What is a Specialist Economy in Civilization IV and how does it work?",
         expected_answer="A civilization in which the majority of research is funded by Scientist specialists supported by 1-2 farms or other high food resources. This approach tends to generate Great Scientists more quickly than other economies.",
         expected_document_ids=["03_specialist_economy"],
         category="economy", difficulty="medium"),
    dict(question_id="q06",
         question="What are the advantages and limitations of running a Specialist Economy compared to a Cottage Economy in Civilization IV?",
         expected_answer="Advantages: the science slider increases flexibility to divert commerce directly to gold or to culture/happiness; it is more effective at higher difficulty levels (Emperor and above). Limitations: it may require more micromanaging than a Cottage Economy; specialists require supporting food resources or farms, which can limit early city placement unless food or fresh water is available (less of an issue after Civil Service).",
         expected_document_ids=["03_specialist_economy"],
         category="economy", difficulty="hard"),

    # --- City specialization ---
    dict(question_id="q07",
         question="How can you tell how much a potential city site can be specialized in Civilization IV?",
         expected_answer="Count the positive and negative food tiles to find how much food the city lacks to work all tiles in its BFC. The larger the (hopefully positive) food surplus number, the more tiles the city can devote towards specialization, making it an indicator of how good the site is. It does not indicate what type of specialization the city should receive.",
         expected_document_ids=["04_city_specialization"],
         category="city_specialization", difficulty="medium"),
    dict(question_id="q08",
         question="What are the three basic things cities specialize in according to Civ4 city specialization strategy?",
         expected_answer="Cities essentially specialize in one of three things: surplus food, production (hammers), or commerce. Production cities specialize in hammers, food cities in surplus food, and commerce cities in commerce. This is what allows them to become GP Farms, Production Hubs, or Commerce Centres; which element a city specializes in depends on the land and the game era.",
         expected_document_ids=["04_city_specialization"],
         category="city_specialization", difficulty="medium"),

    # --- Combat mechanics ---
    dict(question_id="q09",
         question="How is an attacker's modified strength calculated in Civilization IV combat?",
         expected_answer="First apply any bonuses the attacker gets that are independent of the defender (generic strength promotions only, in the default game), then multiply by the fraction of remaining hit points. For example, a swordsman with strength 6, Combat I (+10%) at 73 hps has modified strength 6 * 1.1 * 73/100 = 4.82.",
         expected_document_ids=["05_combat_explained"],
         category="combat_mechanics", difficulty="medium"),
    dict(question_id="q10",
         question="How is a defender's modified strength calculated in Civilization IV combat?",
         expected_answer="Determine the defender's bonuses from terrain, general strength bonuses, bonuses versus the attacking unit's type, and fortification bonuses, and sum them. Then subtract any bonuses the attacker gets against this unit. If the overall value is positive, divide the defender's original strength by (1 + overall value); if negative, by (1 - overall value). Finally multiply by the defender's remaining hit points.",
         expected_document_ids=["05_combat_explained"],
         category="combat_mechanics", difficulty="hard"),

    # --- Technology research ---
    dict(question_id="q11",
         question="How does a Civilization IV city turn commerce into research beakers?",
         expected_answer="Each city generates raw commerce from the worked tiles and trade routes. The science/culture sliders translate this commerce into beakers (research), coins (gold), or notes (culture). Each city adds extra sources (shrines, production processes, specialists), then multiplies by building modifiers, e.g. beakers are multiplied by 1.25 with a library.",
         expected_document_ids=["06_technology_research_explained"],
         category="technology_research", difficulty="medium"),
    dict(question_id="q12",
         question="How is the cost of a technology calculated in Civilization IV?",
         expected_answer="Take the base cost of the technology, take the difficulty modifier plus 0.5 times the number of teammates, multiply and take the FLOOR, then multiply by the map modifier and take the FLOOR again, applying further modifiers similarly.",
         expected_document_ids=["06_technology_research_explained"],
         category="technology_research", difficulty="hard"),

    # --- Military strategy: Early Rush + Stack of Doom ---
    dict(question_id="q13",
         question="What is the objective of an early rush in Civilization IV?",
         expected_answer="The early rush is a strategy for conquering your nearest neighbour for fun and profit before the BCs become the ADs, i.e. attacking a close neighbour very early in the game.",
         expected_document_ids=["07_early_rush"],
         category="military_strategy", difficulty="easy"),
    dict(question_id="q14",
         question="What ingredients go into a successful early rush according to Sisiutil's guide?",
         expected_answer="The guide covers ingredients including speed, opponents, units, resources, exploration, technologies, workers/settlers/cities, food, hammers and forests, preparation, builds, tile improvements, chopping, and execution such as unit movement.",
         expected_document_ids=["07_early_rush"],
         category="military_strategy", difficulty="medium"),
    dict(question_id="q15",
         question="What is the Stack of Doom in Civilization IV?",
         expected_answer="The Stack o' Doom is a large offensive military stack in Civilization IV; the article describes its effective composition and use, including different stack objectives such as pillaging stacks and city-busting stacks.",
         expected_document_ids=["08_stack_of_doom"],
         category="military_strategy", difficulty="easy"),
    dict(question_id="q16",
         question="What types of offensive stacks does Sisiutil's Stack o' Doom article describe and what is each for?",
         expected_answer="The article describes pillaging stacks (for pillaging enemy land), city-busting stacks (for taking cities, with their own composition, promotions, and tactics), and mid-game stacks dealing with the impact of gunpowder.",
         expected_document_ids=["08_stack_of_doom"],
         category="military_strategy", difficulty="medium"),

    # --- Civ-specific: Cyrus + Julius Caesar ---
    dict(question_id="q17",
         question="Why is Cyrus of the Persians well suited to an Immortal rush in Civilization IV?",
         expected_answer="Cyrus is Imperialistic and Charismatic. Charismatic grants happy faces (+1 per city and per monument) and makes promotions require 25% less XP per level, giving more promotions and stronger units. Imperialistic gives a 100% bonus to Great General generation, resulting in more Great Military Instructors (+2 XP).",
         expected_document_ids=["09_cyrus_immortal_rush_guide"],
         category="civ_specific", difficulty="medium"),
    dict(question_id="q18",
         question="What makes the Persian Immortal unit so strong in Civilization IV?",
         expected_answer="The Immortal is a chariot unit available with The Wheel, requiring horses, with 4 strength, +100% attack versus axemen, a 10% withdrawal chance, and 2 movement points. It also has a 50% bonus versus Archery units and, unlike normal mounted units, receives defensive bonuses from terrain, making it a solid choice against most units.",
         expected_document_ids=["09_cyrus_immortal_rush_guide"],
         category="civ_specific", difficulty="medium"),
    dict(question_id="q19",
         question="What are Julius Caesar's traits, unique unit, and unique building in Civilization IV?",
         expected_answer="Traits: Organized and Imperialistic. Organized reduces civic upkeep by 50% and makes courthouses, lighthouses, and factories cheaper. Imperialistic synergizes with warmongering. His UB is the Forum, which replaces the Market and gives +25% GP points. His UU is the Praetorian, which replaces the Swordsman but has a strength of 8 and no city attack bonus.",
         expected_document_ids=["10_julius_caesar_guide"],
         category="civ_specific", difficulty="medium"),
    dict(question_id="q20",
         question="How does the Julius Caesar leader guide recommend opening the game with Rome in Civilization IV?",
         expected_answer="Research early technologies such as Bronze Working and the Wheel (needed to connect Iron for praetorians). With a landlocked start the build order should be worker-warrior-settler-worker-barracks, then keep building praetorians until you conquer everything. The first city should be a good production city (hills and forests) claiming copper if possible, and the second city must claim the Iron.",
         expected_document_ids=["10_julius_caesar_guide"],
         category="civ_specific", difficulty="hard"),

    # --- Victory: Cultural + Diplomatic ---
    dict(question_id="q21",
         question="What happens when you switch the culture slider to 100% going for a cultural victory, and how should you exploit it?",
         expected_answer="Once you switch the culture slider to 100%, you no longer have to worry about happiness, so you should maximize the size of your cities for more artist specialists. At 100% culture funding you can trade your happiness resources for health ones or money.",
         expected_document_ids=["11_cultural_victory"],
         category="cultural_victory", difficulty="medium"),
    dict(question_id="q22",
         question="Which civics should you run at 100% culture in Civilization IV, and which should you avoid?",
         expected_answer="Do not use the Free Religion civic when turning the culture slider to 100%: its happiness bonus is negated by the happiness you get from theaters (at least +10 happy faces). The author suggests using Pacifism and Caste System instead; if you run low on happiness, build a colosseum.",
         expected_document_ids=["11_cultural_victory"],
         category="cultural_victory", difficulty="medium"),
    dict(question_id="q23",
         question="How can you check the diplomatic relations between AI civilizations in Civilization IV?",
         expected_answer="Go to the foreign screen and select one AI, then look at the standing shown below the picture of the other AIs. For a more detailed explanation, mouse over the other AIs' pictures to see their reasoning for being happy or unhappy with the selected AI.",
         expected_document_ids=["12_diplomatic_victory"],
         category="diplomatic_victory", difficulty="medium"),
    dict(question_id="q24",
         question="When will an AI civilization vote for your diplomatic victory in Civilization IV?",
         expected_answer="The AI will almost always vote for your diplomatic victory if it is friendly with you.",
         expected_document_ids=["12_diplomatic_victory"],
         category="diplomatic_victory", difficulty="easy"),
]

print(f"{len(EVAL_QUESTIONS)} evaluation questions defined")

# COMMAND ----------

# DBTITLE 1,Validate the question set
# Validation of the evaluation dataset before persisting:
#  - exactly 24 questions, unique ids and texts
#  - no empty question / expected_answer
#  - every expected document_id exists in the source corpus
#  - categories and difficulties use the controlled vocabularies

import pandas as pd

corpus = spark.sql("""
    SELECT document_id, title FROM workspace.gcivilization.civ4_chunks
    GROUP BY document_id, title
""").toPandas()
doc_title = dict(zip(corpus.document_id, corpus.title))
valid_ids = set(corpus.document_id)

errors = []
assert len(EVAL_QUESTIONS) == 24, f"expected 24 questions, got {len(EVAL_QUESTIONS)}"
for q in EVAL_QUESTIONS:
    qid = q["question_id"]
    if not q["question"].strip() or not q["expected_answer"].strip():
        errors.append(f"{qid}: empty question or expected_answer")
    for d in q["expected_document_ids"]:
        if d not in valid_ids:
            errors.append(f"{qid}: unknown document_id {d}")
    if not q["expected_document_ids"]:
        errors.append(f"{qid}: no expected_document_ids")
    if q["category"] not in {"beginner_strategy", "economy", "city_specialization",
                             "combat_mechanics", "technology_research", "military_strategy",
                             "civ_specific", "cultural_victory", "diplomatic_victory"}:
        errors.append(f"{qid}: bad category {q['category']}")
    if q["difficulty"] not in {"easy", "medium", "hard"}:
        errors.append(f"{qid}: bad difficulty {q['difficulty']}")

qids = [q["question_id"] for q in EVAL_QUESTIONS]
assert len(qids) == len(set(qids)), "duplicate question_id"
qtexts = [q["question"].strip().lower() for q in EVAL_QUESTIONS]
assert len(qtexts) == len(set(qtexts)), "duplicate question text"

assert not errors, "Validation errors:\n" + "\n".join(errors)
print(f"VALIDATION PASSED: {len(EVAL_QUESTIONS)} questions, all document IDs exist in corpus")

# COMMAND ----------

# DBTITLE 1,Persist to Delta (idempotent overwrite)
# Enrich with expected_source_titles (from the corpus), fingerprint the
# dataset version, and persist to Delta (overwrite — deterministic).

import hashlib, json
from pyspark.sql import functions as F

for q in EVAL_QUESTIONS:
    q["expected_source_titles"] = [doc_title[d] for d in q["expected_document_ids"]]

fingerprint = hashlib.md5(json.dumps(
    [(q["question_id"], q["question"]) for q in EVAL_QUESTIONS]).encode()).hexdigest()[:8]
EVAL_DATASET_VERSION = f"v1-{fingerprint}"
print("eval_dataset_version:", EVAL_DATASET_VERSION)

df = spark.createDataFrame(pd.DataFrame(EVAL_QUESTIONS))
df = df.withColumn("eval_dataset_version", F.lit(EVAL_DATASET_VERSION))
df.write.mode("overwrite").saveAsTable("workspace.gcivilization.civ4_eval_questions")
print("Saved workspace.gcivilization.civ4_eval_questions")

display(spark.sql("""
    SELECT category, difficulty, count(*) AS n
    FROM workspace.gcivilization.civ4_eval_questions
    GROUP BY category, difficulty ORDER BY category, difficulty
"""))

# COMMAND ----------

