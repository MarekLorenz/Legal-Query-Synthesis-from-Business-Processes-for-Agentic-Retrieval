# Legal Query Synthesis from Business Processes for Agentic Retrieval

## Setup

Start a virtual environment using Python 3.12 and open it.

`python -m venv .venv`
`source .venv/bin/activate`

Install dependencies from requirements.txt.

`pip install -r requirements.txt`


### LLM providers (for query variants and judge)

`main.py` can generate query variants with **Ollama** (default) or **Gemini**:

- **Ollama:** install and start [Ollama](https://ollama.com), then pull a model (e.g. `ollama pull deepseek-r1:latest`). Optional env: `OLLAMA_MODEL`, `OLLAMA_BASE_URL`.
- **Gemini:** set `GOOGLE_API_KEY` in the environment (or `.env` via `python-dotenv`).

Use `--provider records` to skip variant generation and read variants from an existing CSV (see below).

## Reproducing Results

To reproduce the final results from the pipeline evaluation in the presentation and the report, open the notebooks under regulatory_relevance4process/SOTA_NLP_LIR/query_diversification_evaluation in their corresponding reproducability version and re-run them, to compare them to the original results.

The exact ranking results that have been used for the evaluation can be found in regulatory_relevance4process-D73C/SOTA_NLP_LIR/query_diversification_results.

There are more experiments, however most of them only served for improving the model.

## Web App

### Run

From the project root:

```bash
cd webapp
npm install
npm run backend
```

In a second terminal:

```bash
cd webapp
npm run dev
```

Open `http://127.0.0.1:5173`.

### Upload your own process

1. Upload a **BPMN diagram** (`.xml` or `.bpmn`).
2. Upload a **corpus Excel file** (`.xlsx`) with a `requirement_text` column.
3. Upload the **process JSON** export (`top_level`, `medium_levels`, `low_levels`)
4. Click **Load process**, then **Run annotations**.

The backend stores uploads in a temporary session and parses subprocess/task structure from the BPMN XML element ids (used for diagram highlighting and clicks).

Use Case 1 is in the appropriate format at `webapp/data/uc1`

## Basic run (one use case)

```bash
# From project root, with venv activated
python main.py --use-case uc1 --provider gemini
```

Equivalent defaults: `--use-case uc1` and `--provider ollama` if you do not pass flags.

### Run all three use cases

```bash
python main.py --use-case uc1 --provider gemini
python main.py --use-case uc2 --provider gemini
python main.py --use-case uc3 --provider gemini
```

Each command appends/overwrites that use case’s `output_with_agents_<uc>.csv` and `output_ranking_<uc>.xlsx`.

### Reuse cached query variants (no new LLM calls for agents) to save time (!)

After variants exist in `output_with_agents_uc1.csv` (etc.), rerun retrieval only:

```bash
python main.py --use-case uc1 --provider records
```

Optional custom CSV:

```bash
python main.py --use-case uc1 --provider records --records-path path/to/my_variants.csv
```

The CSV must include columns: `level`, `query`, `legal_terminology_rewrite`, `regulatory_compliance_query`, `contract_clause_query`, `risk_scenario_query`.

### With LLM judge (second retrieval pass)

```bash
python main.py --use-case uc1 --provider records --enable-llm-judge --judge-provider gemini
```

Judge provider defaults to the variant `--provider` when it is not `records`; otherwise it uses the default in `main.py` (`LLM_PROVIDER`, typically `ollama`).

### CLI reference

| Flag | Description |
|------|-------------|
| `--use-case {uc1,uc2,uc3}` | Benchmark use case (default: `uc1`) |
| `--provider {ollama,gemini,records}` | How to obtain the four query variants |
| `--records-path PATH` | CSV for `records` provider (default: `output_with_agents_<uc>.csv`) |
| `--enable-llm-judge` | Refine variants with an LLM judge and add judged ranking sheets |
| `--judge-provider {ollama,gemini}` | Provider for the judge step |

### Output files (per use case)

| Path | Content |
|------|---------|
| `output_with_agents_<uc>.csv` | Original queries plus four agent variant columns |
| `output_ranking_<uc>.xlsx` | One sheet per method (e.g. `BM25_baseline`, `BM25_RRF_weighted_CE`, …) |
| `regulatory_relevance4process-D73C/SOTA_NLP_LIR/output_ranking_input_eval/<uc>/algo_output/` | Per-level `*_algo_output_<method>.xlsx` for evaluation notebooks |