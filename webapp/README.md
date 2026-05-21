# Regulatory Process Annotator

Local BPMN viewer and regulatory-text annotator.

## Run

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

## Upload your own process

1. Upload a **BPMN diagram** (`.xml` or `.bpmn`).
2. Upload a **corpus Excel file** (`.xlsx`) with a `requirement_text` column.
3. Upload the **process JSON** export (`top_level`, `medium_levels`, `low_levels`)
4. Click **Load process**, then **Run annotations**.

The backend stores uploads in a temporary session and parses subprocess/task structure from the BPMN XML element ids (used for diagram highlighting and clicks).

## API

- `POST /api/upload` — multipart: `diagram_xml`, `corpus_xlsx`, optional `process_json`
- `POST /api/annotate` — JSON: `{ "session_id": "...", "top_k": 5 }`
- `GET /api/health`
