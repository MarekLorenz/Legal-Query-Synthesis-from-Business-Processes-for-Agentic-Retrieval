import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import BpmnJS from "bpmn-js/lib/NavigatedViewer";
import "bpmn-js/dist/assets/diagram-js.css";
import "bpmn-js/dist/assets/bpmn-js.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

function parseErrorMessage(rawText) {
  if (!rawText) return "Request failed";
  try {
    const parsed = JSON.parse(rawText);
    if (typeof parsed.detail === "string") return parsed.detail;
    if (Array.isArray(parsed.detail)) {
      return parsed.detail.map((item) => item.msg || String(item)).join("; ");
    }
  } catch {
    // Keep plain-text backend errors.
  }
  return rawText;
}

async function fetchJson(path, options) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(parseErrorMessage(text) || `Request failed: ${response.status}`);
  }
  return response.json();
}

function targetById(processModel) {
  if (!processModel) return {};
  const map = { [processModel.topLevel.id]: processModel.topLevel };
  for (const subprocess of processModel.subprocesses || []) {
    map[subprocess.id] = subprocess;
    for (const child of subprocess.children || []) {
      map[child.id] = child;
    }
  }
  return map;
}

function annotationScore(annotations) {
  if (!annotations?.length) return null;
  return Math.max(...annotations.map((annotation) => Number(annotation.score)).filter(Number.isFinite));
}

function annotationTargets(processModel) {
  if (!processModel) return [];
  return [
    processModel.topLevel,
    ...(processModel.subprocesses || []),
    ...(processModel.subprocesses || []).flatMap((item) => item.children || []),
  ].filter(Boolean);
}

function regulatoryMatchLevels(processModel, annotations) {
  if (!processModel || !annotations) return {};

  const scoresById = Object.fromEntries(
    annotationTargets(processModel)
      .map((target) => [target.id, annotationScore(annotations[target.id])])
      .filter(([, score]) => score !== null),
  );
  const scores = Object.values(scoresById);
  if (!scores.length) return {};

  const sortedScores = [...scores].sort((a, b) => b - a);
  const highThreshold = sortedScores[Math.max(0, Math.ceil(sortedScores.length * 0.3) - 1)];

  return Object.fromEntries(
    Object.entries(scoresById).map(([id, score]) => [id, score >= highThreshold ? "high" : "standard"]),
  );
}

function AnnotationList({ title, annotations }) {
  if (!annotations) {
    return (
      <section className="annotation-section">
        <h3>{title}</h3>
        <p className="muted">Run annotations to retrieve regulatory text.</p>
      </section>
    );
  }

  return (
    <section className="annotation-section">
      <h3>{title}</h3>
      <div className="annotation-list">
        {annotations.map((annotation, index) => (
          <article className="annotation-card" key={`${annotation.text}-${index}`}>
            <div className="annotation-meta">
              <span>{annotation.method === "bm25_ce" ? "Cross-encoder match" : "Regulatory match"}</span>
            </div>
            <p>{annotation.text}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function DetailPanel({ processModel, selected, expanded, annotations, onToggleExpanded }) {
  if (!selected) {
    return (
      <aside className="details-panel">
        <p className="eyebrow">process</p>
        <h2>{processModel?.topLevel?.name || "Process Annotations"}</h2>
        <p className="muted">
          {processModel?.topLevel?.description ||
            "Click a subprocess or task in the diagram to inspect relevant regulatory text."}
        </p>
        {processModel?.topLevel && (
          <AnnotationList title="Process-Level Regulatory Texts" annotations={annotations?.[processModel.topLevel.id]} />
        )}
      </aside>
    );
  }

  const children = selected.children || [];

  return (
    <aside className="details-panel">
      <div className="panel-header">
        <p className="eyebrow">{selected.level}</p>
        <h2>
          {selected.number ? `${selected.number}. ` : ""}
          {selected.name}
        </h2>
      </div>
      <p className="description">{selected.description}</p>
      <AnnotationList title="Relevant Regulatory Texts" annotations={annotations?.[selected.id]} />

      {children.length > 0 && (
        <section className="children-section">
          <button className="expand-section-button" type="button" onClick={() => onToggleExpanded(selected.id)}>
            <span>{expanded ? "Collapse event and task level" : "Expand event and task level"}</span>
            <span className="expand-count">{children.length} items</span>
          </button>
          {expanded && (
            <>
              <h3>Event and Task Level</h3>
              {children.map((child) => (
                <div className="child-card" key={child.id}>
                  <h4>
                    {child.number}. {child.name}
                  </h4>
                  <p>{child.description}</p>
                  <AnnotationList title="Regulatory Texts" annotations={annotations?.[child.id]} />
                </div>
              ))}
            </>
          )}
        </section>
      )}
    </aside>
  );
}

function UploadPanel({ diagramFile, corpusFile, processJsonFile, uploading, onDiagramChange, onCorpusChange, onProcessJsonChange, onUpload }) {
  const canUpload = Boolean(diagramFile && corpusFile && processJsonFile);

  return (
    <section className="upload-panel">
      <div className="upload-panel-copy">
        <h2>Load your process</h2>
        <p className="muted">
          Upload a BPMN diagram XML, a regulatory corpus Excel file, and the process JSON export (medium/low levels) for
          correct subprocess and task counts.
        </p>
      </div>
      <div className="upload-grid">
        <label className="upload-field">
          <span>BPMN diagram (.xml / .bpmn)</span>
          <input accept=".xml,.bpmn,application/xml,text/xml" onChange={onDiagramChange} type="file" />
          <span className="file-name">{diagramFile?.name || "No file selected"}</span>
        </label>
        <label className="upload-field">
          <span>Corpus (.xlsx, column <code>requirement_text</code>)</span>
          <input accept=".xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" onChange={onCorpusChange} type="file" />
          <span className="file-name">{corpusFile?.name || "No file selected"}</span>
        </label>
        <label className="upload-field">
          <span>Process JSON (required)</span>
          <input accept=".json,application/json" onChange={onProcessJsonChange} type="file" />
          <span className="file-name">{processJsonFile?.name || "No file selected"}</span>
        </label>
      </div>
      <button className="primary-button" disabled={!canUpload || uploading} onClick={onUpload} type="button">
        {uploading ? "Loading..." : "Load process"}
      </button>
    </section>
  );
}

export default function App() {
  const canvasRef = useRef(null);
  const viewerRef = useRef(null);
  const [sessionId, setSessionId] = useState(null);
  const [processModel, setProcessModel] = useState(null);
  const [annotations, setAnnotations] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [expandedIds, setExpandedIds] = useState(new Set());
  const [status, setStatus] = useState("Upload a BPMN diagram and corpus to begin");
  const [annotating, setAnnotating] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [topK, setTopK] = useState(5);
  const [diagramFile, setDiagramFile] = useState(null);
  const [corpusFile, setCorpusFile] = useState(null);
  const [processJsonFile, setProcessJsonFile] = useState(null);
  const [sessionSummary, setSessionSummary] = useState(null);

  const targets = useMemo(() => targetById(processModel), [processModel]);
  const matchLevels = useMemo(() => regulatoryMatchLevels(processModel, annotations), [annotations, processModel]);
  const selected = selectedId ? targets[selectedId] : null;
  const expanded = selectedId ? expandedIds.has(selectedId) : false;
  const isLoaded = Boolean(sessionId && processModel);

  const toggleExpanded = useCallback((id) => {
    setExpandedIds((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const importDiagram = useCallback(async (xml) => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    await viewer.importXML(xml);
    viewer.get("canvas").zoom("fit-viewport", "auto");
  }, []);

  useEffect(() => {
    const viewer = new BpmnJS({ container: canvasRef.current });
    viewerRef.current = viewer;
    return () => {
      viewer.destroy();
      viewerRef.current = null;
    };
  }, []);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !processModel) return undefined;

    const eventBus = viewer.get("eventBus");
    const handler = (event) => {
      const id = event.element?.id;
      if (targets[id]) {
        setSelectedId(id);
      }
    };
    eventBus.on("element.click", handler);
    return () => eventBus.off("element.click", handler);
  }, [processModel, targets]);

  const diagramMarkerIds = useMemo(() => {
    if (!processModel) return new Set();
    return new Set(
      (processModel.subprocesses || [])
        .filter((subprocess) => subprocess.diagramElement !== false)
        .map((subprocess) => subprocess.id),
    );
  }, [processModel]);

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !processModel) return;

    const canvas = viewer.get("canvas");
    diagramMarkerIds.forEach((elementId) => {
      try {
        canvas.removeMarker(elementId, "regulatory-match-high");
        canvas.removeMarker(elementId, "regulatory-match-standard");
        if (matchLevels[elementId] === "high") {
          canvas.addMarker(elementId, "regulatory-match-high");
        } else if (matchLevels[elementId] === "standard") {
          canvas.addMarker(elementId, "regulatory-match-standard");
        }
      } catch {
        // Skip elements that are not rendered in the current diagram.
      }
    });
  }, [diagramMarkerIds, matchLevels, processModel]);

  async function handleUpload() {
    if (!diagramFile || !corpusFile) return;

    setUploading(true);
    setStatus("Uploading files and parsing BPMN");
    setAnnotations(null);
    setSelectedId(null);
    setExpandedIds(new Set());

    const formData = new FormData();
    formData.append("diagram_xml", diagramFile);
    formData.append("corpus_xlsx", corpusFile);
    if (processJsonFile) {
      formData.append("process_json", processJsonFile);
    }
    formData.append("session_label", diagramFile.name.replace(/\.[^.]+$/, ""));

    try {
      const response = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(parseErrorMessage(text) || `Upload failed: ${response.status}`);
      }
      const data = await response.json();
      setSessionId(data.sessionId);
      setProcessModel(data.process);
      setSessionSummary({
        corpusDocuments: data.corpusDocuments,
        subprocessCount: data.subprocessCount,
        taskCount: data.taskCount,
      });
      await importDiagram(data.diagram.xml);
      setStatus(
        `Loaded ${data.subprocessCount} subprocesses, ${data.taskCount} tasks, ${data.corpusDocuments} corpus documents`,
      );
    } catch (error) {
      setStatus(error.message);
    } finally {
      setUploading(false);
    }
  }

  async function runAnnotations() {
    if (!sessionId) return;

    setAnnotating(true);
    setStatus("Running BM25 + cross-encoder annotations");
    try {
      const data = await fetchJson("/api/annotate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          top_k: topK,
          include_cross_encoder: true,
        }),
      });
      setAnnotations(data.annotations);
      setStatus("Annotations ready");
    } catch (error) {
      setStatus(error.message);
    } finally {
      setAnnotating(false);
    }
  }

  function resetSession() {
    setSessionId(null);
    setProcessModel(null);
    setAnnotations(null);
    setSelectedId(null);
    setExpandedIds(new Set());
    setSessionSummary(null);
    setStatus("Upload a BPMN diagram and corpus to begin");
  }

  function zoom(delta) {
    const canvas = viewerRef.current?.get("canvas");
    if (!canvas) return;
    const currentZoom = canvas.zoom();
    canvas.zoom(Math.max(0.2, Math.min(2.5, currentZoom + delta)));
  }

  return (
    <main className="app">
      <header className="topbar">
        <div>
          <p className="eyebrow">BPMN Annotator</p>
          <h1>Regulatory Text Retrieval</h1>
        </div>
        {isLoaded && (
          <div className="controls">
            <label>
              Top K
              <input min="1" max="20" type="number" value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
            </label>
            <button className="secondary-button" onClick={resetSession} type="button">
              Change files
            </button>
            <button className="primary-button" disabled={annotating} onClick={runAnnotations} type="button">
              {annotating ? "Annotating..." : "Run annotations"}
            </button>
          </div>
        )}
      </header>

      {!isLoaded && (
        <UploadPanel
          corpusFile={corpusFile}
          diagramFile={diagramFile}
          onCorpusChange={(event) => setCorpusFile(event.target.files?.[0] || null)}
          onDiagramChange={(event) => setDiagramFile(event.target.files?.[0] || null)}
          onProcessJsonChange={(event) => setProcessJsonFile(event.target.files?.[0] || null)}
          onUpload={handleUpload}
          processJsonFile={processJsonFile}
          uploading={uploading}
        />
      )}

      <section className="status-row">
        <span>{status}</span>
        {sessionSummary && (
          <span className="session-summary">
            {sessionSummary.subprocessCount} subprocesses · {sessionSummary.taskCount} tasks · {sessionSummary.corpusDocuments}{" "}
            corpus rows
          </span>
        )}
        {annotations && (
          <span className="match-legend">
            <span className="legend-dot" />
            Strong regulatory match
          </span>
        )}
        {isLoaded && (
          <div className="zoom-controls">
            <button onClick={() => zoom(-0.1)} type="button">
              −
            </button>
            <button onClick={() => viewerRef.current?.get("canvas").zoom("fit-viewport", "auto")} type="button">
              Fit
            </button>
            <button onClick={() => zoom(0.1)} type="button">
              +
            </button>
          </div>
        )}
      </section>

      <section className={`workspace ${isLoaded ? "" : "workspace-hidden"}`}>
        <div className="diagram-shell">
          <div className="diagram-canvas" ref={canvasRef} />
        </div>
        <DetailPanel
          annotations={annotations}
          expanded={expanded}
          onToggleExpanded={toggleExpanded}
          processModel={processModel}
          selected={selected}
        />
      </section>
    </main>
  );
}
