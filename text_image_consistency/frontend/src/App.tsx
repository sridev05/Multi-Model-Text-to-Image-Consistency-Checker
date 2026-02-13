import React, { useState, useRef } from "react";

type EvaluationResult = {
  blip_caption: string;
  clip_score: number;
  grounding_score: number;
  tifa_score: number;
  ocr_score: number;
  forward_score: number;
  backward_score: number;
  final_score: number;
  verdict: "MATCH" | "PARTIAL MATCH" | "MISMATCH" | string;
};

type Status = "idle" | "evaluating" | "done" | "error";

const App: React.FC = () => {
  const [prompt, setPrompt] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<EvaluationResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileChange = (file: File | null) => {
    setImageFile(file);
    setResult(null);
    setError(null);
    if (file) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    } else {
      setPreviewUrl(null);
    }
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) {
      handleFileChange(file);
    }
  };

  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const triggerFileSelect = () => {
    fileInputRef.current?.click();
  };

  const verdictClass = (verdict: string | undefined) => {
    switch (verdict) {
      case "MATCH":
        return "badge badge-success";
      case "PARTIAL MATCH":
        return "badge badge-warning";
      case "MISMATCH":
        return "badge badge-danger";
      default:
        return "badge";
    }
  };

  const formatScore = (v: number | undefined) =>
    typeof v === "number" ? v.toFixed(4) : "–";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setResult(null);

    if (!imageFile) {
      setError("Please upload an image.");
      return;
    }
    if (!prompt.trim()) {
      setError("Please enter a prompt.");
      return;
    }

    setStatus("evaluating");
    try {
      const formData = new FormData();
      formData.append("image", imageFile);
      formData.append("prompt", prompt.trim());

      const response = await fetch("/api/evaluate", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `HTTP ${response.status}`);
      }

      const json: EvaluationResult = await response.json();
      setResult(json);
      setStatus("done");
    } catch (err: any) {
      setStatus("error");
      setError(err?.message || "Evaluation failed.");
    }
  };

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="header-left">
          <h1>Text–Image Consistency Evaluator</h1>
          <p>
            Upload a generated image and its prompt to obtain a research-grade
            consistency score.
          </p>
        </div>
        <div className="header-right">
          <span className="pill">Research‑grade</span>
          <span className="pill pill-outline">CPU‑only</span>
        </div>
      </header>

      <main className="app-main">
        <section className="panel panel-input">
          <form onSubmit={handleSubmit} className="input-grid">
            <div
              className="dropzone"
              onDrop={onDrop}
              onDragOver={onDragOver}
              onClick={triggerFileSelect}
            >
              {previewUrl ? (
                <>
                  <img
                    src={previewUrl}
                    alt="preview"
                    className="preview-image"
                  />
                  <div className="dropzone-overlay">
                    <span>Click or drop another image to replace</span>
                  </div>
                </>
              ) : (
                <div className="dropzone-placeholder">
                  <div className="icon-circle">🖼️</div>
                  <p className="dropzone-title">
                    Drop an image here or click to upload
                  </p>
                  <p className="dropzone-subtitle">
                    PNG / JPG, up to a few MB
                  </p>
                </div>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                style={{ display: "none" }}
                onChange={(e) => {
                  const file = e.target.files?.[0] || null;
                  if (file && !file.type.startsWith("image/")) return;
                  handleFileChange(file);
                }}
              />
            </div>

            <div className="prompt-column">
              <label className="field-label">Text prompt</label>
              <textarea
                className="prompt-input"
                placeholder="Describe the intended content of the image..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={8}
              />

              <div className="actions-row">
                <button
                  type="submit"
                  className="btn-primary"
                  disabled={status === "evaluating"}
                >
                  {status === "evaluating" ? "Evaluating..." : "Evaluate"}
                </button>
                {status === "evaluating" && (
                  <span className="status-pill">Running full pipeline…</span>
                )}
              </div>

              {error && <div className="error-banner">{error}</div>}
            </div>
          </form>
        </section>

        {result && (
          <section className="panel panel-results">
            <div className="results-header">
              <div>
                <h2>Evaluation Results</h2>
                <p className="caption">
                  BLIP caption: <span>{result.blip_caption}</span>
                </p>
              </div>
              <div className="final-score-block">
                <div className="final-score">
                  {result.final_score.toFixed(2)}
                  <span className="final-score-unit">/ 100</span>
                </div>
                <span className={verdictClass(result.verdict)}>
                  {result.verdict}
                </span>
              </div>
            </div>

            <div className="score-grid">
              <div className="score-card">
                <h3>Forward (Text → Image)</h3>
                <ul>
                  <li>
                    <span>CLIP semantic similarity</span>
                    <strong>{formatScore(result.clip_score)}</strong>
                  </li>
                  <li>
                    <span>Grounding DINO (entities)</span>
                    <strong>{formatScore(result.grounding_score)}</strong>
                  </li>
                  <li>
                    <span>TIFA faithfulness</span>
                    <strong>{formatScore(result.tifa_score)}</strong>
                  </li>
                  <li>
                    <span>OCR consistency</span>
                    <strong>{formatScore(result.ocr_score)}</strong>
                  </li>
                </ul>
                <div className="score-footer">
                  <span>Forward score</span>
                  <strong>{formatScore(result.forward_score)}</strong>
                </div>
              </div>

              <div className="score-card">
                <h3>Backward (Image → Text → Image)</h3>
                <ul>
                  <li>
                    <span>CLIP(image, BLIP caption)</span>
                    <strong>{formatScore(result.backward_score)}</strong>
                  </li>
                </ul>
                <div className="score-footer">
                  <span>Bidirectional fusion</span>
                  <strong>{result.final_score.toFixed(2)} / 100</strong>
                </div>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
};

export default App;

