(() => {
  "use strict";

  const ALLOWED_HOSTS = [
    "hdfcfund.com",
    "sebi.gov.in",
    "amfiindia.com",
    "investor.sebi.gov.in",
  ];

  const TYPE_LABELS = {
    answer: "ANSWER FROM OFFICIAL SOURCES",
    refusal_advice: "FACTS-ONLY ASSISTANT",
    refusal_performance: "FACTS-ONLY ASSISTANT",
    not_in_sources: "OUTSIDE CURATED SOURCES",
    clarify: "CHOOSE A SCHEME",
    pii: "PRIVACY PROTECTION",
    error: "SOMETHING WENT WRONG",
  };

  const TYPE_TAGS = {
    refusal_advice: ["No advice or recommendations"],
    refusal_performance: ["No return or performance views"],
    not_in_sources: ["Only curated official documents"],
    clarify: ["Scheme context required"],
    pii: ["No personal data shown or stored"],
    error: ["No unverified answer shown"],
  };

  const state = {
    selectedScheme: null,
    controller: null,
    requestNumber: 0,
    lastFocus: null,
  };

  const root = document.getElementById("answer-root");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("question-input");
  const sendButton = document.getElementById("send-button");
  const chips = Array.from(document.querySelectorAll("[data-scheme]"));

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function safeHref(value) {
    if (typeof value !== "string" || !value.trim()) return null;
    try {
      const url = new URL(value, window.location.origin);
      if (!["http:", "https:"].includes(url.protocol)) return null;
      const host = url.hostname.toLowerCase();
      const allowed = ALLOWED_HOSTS.some(
        (candidate) => host === candidate || host.endsWith("." + candidate),
      );
      return allowed ? url.href : null;
    } catch (_error) {
      return null;
    }
  }

  function externalLink(value, label) {
    const href = safeHref(value);
    if (!href) return null;
    const link = element("a", "source-link", label);
    link.href = href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.append(" \u2197");
    return link;
  }

  function metaItem(text) {
    return text ? element("span", "", text) : null;
  }

  function formatDate(value) {
    if (!value) return "";
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return String(value);
    return parsed.toLocaleDateString("en-IN", {
      day: "numeric",
      month: "short",
      year: "numeric",
    });
  }

  function number(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return "0";
    return numeric.toLocaleString("en-IN");
  }

  function scrollToAnswer() {
    const section = document.querySelector(".answer-section");
    if (section) section.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function showLoading() {
    const card = element("div", "loading-card");
    card.append(
      element("span", "spinner"),
      element("span", "", "Checking the official documents\u2026"),
    );
    root.replaceChildren(card);
    scrollToAnswer();
  }

  function showEmpty() {
    const card = element("div", "answer-empty");
    card.append(element("p", "", "Your answer will appear here"));
    root.replaceChildren(card);
  }

  function renderError(message) {
    renderEnvelope({
      type: "error",
      answer:
        message ||
        "I could not produce a verified answer right now. Please try again.",
      source_url: null,
      source_name: null,
      as_on_date: null,
      proof: null,
      can_do: [],
      chunks_used: [],
    });
  }

  function followups(questions) {
    if (!Array.isArray(questions) || !questions.length) return null;
    const section = element("div", "followups");
    section.append(element("h3", "", "EXPLORE OFFICIAL FACTS"));
    const list = element("div", "followup-chips");
    questions.slice(0, 6).forEach((question) => {
      if (typeof question !== "string" || !question.trim()) return;
      const chip = element("button", "followup-chip", question);
      chip.type = "button";
      chip.addEventListener("click", () => {
        setSelectedScheme(null);
        input.value = question;
        input.focus({ preventScroll: true });
      });
      list.append(chip);
    });
    if (!list.children.length) return null;
    section.append(list);
    return section;
  }

  function sourceRow(envelope) {
    const row = element("div", "source-row");
    const left = element("div", "");
    const label =
      typeof envelope.source_name === "string" && envelope.source_name.trim()
        ? envelope.source_name.trim()
        : "Official source";

    const link = externalLink(
      envelope.source_url,
      `Open official source${label ? " \u00b7 " + label : ""}`,
    );
    if (link) left.append(link);

    if (envelope.type === "answer") {
      left.append(
        element(
          "p",
          "source-date",
          envelope.as_on_date
            ? `Last updated from sources: ${formatDate(envelope.as_on_date)}`
            : `Last updated from sources: retrieved on ${formatDate(
                envelope.retrieved_at || new Date().toISOString(),
              )}`,
        ),
      );
    }
    row.append(left);

    const tags = element("div", "tag-row");
    const typeTags = TYPE_TAGS[envelope.type] || [];
    typeTags.forEach((tag) => tags.append(element("span", "tag", tag)));
    if (envelope.type === "answer") {
      if (label) tags.append(element("span", "tag", label));
      tags.append(element("span", "tag", "Verified against retrieved text"));
      const count = Array.isArray(envelope.chunks_used) ? envelope.chunks_used.length : 0;
      if (count) tags.append(element("span", "tag", `${count} source chunk${count === 1 ? "" : "s"}`));
    }
    if (tags.children.length) row.append(tags);
    return row;
  }

  function renderEnvelope(envelope, question) {
    if (!envelope || typeof envelope !== "object" || typeof envelope.type !== "string") {
      renderError();
      return;
    }

    const card = element("article", `answer-card type-${envelope.type}`);
    const top = element("div", "card-top");
    const kicker = element("p", "card-kicker");
    kicker.append(element("span", "", TYPE_LABELS[envelope.type] || "OFFICIAL FACTS"));
    if (envelope.type === "answer") {
      kicker.append(element("span", "verified-pill", "Verified source"));
    }
    top.append(kicker);

    if (envelope.type === "answer" && typeof question === "string" && question.trim()) {
      top.append(element("p", "query-pill", question.trim()));
    }
    card.append(top);

    const answer = element(
      "p",
      "answer-text",
      typeof envelope.answer === "string" && envelope.answer.trim()
        ? envelope.answer.trim()
        : "I could not produce a verified answer right now. Please try again.",
    );
    card.append(answer);

    if (envelope.type === "answer") {
      card.append(sourceRow(envelope));
    } else if (envelope.source_url) {
      card.append(sourceRow(envelope));
    }

    if (typeof envelope.proof === "string" && envelope.proof.trim()) {
      const proof = element("details", "proof");
      proof.append(element("summary", "", "Show me the proof"));
      proof.append(element("blockquote", "", envelope.proof.trim()));
      card.append(proof);
    }

    const suggestions = followups(envelope.can_do);
    if (suggestions) card.append(suggestions);

    root.replaceChildren(card);
    scrollToAnswer();
  }

  async function ask(rawQuestion) {
    const question = (rawQuestion || "").trim();
    if (!question) {
      input.focus();
      return;
    }

    if (state.controller) state.controller.abort();
    const controller = new AbortController();
    state.controller = controller;
    const requestNumber = ++state.requestNumber;

    input.value = "";
    showLoading();
    setBusy(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          scheme: state.selectedScheme,
        }),
        signal: controller.signal,
      });

      let data = null;
      try {
        data = await response.json();
      } catch (_error) {
        data = null;
      }

      if (requestNumber !== state.requestNumber) return;

      if (!response.ok && !(data && data.type)) {
        renderError(
          response.status === 503
            ? "The assistant is unavailable right now. Please try again in a moment."
            : "I could not process that request. Please check it and try again.",
        );
        return;
      }

      renderEnvelope(data, question);
    } catch (error) {
      if (error && error.name === "AbortError") return;
      if (requestNumber !== state.requestNumber) return;
      renderError("I could not reach the assistant. Please check your connection and try again.");
    } finally {
      if (requestNumber === state.requestNumber) {
        state.controller = null;
        setBusy(false);
      }
    }
  }

  function setBusy(busy) {
    sendButton.disabled = busy;
    if (!busy) input.focus({ preventScroll: true });
  }

  function setSelectedScheme(value) {
    state.selectedScheme = value;
    chips.forEach((chip) => {
      const active = chip.dataset.scheme === value;
      chip.setAttribute("aria-pressed", active ? "true" : "false");
    });
  }

  function openModal(id) {
    const modal = document.getElementById(`${id}-modal`);
    if (!modal) return;
    state.lastFocus = document.activeElement;
    modal.hidden = false;
    document.body.style.overflow = "hidden";
    const close = modal.querySelector("[data-close-modal]");
    if (close) close.focus();

    if (id === "sources") loadSources();
    if (id === "pipeline") loadPipeline();
  }

  function closeModal(modal) {
    if (!modal) return;
    modal.hidden = true;
    document.body.style.overflow = "";
    if (state.lastFocus && typeof state.lastFocus.focus === "function") {
      state.lastFocus.focus();
    }
    state.lastFocus = null;
  }

  async function loadSources() {
    const body = document.getElementById("sources-body");
    body.replaceChildren(element("p", "muted", "Loading official sources\u2026"));
    try {
      const response = await fetch("/api/sources");
      if (!response.ok) throw new Error("sources unavailable");
      const data = await response.json();
      renderSources(body, data);
    } catch (_error) {
      body.replaceChildren(
        element("p", "muted", "Could not load the source inventory right now."),
      );
    }
  }

  function renderSources(body, data) {
    const sources = Array.isArray(data && data.sources) ? data.sources : [];
    body.replaceChildren();
    body.append(
      element(
        "p",
        "muted",
        `${number(data && data.count !== undefined ? data.count : sources.length)} official source${
          sources.length === 1 ? "" : "s"
        } in the curated inventory.`,
      ),
    );

    const list = element("div", "source-list");
    sources.forEach((source) => {
      const item = element("div", "source-item");
      const left = element("div", "");

      const title =
        (source && (source.title || source.source_id)) || "Official source";
      left.append(element("h3", "", title));

      const description = source && source.publisher ? source.publisher : "";
      if (description) left.append(element("p", "", description));

      const meta = element("div", "source-meta");
      const pieces = [
        source && source.doc_type,
        source && source.scheme,
        source && source.chunks !== undefined ? `${number(source.chunks)} chunks` : "",
        source && source.fetched_at ? `Fetched ${formatDate(source.fetched_at)}` : "",
      ].filter(Boolean);
      pieces.forEach((piece) => {
        const node = metaItem(piece);
        if (node) meta.append(node);
      });
      if (meta.children.length) left.append(meta);

      const link = externalLink(source && source.url, "Open source");
      if (link) left.append(link);

      const right = element("div", "");
      const statusText = (source && source.status) || "not ingested";
      const pill = element(
        "span",
        `status-pill${statusText === "ok" || statusText === "ingested" ? "" : " degraded"}`,
        statusText,
      );
      right.append(pill);
      item.append(left, right);
      list.append(item);
    });

    if (sources.length) {
      body.append(list);
    } else {
      body.append(element("p", "muted", "No sources are available."));
    }
  }

  async function loadPipeline() {
    const body = document.getElementById("pipeline-body");
    body.replaceChildren(element("p", "muted", "Loading pipeline summaries\u2026"));
    try {
      const [pipelineResponse, healthResponse] = await Promise.all([
        fetch("/api/pipeline"),
        fetch("/health").catch(() => null),
      ]);
      if (!pipelineResponse.ok) throw new Error("pipeline unavailable");
      const pipelineData = await pipelineResponse.json();
      const healthData =
        healthResponse && healthResponse.ok ? await healthResponse.json() : null;
      renderPipeline(body, pipelineData, healthData);
    } catch (_error) {
      body.replaceChildren(
        element("p", "muted", "Could not load the pipeline artifacts right now."),
      );
    }
  }

  function getPath(value, path) {
    let current = value;
    for (const key of path.split(".")) {
      if (!current || typeof current !== "object") return undefined;
      current = current[key];
    }
    return current;
  }

  function stageDetail(stage, data) {
    const parts = [];
    const add = (label, path, fallback) => {
      const value = getPath(data, path);
      const shown = value === undefined || value === null || value === "" ? fallback : value;
      if (shown !== undefined && shown !== null && shown !== "") {
        parts.push(`${label}: ${typeof shown === "number" ? number(shown) : shown}`);
      }
    };

    if (stage === "load") {
      add("Fetched", "summary.ok");
      add("Failed", "summary.failed");
      add("Total", "summary.total");
      add("Inventory", "inventory");
    } else if (stage === "chunk") {
      add("Chunks", "stats.total_chunks");
      add("Sources", "stats.source_count");
      add("Avg tokens", "stats.avg_tokens");
    } else if (stage === "embed") {
      add("Model", "model");
      add("Dimension", "dimension");
      add("Chunks", "chunk_count");
    } else if (stage === "store") {
      add("Collection", "collection");
      add("Vectors", "count");
      add("Distance", "distance");
      add("Corpus", "corpus_version");
    } else if (stage === "retrieve") {
      add("Top K", "top_k");
      add("Threshold", "score_threshold");
      add("Context chunks", "max_context_chunks");
      add("Traces", "traces");
    }
    return parts;
  }

  function renderPipeline(body, data, health) {
    body.replaceChildren();
    const counts = (data && data.counts) || {};

    if (health) {
      const healthItem = element("div", "source-item");
      const healthLeft = element("div", "");
      healthLeft.append(element("h3", "", "Assistant health"));
      healthLeft.append(
        element(
          "p",
          "",
          `${health.provider || "provider"} \u00b7 ${health.model || "model"}`,
        ),
      );
      const warnings = Array.isArray(health.warnings) ? health.warnings : [];
      const meta = element("div", "source-meta");
      const pieces = [
        health.collection_count !== undefined
          ? `${number(health.collection_count)} vectors`
          : "",
        health.corpus_version || "",
        warnings.length ? `${warnings.length} warning${warnings.length === 1 ? "" : "s"}` : "No warnings",
      ].filter(Boolean);
      pieces.forEach((piece) => meta.append(metaItem(piece)));
      healthLeft.append(meta);
      if (health.next_step) {
        healthLeft.append(element("p", "", health.next_step));
      }
      healthItem.append(healthLeft);
      healthItem.append(
        element(
          "span",
          `status-pill${health.status === "ok" ? "" : " degraded"}`,
          health.status || "unknown",
        ),
      );
      body.append(healthItem);
    }

    const countGrid = element("div", "count-grid");
    const countLabels = [
      ["sources", "Sources"],
      ["chunks", "Chunks"],
      ["embeddings", "Embeddings"],
      ["vectors", "Vectors"],
      ["retrieval_traces", "Retrieval traces"],
    ];
    countLabels.forEach(([key, label]) => {
      const card = element("div", "count-card");
      card.append(element("strong", "", number(counts[key])));
      card.append(element("span", "", label));
      countGrid.append(card);
    });
    body.append(countGrid);

    body.append(element("h3", "modal-subheading", "Pipeline stages"));
    const stageList = element("div", "stage-list");
    Object.entries((data && data.stages) || {}).forEach(([stage, stageData]) => {
      const item = element("div", "stage-item");
      item.append(element("h3", "", stage));
      const details = stageDetail(stage, stageData);
      item.append(element("p", "", details.length ? details.join(" \u00b7 ") : "Artifact available."));
      const generated = stageData && stageData.generated_at;
      if (generated) {
        item.append(element("p", "", `Generated ${formatDate(generated)}`));
      }
      const raw = element("details", "proof");
      raw.append(element("summary", "", "View stage JSON"));
      raw.append(
        element("blockquote", "", JSON.stringify(stageData, null, 2).slice(0, 6000)),
      );
      item.append(raw);
      stageList.append(item);
    });
    body.append(stageList);

    const samples = Array.isArray(data && data.sample_chunks) ? data.sample_chunks : [];
    if (samples.length) {
      body.append(element("h3", "modal-subheading", "Sample indexed chunks"));
      const sampleList = element("div", "sample-list");
      samples.forEach((sample) => {
        const item = element("div", "sample-item");
        const title = [sample.topic, sample.section].filter(Boolean).join(" \u00b7 ");
        item.append(element("h3", "", title || sample.chunk_id || "Chunk"));
        const text = element("p", "sample-text", (sample.text || "").trim());
        item.append(text);
        const meta = element("div", "source-meta");
        [sample.publisher, sample.scheme, sample.chunk_id].filter(Boolean).forEach((piece) => {
          const node = metaItem(piece);
          if (node) meta.append(node);
        });
        item.append(meta);
        sampleList.append(item);
      });
      body.append(sampleList);
    }
  }

  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const value = chip.dataset.scheme || null;
      setSelectedScheme(state.selectedScheme === value ? null : value);
      input.focus({ preventScroll: true });
    });
  });

  function bindExample(example) {
    example.addEventListener("click", () => {
      setSelectedScheme(null);
      const question = example.dataset.question || example.textContent || "";
      input.value = question;
      input.focus({ preventScroll: true });
    });
  }

  async function loadExamples() {
    const grid = document.querySelector(".example-grid");
    if (!grid) return;
    try {
      const response = await fetch("/api/suggestions");
      if (!response.ok) return;
      const data = await response.json();
      const questions = Array.isArray(data.suggestions)
        ? data.suggestions.filter((item) => typeof item === "string" && item.trim())
        : [];
      if (!questions.length) return;
      const buttons = questions.slice(0, 6).map((question) => {
        const button = element("button", "example", question);
        button.type = "button";
        button.dataset.question = question;
        bindExample(button);
        return button;
      });
      grid.replaceChildren(...buttons);
    } catch (_error) {
      return;
    }
  }

  document.querySelectorAll("[data-question]").forEach(bindExample);

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    ask(input.value);
  });

  document.querySelectorAll("[data-open-modal]").forEach((button) => {
    button.addEventListener("click", () => openModal(button.dataset.openModal));
  });

  document.querySelectorAll("[data-close-modal]").forEach((control) => {
    control.addEventListener("click", () => {
      closeModal(control.closest(".modal"));
    });
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      document.querySelectorAll(".modal:not([hidden])").forEach(closeModal);
    }
  });

  async function loadDisclaimer() {
    const disclaimer = document.getElementById("disclaimer");
    if (!disclaimer) return;
    try {
      const response = await fetch("/health");
      if (!response.ok) return;
      const data = await response.json();
      if (typeof data.disclaimer === "string" && data.disclaimer.trim()) {
        disclaimer.textContent = data.disclaimer.trim();
      }
    } catch (_error) {
      return;
    }
  }

  loadExamples();
  loadDisclaimer();
  showEmpty();
})();
