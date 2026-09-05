"use strict";
const $ = (selector) => document.querySelector(selector);
const state = {
  dataset: null,
  report: null,
  selected: null,
  tab: "samples",
  limit: 24,
  run: null,
  loading: 0,
  prediction: 0,
  poll: null,
};
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};
const imageUrl = (path, thumbnail = false) =>
  `/api/datasets/${encodeURIComponent(state.dataset)}/image?${new URLSearchParams({ path, thumbnail, v: state.report?.fingerprint || "" })}`;
const friendly = (name) => name.replaceAll("_", " ").replaceAll("-", " ");
const humanGroup = (sample) =>
  sample.split === "train" ? "Training" : "Inspection";
const labelText = (sample) =>
  sample.label === "good" ? "Healthy" : sample.label || "Unlabeled";
const isMatchingRun = () =>
  state.run &&
  state.report &&
  state.run.dataset === state.dataset &&
  state.run.fingerprint === state.report.fingerprint &&
  state.run.method === $("#method").value &&
  state.run.device === $("#device").value;

async function api(url, options = {}) {
  const response = await fetch(url, options);
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(
      "The server returned an unexpected response. Refresh or check the server.",
    );
  }
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "Please check the supplied fields and try again.",
    );
  return data;
}
const post = (url, body) =>
  api(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
function notice(message, error = false) {
  $("#notice").hidden = false;
  $("#notice").textContent = message;
  $("#notice").className = error ? "error" : "";
}
function showTab(tab) {
  state.tab = tab;
  for (const name of ["samples", "checks", "inspection"]) {
    $(`#${name}-panel`).hidden = name !== tab;
    $(`#tab-${name}`).setAttribute("aria-selected", String(name === tab));
  }
}
function visibleSamples() {
  const query = $("#search").value.toLowerCase();
  const group = $("#collection-filter").value;
  return (state.report?.samples || []).filter(
    (s) =>
      s.path.toLowerCase().includes(query) &&
      (group === "all" || `${s.split}/${s.label}` === group),
  );
}
function selectSample(path) {
  state.selected = path;
  for (const card of document.querySelectorAll(".sample-card")) {
    card.classList.toggle("selected", card.dataset.path === path);
    card.setAttribute("aria-pressed", String(card.dataset.path === path));
  }
  renderDetail();
}
function renderGallery() {
  const samples = visibleSamples();
  if (!samples.some((s) => s.path === state.selected))
    state.selected = samples[0]?.path || null;
  const gallery = $("#gallery");
  gallery.replaceChildren();
  $("#filtered-count").textContent = `${samples.length} samples`;
  if (!samples.length) {
    const empty = el("div", "empty-state");
    empty.append(
      el("span", "large-symbol", "▧"),
      el(
        "h3",
        "",
        state.report.samples.length
          ? "No matching samples"
          : "A fresh collection",
      ),
      el(
        "p",
        "",
        state.report.samples.length
          ? "Try another filename or collection filter."
          : "Import healthy reference parts to get started.",
      ),
    );
    gallery.append(empty);
  }
  for (const sample of samples.slice(0, state.limit)) {
    const card = el("button", "sample-card");
    card.type = "button";
    card.dataset.path = sample.path;
    card.title = sample.path;
    card.setAttribute("aria-label", `Review ${sample.path}`);
    const img = el("img", "sample-image");
    img.src = imageUrl(sample.path, true);
    img.alt = sample.path.split("/").at(-1);
    img.loading = "lazy";
    const caption = el("div", "card-caption");
    caption.append(el("strong", "", sample.path.split("/").at(-1)));
    const meta = el("div", "card-meta");
    meta.append(
      el(
        "span",
        `badge ${sample.label === "good" ? "good" : "defect"}`,
        labelText(sample),
      ),
      el("span", "", humanGroup(sample)),
    );
    caption.append(meta);
    card.append(img, caption);
    card.onclick = () => selectSample(sample.path);
    gallery.append(card);
  }
  $("#load-more").hidden = samples.length <= state.limit;
  selectSample(state.selected);
}
function renderDetail() {
  const panel = $("#sample-detail");
  panel.replaceChildren();
  panel.className = "";
  const sample = state.report?.samples.find((s) => s.path === state.selected);
  const samples = visibleSamples();
  const index = samples.findIndex((s) => s.path === state.selected);
  $("#previous").disabled = index <= 0;
  $("#next").disabled = index < 0 || index >= samples.length - 1;
  if (!sample) {
    panel.className = "detail-empty";
    panel.textContent = "Select an image to review its details.";
    return;
  }
  const image = el("img", "detail-image");
  image.src = imageUrl(sample.path);
  image.alt = `Selected sample: ${sample.path}`;
  const body = el("div", "detail-body");
  body.append(
    el("h3", "", sample.path.split("/").at(-1)),
    el(
      "span",
      `badge ${sample.label === "good" ? "good" : "defect"}`,
      labelText(sample),
    ),
  );
  const list = el("dl");
  for (const [label, value] of [
    ["Collection", humanGroup(sample)],
    ["Defect type", labelText(sample)],
    [
      "Dimensions",
      sample.width ? `${sample.width} × ${sample.height} px` : "Unreadable",
    ],
    [
      "Ground truth",
      sample.mask
        ? "Mask available"
        : sample.label === "good"
          ? "Healthy reference"
          : "No mask",
    ],
    ["Checksum", sample.sha256 ? sample.sha256.slice(0, 12) : "Unavailable"],
  ]) {
    const row = el("div");
    row.append(el("dt", "", label), el("dd", "", value));
    list.append(row);
  }
  body.append(list);
  if (sample.mask) {
    const details = el("details", "mask-preview");
    details.append(el("summary", "", "Review defect mask"));
    const mask = el("img");
    mask.src = imageUrl(sample.mask);
    mask.alt = `Ground-truth mask for ${sample.path}`;
    details.append(mask);
    body.append(details);
  }
  if (sample.split === "test") {
    const inspect = el("button", "button secondary", "Inspect this sample →");
    inspect.onclick = () => {
      $("#inspection-sample").value = sample.path;
      showTab("inspection");
    };
    body.append(inspect);
  }
  const path = el("div", "path", sample.path);
  body.append(path);
  panel.append(image, body);
}
function renderChecks() {
  const list = $("#readiness-list");
  list.replaceChildren();
  for (const [key, label] of [
    ["training", "Baseline training"],
    ["inspection", "Visual inspection"],
    ["image_evaluation", "Image ROC AUC"],
    ["pixel_evaluation", "Pixel ROC AUC"],
  ]) {
    const ready = state.report.readiness[key];
    const card = el("article", "check-card");
    card.append(
      el(
        "span",
        `badge ${ready ? "good" : "defect"}`,
        ready ? "✓ Ready" : "○ Needs attention",
      ),
      el("strong", "", label),
    );
    list.append(card);
  }
  const findings = $("#findings");
  findings.replaceChildren();
  if (!state.report.issues.length) {
    const clean = el("div", "empty-state");
    clean.append(
      el("span", "large-symbol", "✓"),
      el("h3", "", "The collection passes all structural checks."),
      el(
        "p",
        "",
        "Review representative captures before making inspection decisions.",
      ),
    );
    findings.append(clean);
  }
  for (const issue of state.report.issues) {
    const item = el("article", `finding ${issue.severity}`);
    item.append(el("span", "", issue.severity === "error" ? "!" : "○"));
    const text = el("div");
    text.append(
      el("strong", "", friendly(issue.code)),
      el("p", "", issue.message),
    );
    if (issue.path) {
      const button = el("button", "", issue.path);
      button.onclick = () => {
        const sample = state.report.samples.find(
          (s) => s.path === issue.path || s.mask === issue.path,
        );
        if (sample) {
          $("#search").value = "";
          $("#collection-filter").value = "all";
          state.selected = sample.path;
          state.limit = Math.max(24, state.report.samples.indexOf(sample) + 1);
          renderGallery();
          showTab("samples");
        } else
          notice(
            `No sample is linked to ${issue.path}. Check the file in your dataset folder.`,
          );
      };
      text.append(button);
    }
    item.append(text);
    findings.append(item);
  }
}
function clearPrediction() {
  state.prediction++;
  const empty = el("div", "empty-state");
  empty.append(
    el("span", "large-symbol", "◎"),
    el("h3", "", "A second look at every part."),
    el(
      "p",
      "",
      "Train a baseline, then select an inspection image to see its anomaly map.",
    ),
  );
  $("#prediction").replaceChildren(empty);
}
function renderReport() {
  const report = state.report;
  $("#dataset-content").hidden = false;
  $("#welcome").hidden = true;
  $("#open-import").disabled = false;
  $("#dataset-title").textContent = friendly(state.dataset);
  $("#breadcrumb").textContent = state.dataset;
  $("#dataset-subtitle").textContent =
    "Build a reliable reference. Review every detail. Inspect with confidence.";
  const train = report.counts["train/good"] || 0;
  const test = report.samples.filter((s) => s.split === "test");
  const good = test.filter((s) => s.label === "good").length;
  const defects = test.filter((s) => s.label && s.label !== "good");
  const masks = defects.filter((s) => s.mask).length;
  $("#train-count").textContent = train;
  $("#test-count").textContent = test.length;
  $("#test-summary").textContent =
    `${good} healthy · ${defects.length} defective`;
  $("#mask-count").textContent = defects.length
    ? `${Math.round((masks / defects.length) * 100)}%`
    : "—";
  $("#mask-summary").textContent = defects.length
    ? `${masks} of ${defects.length} defect images have masks`
    : "No defective images yet";
  $("#ready-label").textContent = report.readiness.inspection
    ? "Ready to inspect"
    : "Needs attention";
  $(".readiness-card").classList.toggle("ready", report.readiness.inspection);
  $("#ready-summary").textContent =
    `${report.issues.length} ${report.issues.length === 1 ? "finding" : "findings"} to review`;
  $("#sample-count").textContent = report.samples.length;
  $("#issue-count").textContent = report.issues.length;
  $("#inventory-meta").textContent =
    `${report.samples.length} samples · Inventory ${report.fingerprint.slice(0, 10)}`;
  const filter = $("#collection-filter");
  const previous = filter.value;
  filter.replaceChildren(new Option("All samples", "all"));
  for (const group of Object.keys(report.counts))
    filter.add(new Option(`${group} (${report.counts[group]})`, group));
  filter.value = [...filter.options].some((o) => o.value === previous)
    ? previous
    : "all";
  const selector = $("#inspection-sample");
  const chosen = selector.value;
  selector.replaceChildren();
  for (const sample of test) selector.add(new Option(sample.path, sample.path));
  if (test.some((s) => s.path === chosen)) selector.value = chosen;
  renderGallery();
  renderChecks();
  renderRun();
}
async function loadDataset(name) {
  const token = ++state.loading;
  state.dataset = name;
  state.report = null;
  state.selected = null;
  $("#dataset-content").hidden = true;
  $("#open-import").disabled = true;
  $("#notice").hidden = true;
  $("#dataset-title").textContent = `Loading ${friendly(name)}…`;
  clearPrediction();
  $("#search").value = "";
  state.limit = 24;
  for (const button of document.querySelectorAll(".dataset-link"))
    button.classList.toggle("active", button.dataset.name === name);
  try {
    const report = await api(
      `/api/datasets/${encodeURIComponent(name)}/report`,
    );
    if (token !== state.loading) return;
    state.report = report;
    renderReport();
  } catch (error) {
    if (token === state.loading) {
      $("#dataset-title").textContent = friendly(name);
      notice(error.message, true);
    }
  }
}
async function loadDatasets(preferred = state.dataset) {
  const data = await api("/api/datasets");
  const nav = $("#datasets");
  nav.replaceChildren();
  for (const name of data.datasets) {
    const button = el("button", "dataset-link", name);
    button.dataset.name = name;
    button.onclick = () => loadDataset(name);
    nav.append(button);
  }
  if (data.datasets.length)
    await loadDataset(
      data.datasets.includes(preferred) ? preferred : data.datasets[0],
    );
  else {
    state.dataset = null;
    state.report = null;
    $("#welcome").hidden = false;
    $("#dataset-content").hidden = true;
    $("#open-import").disabled = true;
    $("#dataset-title").textContent = "Your inspection workspace";
    $("#breadcrumb").textContent = "New workspace";
    nav.append(el("span", "muted", "No collections yet"));
  }
}
async function refreshReport() {
  if (!state.dataset) return loadDatasets();
  const name = state.dataset;
  const token = ++state.loading;
  const report = await api(`/api/datasets/${encodeURIComponent(name)}/report`);
  if (token !== state.loading || name !== state.dataset) return;
  if (report.fingerprint !== state.report?.fingerprint) clearPrediction();
  state.report = report;
  renderReport();
}
function renderRun() {
  const run = state.run;
  const matching = isMatchingRun();
  const busy = run && ["training", "predicting"].includes(run.status);
  $("#train").disabled = !state.report?.readiness.inspection || busy;
  $("#train").textContent =
    run?.status === "training" ? "Training baseline…" : "Train baseline";
  $("#predict").disabled =
    !matching || run.status !== "ready" || !$("#inspection-sample").value;
  let message = state.report?.readiness.inspection
    ? "Ready to train on your healthy reference images."
    : "Add healthy training and inspection images, then resolve quality findings.";
  let status = "";
  if (run && run.dataset === state.dataset) {
    status = run.status;
    message = matching
      ? run.message
      : "Dataset or settings changed. Train a new baseline for this collection.";
    if (run.status === "training")
      message = `Training ${run.method} on ${run.dataset}. You can keep reviewing samples.`;
  } else if (busy)
    message = `A task is running for ${run.dataset}. Wait for it to finish before starting another.`;
  $("#run-status").textContent = message;
  $("#run-status").dataset.status = status;
}
async function pollRun(restore = false) {
  clearTimeout(state.poll);
  try {
    state.run = await api("/api/runs/current");
    if (restore && state.run?.dataset === state.dataset) {
      $("#method").value = state.run.method;
      $("#device").value = state.run.device;
    }
    renderRun();
  } catch (error) {
    notice(error.message, true);
  }
  if (state.run && ["training", "predicting"].includes(state.run.status))
    state.poll = setTimeout(pollRun, 1500);
}
function moveSample(direction) {
  const samples = visibleSamples();
  const index = samples.findIndex((s) => s.path === state.selected);
  const next = samples[index + direction];
  if (next) {
    state.limit = Math.max(state.limit, index + direction + 1);
    state.selected = next.path;
    renderGallery();
  }
}
for (const button of document.querySelectorAll("[data-tab]"))
  button.onclick = () => showTab(button.dataset.tab);
for (const button of document.querySelectorAll("[data-open]"))
  button.onclick = () => {
    const dialog = document.getElementById(button.dataset.open);
    dialog.querySelector(".form-error").textContent = "";
    dialog.showModal();
  };
for (const button of document.querySelectorAll("[data-close]"))
  button.onclick = () => button.closest("dialog").close();
$("#search").oninput = () => {
  state.limit = 24;
  renderGallery();
};
$("#collection-filter").onchange = () => {
  state.limit = 24;
  renderGallery();
};
$("#load-more").onclick = () => {
  state.limit += 24;
  renderGallery();
};
$("#previous").onclick = () => moveSample(-1);
$("#next").onclick = () => moveSample(1);
document.addEventListener("keydown", (event) => {
  if (
    !state.report ||
    state.tab !== "samples" ||
    document.querySelector("dialog[open]") ||
    ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(event.target.tagName)
  )
    return;
  if (["ArrowLeft", "ArrowRight"].includes(event.key)) {
    event.preventDefault();
    moveSample(event.key === "ArrowLeft" ? -1 : 1);
  }
});
$("#refresh").onclick = async () => {
  $("#refresh").disabled = true;
  try {
    await loadDatasets();
    await pollRun();
  } catch (error) {
    notice(error.message, true);
  } finally {
    $("#refresh").disabled = false;
  }
};
$("#export").onclick = () => {
  if (!state.report) return;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(state.report, null, 2)], {
      type: "application/json",
    }),
  );
  const link = el("a");
  link.href = url;
  link.download = `${state.dataset}-manifest.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("#create-form").onsubmit = async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector("[type=submit]");
  button.disabled = true;
  try {
    const data = await post("/api/datasets", {
      name: form.elements.name.value,
    });
    $("#create-dialog").close();
    form.reset();
    await loadDatasets(data.name);
    notice(`Created ${data.name}. Add your first images to get started.`);
  } catch (error) {
    form.querySelector(".form-error").textContent = error.message;
  } finally {
    button.disabled = false;
  }
};
$("#import-purpose").onchange = () => {
  const purpose = $("#import-purpose").value;
  const defect = ["test-defect", "ground_truth"].includes(purpose);
  $("#defect-field").hidden = !defect;
  $("#defect-label").required = defect;
  $("#import-help").textContent =
    purpose === "ground_truth"
      ? "Use binary PNG masks named <image-stem>_mask.png. Dimensions must match the inspection image; white marks defects."
      : purpose === "train"
        ? "Use healthy parts only. Keep separate captures for inspection."
        : "Use separate captures from training to check how well the baseline generalizes.";
};
function summarizeFiles() {
  const files = [...$("#import-files").files];
  const bytes = files.reduce((sum, f) => sum + f.size, 0);
  $("#file-summary").textContent = files.length
    ? `${files.length} files selected · ${(bytes / 1024 / 1024).toFixed(1)} MB`
    : "No files selected";
}
$("#import-files").onchange = summarizeFiles;
$("#drop-zone").ondragover = (event) => {
  event.preventDefault();
  $("#drop-zone").classList.add("dragging");
};
$("#drop-zone").ondragleave = () =>
  $("#drop-zone").classList.remove("dragging");
$("#drop-zone").ondrop = (event) => {
  event.preventDefault();
  $("#drop-zone").classList.remove("dragging");
  $("#import-files").files = event.dataTransfer.files;
  summarizeFiles();
};
$("#import-form").onsubmit = async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const name = state.dataset;
  const button = $("#save-import");
  button.disabled = true;
  button.textContent = "Saving images…";
  try {
    const files = [...$("#import-files").files];
    if (!files.length) throw new Error("Choose one or more images.");
    if (
      files.length > 500 ||
      files.reduce((sum, f) => sum + f.size, 0) > 100 * 1024 * 1024
    )
      throw new Error("Import at most 500 files and 100 MB at a time.");
    const purpose = $("#import-purpose").value;
    const data = new FormData();
    data.set(
      "split",
      purpose === "train"
        ? "train"
        : purpose === "ground_truth"
          ? "ground_truth"
          : "test",
    );
    data.set(
      "label",
      ["test-defect", "ground_truth"].includes(purpose)
        ? $("#defect-label").value
        : "good",
    );
    for (const file of files) data.append("files", file);
    const result = await api(
      `/api/datasets/${encodeURIComponent(name)}/import`,
      { method: "POST", body: data },
    );
    $("#import-dialog").close();
    $("#import-files").value = "";
    summarizeFiles();
    if (state.dataset === name) await refreshReport();
    notice(
      `Saved ${result.imported.length} images to ${name}. Original filenames and image bytes preserved.`,
    );
  } catch (error) {
    form.querySelector(".form-error").textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = "Save images";
  }
};
for (const selector of ["#method", "#device", "#inspection-sample"])
  $(selector).onchange = () => {
    clearPrediction();
    renderRun();
  };
$("#train").onclick = async () => {
  $("#train").disabled = true;
  clearPrediction();
  try {
    state.run = await post("/api/runs", {
      dataset: state.dataset,
      method: $("#method").value,
      device: $("#device").value,
    });
    renderRun();
    pollRun();
  } catch (error) {
    notice(error.message, true);
    renderRun();
  }
};
$("#predict").onclick = async () => {
  const token = ++state.prediction;
  const dataset = state.dataset;
  const path = $("#inspection-sample").value;
  $("#predict").disabled = true;
  $("#prediction").replaceChildren(
    el("div", "empty-state", "Inspecting this sample…"),
  );
  try {
    const result = await post(`/api/runs/${state.run.id}/predict`, { path });
    if (token !== state.prediction || dataset !== state.dataset) return;
    const images = el("div", "prediction-images");
    for (const [key, title] of [
      ["original", "Inspection input"],
      ["heatmap", "Anomaly map"],
      ["overlay", "Overlay"],
    ]) {
      const figure = el("figure");
      const image = el("img");
      image.src = result[key];
      image.alt = `${title}: ${path}`;
      figure.append(image, el("figcaption", "", title));
      images.append(figure);
    }
    const summary = el("div", "score-summary");
    const score = el("div");
    score.append(
      el("div", "eyebrow", "ANOMALY SCORE"),
      el("strong", "", result.score.toFixed(3)),
    );
    const text = el("div");
    text.append(
      el("span", "badge neutral", "REVIEW REQUIRED"),
      el(
        "p",
        "",
        "Colors are scaled per image. Compare scores only within this baseline.",
      ),
      el(
        "p",
        "",
        "Input is resized and center-cropped. Check that your defect remains visible.",
      ),
    );
    summary.append(score, text);
    $("#prediction").replaceChildren(images, summary);
  } catch (error) {
    if (token === state.prediction) {
      $("#prediction").replaceChildren(el("div", "empty-state", error.message));
      notice(error.message, true);
    }
  } finally {
    await pollRun();
  }
};
async function boot() {
  try {
    const health = await api("/api/health");
    $("#connection").textContent = "Server connected";
    $("#version").textContent = `v${health.version}`;
    state.run = await api("/api/runs/current");
    await loadDatasets(state.run?.dataset);
    await pollRun(true);
  } catch (error) {
    $("#connection").textContent = "Connection unavailable";
    $("#connection-dot").style.background = "#d68b68";
    notice(error.message, true);
  }
}
boot();
