const byId = (id) => document.getElementById(id);
const form = byId("generate");
const fileInput = byId("file");
const drop = byId("drop");
const sourceImage = byId("source-image");
const submit = byId("submit");
const mode = byId("mode");
const layout = byId("layout");
const quality = byId("quality");
const list = byId("history-list");
const message = byId("input-message");
let file;
let imageUrl;
let ready = false;
let submitting = false;
let selectedJob;
let historyLimit = 24;
let pollTimer;
let previewPromise;
let previewedId;
let viewer;
let previewVersion = 0;

async function request(path, options) {
  const response = await fetch(path, { ...options, cache: "no-store" });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const detail = typeof data.detail === "string" ? data.detail : "Uygulama isteği tamamlayamadı.";
    throw new Error(detail);
  }
  return response.json();
}

function showMessage(text = "") {
  message.hidden = !text;
  message.textContent = text;
}

function updateSubmit() {
  submit.disabled = !ready || !file || submitting;
  submit.textContent = submitting ? "Sıraya ekleniyor…" : "3D Üret";
}

function updateQuality() {
  const geometry = mode.value === "geometry";
  const reference = quality.querySelector('[value="reference"]');
  reference.disabled = geometry;
  if (geometry && quality.value === "reference") quality.value = "quality";
  byId("studio-note").hidden = quality.value !== "studio";
}

mode.addEventListener("change", updateQuality);
quality.addEventListener("change", updateQuality);

function updateLayout() {
  const wideImage = file && sourceImage.naturalHeight && sourceImage.naturalWidth >= sourceImage.naturalHeight * 1.22;
  const showProfile = layout.value === "turnaround" || (layout.value === "auto" && wideImage);
  byId("profile-field").hidden = !showProfile;
  byId("layout-note").hidden = !showProfile;
}

function chooseFile(next) {
  if (next && (!/\.(png|jpe?g|webp)$/i.test(next.name) || next.size > 20 * 1024 * 1024)) {
    showMessage("20 MB veya daha küçük bir PNG, JPG ya da WebP görseli seçin.");
    return;
  }
  if (imageUrl) URL.revokeObjectURL(imageUrl);
  imageUrl = undefined;
  sourceImage.removeAttribute("src");
  file = next;
  if (next) {
    imageUrl = URL.createObjectURL(next);
    sourceImage.src = imageUrl;
    byId("filename").textContent = next.name;
  }
  sourceImage.hidden = !next;
  byId("drop-content").hidden = Boolean(next);
  byId("selected").hidden = !next;
  if (!next) fileInput.value = "";
  showMessage();
  updateSubmit();
  updateLayout();
}

layout.addEventListener("change", updateLayout);
sourceImage.addEventListener("load", updateLayout);
drop.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => chooseFile(fileInput.files[0]));
byId("clear-file").addEventListener("click", () => chooseFile());
for (const eventName of ["dragenter", "dragover"]) {
  drop.addEventListener(eventName, (event) => {
    event.preventDefault();
    drop.classList.add("dragging");
    event.dataTransfer.dropEffect = "copy";
  });
}
drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.classList.remove("dragging");
  chooseFile(event.dataTransfer.files[0]);
});
window.addEventListener("dragover", (event) => {
  if (event.dataTransfer.types.includes("Files")) event.preventDefault();
});
window.addEventListener("drop", (event) => event.preventDefault());

function emptyPreview(text) {
  byId("empty").hidden = false;
  byId("empty-message").textContent = text;
  byId("view-tools").hidden = true;
}

async function showModel(job) {
  const token = ++previewVersion;
  emptyPreview("3D model yükleniyor…");
  try {
    previewPromise ??= import("./viewer.js").then(({ createViewer }) => createViewer(byId("canvas")));
    viewer = await previewPromise;
    if (token !== previewVersion) return;
    const visible = await viewer.open(`/api/jobs/${job.id}/preview`);
    if (token !== previewVersion || !visible) return;
    byId("empty").hidden = true;
    byId("view-tools").hidden = false;
  } catch (error) {
    if (token === previewVersion) emptyPreview(error.message || "Önizleme açılamadı.");
  }
}

function showJob(job) {
  const changed = selectedJob?.id !== job.id;
  selectedJob = job;
  const finished = job.status === "done";
  const failed = job.status === "failed";
  const base = `/api/jobs/${job.id}`;
  byId("preview-name").hidden = false;
  byId("preview-name").textContent = job.views ? `${job.name} · 3 görünüş, tek model` : job.name;
  byId("reuse-source").hidden = false;
  byId("refine-texture").hidden = !job.can_refine;
  byId("compare-texture").hidden = !finished || !job.refinement_of;
  const referenceStatus = byId("reference-status");
  referenceStatus.hidden = !finished || typeof job.reference_applied !== "boolean";
  referenceStatus.textContent = job.reference_applied
    ? `Kaynak ayrıntıları · ${job.reference_views.length} eşleşen görünüş`
    : "Yüzeyle yeterli eşleşme bulunamadı. Üretilen kaplama kullanıldı.";
  const studioStatus = byId("studio-status");
  studioStatus.hidden = !finished || !job.studio_candidates;
  if (!studioStatus.hidden) {
    const seconds = Math.round(job.elapsed_seconds || 0);
    const elapsed = seconds > 0 ? ` · ${Math.floor(seconds / 60)} dk ${seconds % 60} sn` : "";
    studioStatus.textContent = `Stüdyo · ${job.studio_candidates.length} aday${elapsed}`;
  }
  renderReferences(job, base);
  byId("progress-panel").hidden = finished;
  let operation = job.label;
  if (["shape", "surface"].includes(job.stage) && job.candidate_total > 1) {
    operation += ` · Aday ${job.candidate}/${job.candidate_total}`;
  } else if (job.stage === "paint" && job.paint_step_total) {
    operation += ` · Adım ${job.paint_step}/${job.paint_step_total}`;
  }
  byId("stage").textContent = failed ? job.error : operation;
  byId("progress").value = job.progress;
  byId("percent").textContent = failed ? "" : `% ${job.progress}`;
  byId("generation-log").href = `${base}/log`;
  byId("generation-log").hidden = !failed;
  for (const card of list.children) card.setAttribute("aria-current", String(card.dataset.id === job.id));
  byId("downloads").hidden = !finished;
  if (changed) {
    previewVersion++;
    viewer?.clear();
    previewedId = undefined;
  }
  if (finished) {
    byId("glb").href = `${base}/download/glb`;
    byId("obj").href = `${base}/download/obj`;
    const bundle = byId("obj-zip");
    bundle.href = `${base}/download/obj-zip`;
    bundle.hidden = job.mode !== "pbr";
    if (previewedId !== job.id) {
      previewedId = job.id;
      showModel(job);
    }
  } else {
    emptyPreview(failed ? "Bu üretim tamamlanamadı." : job.label);
  }
}

function renderReferences(job, base) {
  const panel = byId("references");
  const views = job.views || [];
  const key = views.length ? `${job.id}/${views.join(",")}` : "";
  if (panel.dataset.key === key) return;
  panel.dataset.key = key;
  panel.replaceChildren();
  panel.hidden = !views.length;
  const names = { front: "Ön", left: "Yan (sola bakıyor)", right: "Yan (sağa bakıyor)", back: "Arka" };
  for (const view of views) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "reference-view";
    button.title = "Bu açıdan görüntüle";
    button.addEventListener("click", () => viewer?.view(view));
    const img = document.createElement("img");
    img.src = `${base}/views/${view}`;
    img.alt = "";
    button.append(img, names[view]);
    panel.append(button);
  }
}

byId("reuse-source").addEventListener("click", async () => {
  const job = selectedJob;
  if (!job) return;
  try {
    const response = await fetch(`/api/jobs/${job.id}/source`);
    if (!response.ok) throw new Error("Kaynak görsel açılamadı.");
    const blob = await response.blob();
    const name = /\.(png|jpe?g|webp)$/i.test(job.name) ? job.name : `${job.name}.png`;
    layout.value = job.layout || "auto";
    byId("profile").value = job.profile || "right";
    chooseFile(new File([blob], name, { type: "image/png" }));
    drop.focus();
  } catch (error) {
    showMessage(error.message);
  }
});

byId("refine-texture").addEventListener("click", async (event) => {
  const job = selectedJob;
  if (!job?.can_refine) return;
  const button = event.currentTarget;
  button.disabled = true;
  try {
    const refined = await request(`/api/jobs/${job.id}/refine`, { method: "POST" });
    showJob(refined);
    void poll();
  } catch (error) {
    showMessage(error.message);
  } finally {
    button.disabled = false;
  }
});

byId("compare-texture").addEventListener("click", () => {
  if (selectedJob?.refinement_of) void selectHistory(selectedJob.refinement_of);
});

const date = new Intl.DateTimeFormat("tr-TR", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });

function renderHistory(jobs, total) {
  byId("no-history").hidden = total > 0;
  byId("history-count").textContent = total ? `${total} kayıt` : "";
  byId("more").hidden = total <= jobs.length;
  const existing = new Map(Array.from(list.children, (card) => [card.dataset.id, card]));
  for (const job of jobs) {
    let card = existing.get(job.id);
    if (!card) {
      card = document.createElement("button");
      card.type = "button";
      card.className = "job";
      card.dataset.id = job.id;
      const thumbnail = document.createElement("img");
      thumbnail.src = `/api/jobs/${job.id}/thumbnail`;
      thumbnail.alt = "";
      thumbnail.loading = "lazy";
      const label = document.createElement("span");
      label.append(document.createElement("strong"), document.createElement("small"));
      card.append(thumbnail, label);
      card.addEventListener("click", () => selectHistory(job.id));
    }
    card.querySelector("strong").textContent = job.name;
    const finishedLabel = job.quality === "studio" ? "Stüdyo"
      : job.reference_applied ? "Kaynak dokulu" : (job.mode === "pbr" ? "PBR" : "Geometri");
    const status = job.status === "done" ? finishedLabel : job.label;
    const views = job.views ? " · 3 görünüş" : "";
    card.querySelector("small").textContent = `${status}${views} · ${date.format(new Date(job.created_at))}`;
    card.setAttribute("aria-current", String(selectedJob?.id === job.id));
    list.append(card);
    existing.delete(job.id);
  }
  for (const card of existing.values()) card.remove();
}

async function selectHistory(id) {
  try {
    showJob(await request(`/api/jobs/${id}`));
  } catch (error) {
    showMessage(error.message);
  }
}

async function refreshHistory() {
  const items = [];
  let total = 0;
  for (let offset = 0; offset < historyLimit; offset += 100) {
    const page = await request(`/api/jobs?limit=${Math.min(historyLimit - offset, 100)}&offset=${offset}`);
    total = page.total;
    items.push(...page.items);
    if (items.length >= total) break;
  }
  renderHistory(items, total);
  if (!selectedJob && items.length) showJob(items[0]);
  else if (selectedJob) {
    const current = items.find(({ id }) => id === selectedJob.id);
    if (current) showJob(current);
  }
}

async function poll() {
  clearTimeout(pollTimer);
  try {
    if (selectedJob && ["queued", "running"].includes(selectedJob.status)) {
      const { id } = selectedJob;
      const updated = await request(`/api/jobs/${id}`);
      if (selectedJob.id === id) showJob(updated);
    }
    await refreshHistory();
  } catch (error) {
    if (!document.hidden) showMessage(error.message);
  } finally {
    pollTimer = setTimeout(poll, selectedJob && ["queued", "running"].includes(selectedJob.status) ? 1200 : 4000);
  }
}

byId("more").addEventListener("click", () => {
  historyLimit += 24;
  refreshHistory().catch((error) => showMessage(error.message));
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!file || !ready || submitting) return;
  showMessage();
  submitting = true;
  updateSubmit();
  const data = new FormData();
  data.append("image", file);
  data.append("mode", mode.value);
  data.append("quality", quality.value);
  data.append("layout", layout.value);
  data.append("profile", byId("profile").value);
  data.append("remove_background", String(byId("cleanup").checked));
  try {
    showJob(await request("/generate", { method: "POST", body: data }));
    void poll();
  } catch (error) {
    showMessage(error.message);
  } finally {
    submitting = false;
    updateSubmit();
  }
});

byId("wireframe").addEventListener("click", (event) => {
  const active = event.currentTarget.getAttribute("aria-pressed") !== "true";
  event.currentTarget.setAttribute("aria-pressed", String(active));
  viewer?.wireframe(active);
});
byId("reset").addEventListener("click", () => viewer?.reset());
byId("light").addEventListener("input", (event) => viewer?.light(Number(event.target.value)));

request("/api/health").then((health) => {
  ready = health.geometry;
  byId("engine").textContent = ready ? `Yerel · ${health.gpu || "CUDA"}` : "Kurulum gerekiyor";
  byId("engine").parentElement.classList.toggle("ready", ready);
  if (!health.pbr) {
    mode.options[0].disabled = true;
    mode.value = "geometry";
  }
  updateQuality();
  if (!ready) showMessage("Modeli kurmak için kurulum.bat dosyasını çalıştırın.");
  updateSubmit();
}).catch((error) => showMessage(error.message));
void poll();
