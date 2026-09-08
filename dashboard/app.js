/**
 * NETRABOT AOI — Multi-Model Inspection Intelligence Dashboard
 *
 * Each "model" is a part type (e.g. WARN-LABEL-001) with its own set of
 * golden master reference images, sample images, spec, and inspection results.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ── State ───────────────────────────────────────────────
  let modelsList = [];
  let activeModelName = null;
  let activeModelData = null;
  let detailModelName = null;

  let currentPartIndex = 0;
  let currentLayer = "annotated";
  let activeSeverityFilter = "ALL";
  let selectedDefectId = null;
  let hoveredDefectId = null;

  let zoom = 1.0, panX = 0, panY = 0;
  let isDragging = false, dragStartX = 0, dragStartY = 0;

  // ── DOM Elements ───────────────────────────────────────────────
  const $ = id => document.getElementById(id);
  const galleryOverlay     = $("model-gallery-overlay");
  const detailOverlay      = $("model-detail-overlay");
  const modelGrid          = $("model-grid");
  const partsListContainer = $("parts-list-container");
  const partsCountBadge    = $("parts-count");
  const baseImg            = $("base-image");
  const overlayCanvas      = $("overlay-canvas");
  const ctx = overlayCanvas.getContext("2d");
  const canvasStage        = $("canvas-stage");
  const canvasViewport     = $("canvas-viewport");
  const zoomLevelText      = $("zoom-level-text");
  const hoverInspectPill   = $("hover-inspect-pill");
  const defectsListContainer  = $("defects-list-container");
  const detectorContainer  = $("detector-activity-container");
  const roiBadgesContainer = $("roi-badges-container");
  const formulaDetail      = $("selected-defect-saliency-breakdown");

  // ═════════════════════════════════════════════════════════════════════════
  //  MODEL GALLERY
  // ═════════════════════════════════════════════════════════════════════════
  function showGallery()  { galleryOverlay.classList.add("visible");    refreshModelsList(); }
  function hideGallery()  { galleryOverlay.classList.remove("visible"); }
  function showDetail()   { detailOverlay.classList.add("visible"); }
  function hideDetail()   { detailOverlay.classList.remove("visible"); }

  async function refreshModelsList() {
    const resp = await fetch("/api/models");
    const data = await resp.json();
    modelsList = data.models || [];
    renderModelGrid();
  }

  function renderModelGrid() {
    if (modelsList.length === 0) {
      modelGrid.innerHTML = `<div style="grid-column:1/-1;padding:2rem;text-align:center;color:#64748b;">
        No models yet. Click <strong>New Model</strong> to create one with its own golden master images.
      </div>`;
      return;
    }
    modelGrid.innerHTML = modelsList.map(m => `
      <div class="model-card ${m.name === activeModelName ? 'active-model-card' : ''}" data-model="${m.name}">
        ${m.thumbnail_url
          ? `<div class="model-thumb"><img src="${m.thumbnail_url}" alt="${m.name}"></div>`
          : `<div class="model-thumb-placeholder">📦</div>`
        }
        <div class="model-card-body">
          <div class="model-card-title" title="${m.name}">${m.name}</div>
          <div class="model-card-stats">
            <span class="model-stat"><span class="model-stat-icon">🌟</span>${m.golden_count}</span>
            <span class="model-stat"><span class="model-stat-icon">📷</span>${m.sample_count}</span>
            <span class="model-stat"><span class="model-stat-icon">📋</span>${m.report_count}</span>
          </div>
          <div style="display:flex;gap:0.4rem;margin-top:0.6rem;">
            <button class="btn-card-select" data-model="${m.name}" style="flex:1;padding:0.4rem;font-size:0.75rem;border-radius:6px;border:none;background:#06b6d4;color:#0f172a;font-weight:600;cursor:pointer;">Select</button>
            <button class="btn-card-manage" data-model="${m.name}" style="flex:1;padding:0.4rem;font-size:0.75rem;border-radius:6px;border:1px solid #334155;background:#1e293b;color:#cbd5e1;cursor:pointer;">Manage</button>
          </div>
        </div>
        <button class="model-card-delete" data-model="${m.name}" title="Delete model">✕</button>
      </div>
    `).join("");

    // Select button → activate model and close modal
    modelGrid.querySelectorAll(".btn-card-select").forEach(btn => {
      btn.addEventListener("click", async e => {
        e.stopPropagation();
        await activateModel(btn.dataset.model);
        hideGallery();
      });
    });

    // Manage button → open model detail modal
    modelGrid.querySelectorAll(".btn-card-manage").forEach(btn => {
      btn.addEventListener("click", e => {
        e.stopPropagation();
        openModelDetail(btn.dataset.model);
      });
    });

    // Card click → activate model directly
    modelGrid.querySelectorAll(".model-card").forEach(el => {
      el.addEventListener("click", async e => {
        if (e.target.closest(".model-card-delete") || e.target.closest(".btn-card-manage") || e.target.closest(".btn-card-select")) return;
        await activateModel(el.dataset.model);
        hideGallery();
      });
    });

    // Delete click
    modelGrid.querySelectorAll(".model-card-delete").forEach(btn => {
      btn.addEventListener("click", async e => {
        e.stopPropagation();
        const name = btn.dataset.model;
        await deleteModel(name);
      });
    });
  }

  async function deleteModel(name) {
    if (!name) return;
    if (!confirm(`Are you sure you want to delete model "${name}" and all its golden images, samples, and reports?`)) return;
    try {
      const resp = await fetch(`/api/models/${encodeURIComponent(name)}`, { method: "DELETE" });
      const data = await resp.json();
      if (data.status === "deleted") {
        showToast(`Model "${name}" deleted successfully`, "info");
        if (activeModelName === name) {
          activeModelName = null;
          activeModelData = null;
          updateActiveModelUI();
        }
        hideDetail();
        await refreshModelsList();
      } else {
        showToast(`Failed to delete model: ${data.error || "Unknown error"}`, "error");
      }
    } catch (err) {
      showToast(`Delete error: ${err.message}`, "error");
    }
  }

  // ═════════════════════════════════════════════════════════════════════════
  //  MODEL DETAIL (golden + samples management)
  // ═════════════════════════════════════════════════════════════════════════
  async function openModelDetail(modelName) {
    detailModelName = modelName;
    hideGallery();
    showDetail();
    $("detail-model-title").textContent = modelName;
    $("detail-model-desc").textContent = "Manage golden master images and samples for this model";
    await refreshModelDetail();
  }

  async function refreshModelDetail() {
    const resp = await fetch(`/api/models/${encodeURIComponent(detailModelName)}`);
    const data = await resp.json();
    $("golden-count").textContent = data.golden_images?.length || 0;
    $("sample-count").textContent = data.sample_images?.length || 0;

    // Render golden thumbs
    renderThumbGrid(
      $("golden-thumb-grid"),
      data.golden_images || [],
      name => `/data/models/${encodeURIComponent(detailModelName)}/golden/${encodeURIComponent(name)}`,
      name => deleteGoldenImage(detailModelName, name)
    );

    // Render sample thumbs
    renderThumbGrid(
      $("samples-thumb-grid"),
      data.sample_images || [],
      name => `/data/models/${encodeURIComponent(detailModelName)}/samples/${encodeURIComponent(name)}`,
      name => deleteSampleImage(detailModelName, name)
    );
  }

  function renderThumbGrid(container, fileNames, urlFn, deleteFn) {
    if (!fileNames.length) {
      container.innerHTML = `<div style="grid-column:1/-1;padding:0.5rem;color:#64748b;font-size:0.75rem;">No images uploaded yet.</div>`;
      return;
    }
    container.innerHTML = fileNames.map(name => `
      <div class="image-thumb-item" title="${name}">
        <img src="${urlFn(name)}" alt="${name}" loading="lazy">
        <div class="thumb-label">${name}</div>
        ${deleteFn ? `<button class="thumb-delete-btn" data-name="${name}" title="Remove image">✕</button>` : ""}
      </div>
    `).join("");

    if (deleteFn) {
      container.querySelectorAll(".thumb-delete-btn").forEach(btn => {
        btn.addEventListener("click", e => {
          e.stopPropagation();
          deleteFn(btn.dataset.name);
        });
      });
    }
  }

  async function deleteGoldenImage(modelName, fileName) {
    if (!confirm(`Delete golden master image "${fileName}"?`)) return;
    try {
      const resp = await fetch(`/api/models/${encodeURIComponent(modelName)}/golden/${encodeURIComponent(fileName)}`, { method: "DELETE" });
      const data = await resp.json();
      if (data.status === "deleted") {
        showToast(`Deleted golden image "${fileName}"`, "info");
        await refreshModelDetail();
        if (activeModelName === modelName) {
          await activateModel(modelName);
        }
      } else {
        showToast(`Failed to delete: ${data.error || "Unknown error"}`, "error");
      }
    } catch (e) {
      showToast(`Delete error: ${e.message}`, "error");
    }
  }

  async function deleteSampleImage(modelName, fileName) {
    if (!confirm(`Delete sample image "${fileName}"?`)) return;
    try {
      const resp = await fetch(`/api/models/${encodeURIComponent(modelName)}/samples/${encodeURIComponent(fileName)}`, { method: "DELETE" });
      const data = await resp.json();
      if (data.status === "deleted") {
        showToast(`Deleted sample image "${fileName}"`, "info");
        await refreshModelDetail();
        if (activeModelName === modelName) {
          await activateModel(modelName);
        }
      } else {
        showToast(`Failed to delete: ${data.error || "Unknown error"}`, "error");
      }
    } catch (e) {
      showToast(`Delete error: ${e.message}`, "error");
    }
  }

  function showToast(msg, type = "info") {
    let toast = document.getElementById("netrabot-toast");
    if (!toast) {
      toast = document.createElement("div");
      toast.id = "netrabot-toast";
      document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.className = `netrabot-toast ${type} show`;
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => toast.classList.remove("show"), 3500);
  }

  // Upload Helpers
  function setupUploadZone(zoneId, inputId, subfolder) {
    const zone = $(zoneId);
    const input = $(inputId);

    zone.addEventListener("click", e => {
      if (e.target.tagName === "LABEL" || e.target.closest("label")) return;
      input.click();
    });
    zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("dragover"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
    zone.addEventListener("drop", e => {
      e.preventDefault();
      zone.classList.remove("dragover");
      uploadFiles(e.dataTransfer.files, subfolder);
    });
    input.addEventListener("change", () => {
      uploadFiles(input.files, subfolder);
      input.value = "";
    });
  }

  async function uploadFiles(fileList, subfolder, targetModel = null) {
    const model = targetModel || detailModelName || activeModelName;
    if (!model || !fileList.length) {
      showToast("Please select or open a model first.", "error");
      return;
    }
    showToast(`Uploading ${fileList.length} image(s)...`, "info");
    const formData = new FormData();
    for (const f of fileList) formData.append("files", f);
    try {
      const resp = await fetch(`/api/models/${model}/upload/${subfolder}`, {
        method: "POST",
        body: formData,
      });
      const data = await resp.json();
      if (data.error) {
        showToast(`Upload failed: ${data.error}`, "error");
        return;
      }
      showToast(`Uploaded ${data.files?.length || fileList.length} image(s) to ${model}!`, "success");
      if (detailModelName) {
        refreshModelDetail();
      }
      if (activeModelName === model) {
        if (subfolder === "samples") {
          showToast(`Running inspection on new sample(s)...`, "info");
          await runInspection();
        } else {
          await activateModel(model);
        }
      }
    } catch (e) {
      showToast(`Upload error: ${e.message}`, "error");
    }
  }

  // ═════════════════════════════════════════════════════════════════════════
  //  ACTIVE MODEL SELECTION & INSPECTION
  // ═════════════════════════════════════════════════════════════════════════
  async function activateModel(modelName) {
    activeModelName = modelName;
    const resp = await fetch(`/api/models/${modelName}`);
    activeModelData = await resp.json();
    updateActiveModelUI();
    hideDetail();

    if (activeModelData.parts && activeModelData.parts.length > 0) {
      currentPartIndex = 0;
      selectedDefectId = null;
      renderPartsList();
      selectPart(0);
    } else {
      partsListContainer.innerHTML = `<div style="padding:0.75rem;font-size:0.78rem;color:#94a3b8;">No inspection reports yet. Click <strong>Re-Run</strong> in the header to detect defects.</div>`;
      if (activeModelData.golden_reference_url || (activeModelData.golden_image_urls && activeModelData.golden_image_urls.length > 0)) {
        const masterUrl = activeModelData.golden_reference_url || activeModelData.golden_image_urls[0];
        baseImg.src = masterUrl;
        baseImg.onload = () => {
          overlayCanvas.width = baseImg.naturalWidth || 950;
          overlayCanvas.height = baseImg.naturalHeight || 700;
          $("foot-dimensions").textContent = `${overlayCanvas.width} × ${overlayCanvas.height} px`;
          drawOverlay();
        };
      } else {
        clearCanvas();
      }
    }
  }

  function updateActiveModelUI() {
    const nameEl = $("active-model-name");
    const thumbEl = $("active-model-thumb");

    if (!activeModelName) {
      nameEl.textContent = "No model selected";
      thumbEl.innerHTML = "🌟";
      $("spec-part-id").textContent = "—";
      $("spec-resolution").textContent = "—";
      return;
    }
    nameEl.textContent = activeModelName;
    const m = modelsList.find(x => x.name === activeModelName);
    if (m && m.thumbnail_url) {
      thumbEl.innerHTML = `<img src="${m.thumbnail_url}" alt="">`;
    } else {
      thumbEl.innerHTML = "🌟";
    }

    if (activeModelData?.spec) {
      $("spec-part-id").textContent = activeModelData.spec.part_id || activeModelName;
      $("spec-resolution").textContent = `${activeModelData.spec.calibration?.mm_per_px || "—"} mm/px`;
      renderRoiBadges(activeModelData.spec.regions || []);
    }
  }

  async function runInspection() {
    if (!activeModelName) { alert("Select a model first."); return; }
    const btn = $("btn-reinspect");
    btn.disabled = true; btn.textContent = "Inspecting…";
    try {
      const resp = await fetch(`/api/models/${activeModelName}/inspect`, { method: "POST" });
      const result = await resp.json();
      if (result.error) { alert("Inspection failed: " + result.error); return; }
      await activateModel(activeModelName);
    } catch (e) {
      alert("Error: " + e.message);
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg> Re-Run`;
    }
  }

  // ═════════════════════════════════════════════════════════════════════════
  //  INSPECTION RESULTS UI (same as before, adapted to activeModelData)
  // ═════════════════════════════════════════════════════════════════════════
  function renderPartsList() {
    if (!activeModelData?.parts?.length) return;
    partsCountBadge.textContent = `${activeModelData.parts.length} Parts`;
    partsListContainer.innerHTML = activeModelData.parts.map((p, idx) => {
      const verdict = p.report.verdict || "PASS";
      const total = p.report.total_candidates || p.report.defects?.length || 0;
      const topSev = p.report.defects?.[0]?.severity || "—";
      return `
        <div class="part-card ${idx === currentPartIndex ? 'active' : ''}" data-index="${idx}">
          <div class="part-card-head">
            <span class="part-name" title="${p.report.file}">${p.report.file}</span>
            <span class="verdict-tag ${verdict.toLowerCase()}">${verdict}</span>
          </div>
          <div class="part-meta-row">
            <span>Candidates: <strong>${total}</strong></span>
            <span>Top: <strong>${topSev}</strong></span>
          </div>
        </div>`;
    }).join("");
    partsListContainer.querySelectorAll(".part-card").forEach(el => {
      el.addEventListener("click", () => selectPart(parseInt(el.dataset.index, 10)));
    });
  }

  function renderRoiBadges(regions) {
    roiBadgesContainer.innerHTML = regions.map(r =>
      `<span class="roi-chip" title="ROI: ${r.box?.join(', ')}">${r.name}</span>`
    ).join("");
  }

  function selectPart(index) {
    currentPartIndex = index; selectedDefectId = null; hoveredDefectId = null;
    partsListContainer.querySelectorAll(".part-card").forEach((el, i) => el.classList.toggle("active", i === index));
    const part = activeModelData.parts[currentPartIndex];
    if (!part) return;

    $("foot-sample-name").textContent = part.report.file;
    $("foot-candidates").textContent = part.report.total_candidates || part.report.defects.length;
    $("foot-shown").textContent = `${part.report.top_n_shown || 15} ranked`;

    const reg = part.report.global_metrics?.registration;
    if (reg) {
      $("tel-shift").textContent = `${reg.shift_px?.toFixed(2)} px (${reg.shift_mm?.toFixed(3)} mm)`;
      $("tel-response").textContent = reg.response?.toFixed(3) || "—";
      const g = $("tel-reg-gate");
      g.textContent = reg.exceeded ? "FAIL" : "PASS";
      g.className = `verdict-tag ${reg.exceeded ? "fail" : "pass"}`;
    }
    const topo = part.report.global_metrics?.topology;
    if (topo) $("tel-topology").textContent = `${topo.contour_count} / ${topo.golden_mean?.toFixed(0)} el`;

    updateImageLayer();
    renderDefectsList();
    renderDetectorActivity(part.report.detector_stats || {});
  }

  function updateImageLayer() {
    const part = activeModelData?.parts?.[currentPartIndex];
    let src = "";
    if (currentLayer === "golden") {
      src = activeModelData?.golden_reference_url || activeModelData?.golden_image_urls?.[0] || (part ? part.sample_image_url : "");
    } else if (part) {
      if (currentLayer === "annotated") src = part.annotated_image_url;
      else if (currentLayer === "sample") src = part.sample_image_url;
      else if (currentLayer === "diff") src = part.heatmap_image_url || part.annotated_image_url;
    }
    if (!src) return;
    baseImg.src = src;
    baseImg.onload = () => {
      overlayCanvas.width = baseImg.naturalWidth || 950;
      overlayCanvas.height = baseImg.naturalHeight || 700;
      $("foot-dimensions").textContent = `${overlayCanvas.width} × ${overlayCanvas.height} px`;
      drawOverlay();
    };
  }

  function clearCanvas() {
    baseImg.src = "";
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    defectsListContainer.innerHTML = "";
    detectorContainer.innerHTML = "";
  }

  function drawOverlay() {
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    const part = activeModelData?.parts?.[currentPartIndex];
    if (!part) return;

    if ($("chk-show-rois").checked && activeModelData.spec?.regions) {
      ctx.lineWidth = 1.5; ctx.setLineDash([4,4]); ctx.strokeStyle = "rgba(6,182,212,0.4)";
      ctx.font = "10px JetBrains Mono"; ctx.fillStyle = "rgba(6,182,212,0.7)";
      activeModelData.spec.regions.forEach(r => {
        if (!r.box) return;
        const [x1,y1,x2,y2] = r.box;
        ctx.strokeRect(x1,y1,x2-x1,y2-y1);
        ctx.fillText(r.name, x1+4, y1+12);
      });
      ctx.setLineDash([]);
    }

    if ($("chk-show-boxes").checked && part.report.defects) {
      const top = part.report.defects.slice(0, part.report.top_n_shown || 15);
      top.forEach((d, idx) => {
        if (!d.w || !d.h) return;
        const isH = d.id === hoveredDefectId, isS = d.id === selectedDefectId;
        let col = "#eab308";
        if (d.severity === "CRITICAL") col = "#ef4444";
        else if (d.severity === "MAJOR") col = "#f97316";
        ctx.lineWidth = isS ? 3 : (isH ? 2.5 : 1.5);
        ctx.strokeStyle = isS ? "#06b6d4" : col;
        const p = 4;
        ctx.strokeRect(d.x-p, d.y-p, d.w+p*2, d.h+p*2);
        if (isS) { ctx.fillStyle = "rgba(6,182,212,0.15)"; ctx.fillRect(d.x-p,d.y-p,d.w+p*2,d.h+p*2); }
        ctx.fillStyle = isS ? "#06b6d4" : col;
        ctx.fillRect(d.x-p,d.y-p-14,30,14);
        ctx.fillStyle = "#000"; ctx.font = "bold 9px JetBrains Mono";
        ctx.fillText(`#${d.rank||idx+1}`, d.x-p+3, d.y-p-3);
      });
    }
  }

  function renderMeasurements(measurements) {
    if (!measurements || !Object.keys(measurements).length) return "";
    const rows = Object.entries(measurements).map(([key, m]) => {
      const unit = m.unit ? ` ${m.unit}` : "";
      const pct = (m.exceeds_by_pct !== undefined && m.exceeds_by_pct !== null)
        ? `<span class="measure-over">+${m.exceeds_by_pct}%</span>` : "";
      return `<div class="measure-row">
        <span class="measure-label">${m.label || key}</span>
        <span class="measure-val">${m.value}${unit} <span class="measure-tol">(tol ${m.threshold}${unit})</span> ${pct}</span>
      </div>`;
    }).join("");
    return `<div class="defect-measurements">
      <div class="evidence-label">WHY FLAGGED — MEASURED PARAMETER vs TOLERANCE</div>
      ${rows}
    </div>`;
  }

  function renderDefectsList() {
    const part = activeModelData?.parts?.[currentPartIndex];
    if (!part?.report?.defects) return;
    let defs = part.report.defects;
    if (activeSeverityFilter !== "ALL") defs = defs.filter(d => d.severity === activeSeverityFilter);
    if (!defs.length) { defectsListContainer.innerHTML = `<div style="padding:0.75rem;font-size:0.75rem;color:#64748b;">No candidates matching filter.</div>`; return; }
    defectsListContainer.innerHTML = defs.map((d,idx) => {
      const rank = d.rank || idx+1, sal = d.saliency||0;
      return `
        <div class="defect-item ${d.id===selectedDefectId?'selected':''}" data-id="${d.id}">
          <div class="defect-item-head">
            <span class="rank-badge">#${rank}</span>
            <span class="sev-badge ${d.severity}">${d.severity}</span>
          </div>
          <div class="defect-type">${d.defect_type||'ANOMALY'}</div>
          <div class="defect-metrics-row">
            <span>Area: <strong>${d.area_mm2?.toFixed(2)} mm²</strong></span>
            <span>Saliency: <strong class="saliency-val">${sal.toFixed(4)}</strong></span>
            <span>ROI: <strong>${d.region||'—'}</strong></span>
          </div>
          <div class="defect-metrics-row">
            <span>Size: <strong>${d.w}×${d.h} px</strong></span>
            <span>Pos: <strong>(${d.x}, ${d.y})</strong></span>
            <span>Seen: <strong>${d.sample_str||'—'}</strong></span>
          </div>
          ${renderMeasurements(d.measurements)}
          ${d.crop_url ? `
          <div class="defect-evidence">
            <div class="evidence-label">MASTER vs SAMPLE vs DIFF</div>
            <img class="evidence-img" src="${d.crop_url.startsWith('/') ? d.crop_url : `/data/models/${activeModelName}/output/${d.crop_url}`}" loading="lazy"
                 alt="Defect #${rank} evidence crop">
          </div>` : ''}
        </div>`;
    }).join("");
    defectsListContainer.querySelectorAll(".defect-item").forEach(el => {
      const id = parseInt(el.dataset.id,10);
      el.addEventListener("click", () => selectDefect(id));
      el.addEventListener("mouseenter", () => { hoveredDefectId=id; drawOverlay(); updateHover(id); });
      el.addEventListener("mouseleave", () => { hoveredDefectId=null; drawOverlay(); hoverInspectPill.textContent="Hover over defects to inspect"; });
    });
  }

  function selectDefect(id) {
    selectedDefectId = id;
    defectsListContainer.querySelectorAll(".defect-item").forEach(el =>
      el.classList.toggle("selected", parseInt(el.dataset.id,10)===id));
    const d = activeModelData.parts[currentPartIndex].report.defects.find(x=>x.id===id);
    if (!d) return;
    const agree = d.confidence||0, at = Math.min(Math.log1p(d.area_mm2)/Math.log1p(50),1),
          pt = Math.min((d.peak_deviation||0)/50,1), fill = (d.w&&d.h)?(d.area_px/(d.w*d.h)):1;
    formulaDetail.innerHTML = `
      <strong>#${d.rank||d.id} (${d.severity}):</strong><br>
      • Agreement: ${(0.45*agree).toFixed(3)} (${(agree*100).toFixed(0)}%)<br>
      • Log Area: ${(0.25*at).toFixed(3)} (${d.area_mm2?.toFixed(2)} mm²)<br>
      • Peak Dev: ${(0.20*pt).toFixed(3)} (${d.peak_deviation})<br>
      • Compactness: ${(0.10*fill).toFixed(3)}<br>
      <strong>Total: ${d.saliency?.toFixed(4)}</strong>`;
    if (d.w && d.h) {
      zoom=2.2; panX=-(d.x+d.w/2-overlayCanvas.width/2)*zoom; panY=-(d.y+d.h/2-overlayCanvas.height/2)*zoom;
      applyTransform();
    }
    drawOverlay(); updateHover(id);
  }

  function topMeasurementText(measurements) {
    if (!measurements) return null;
    const entries = Object.values(measurements);
    if (!entries.length) return null;
    const m = entries.reduce((a, b) => (b.exceeds_by_pct || 0) > (a.exceeds_by_pct || 0) ? b : a);
    const unit = m.unit ? ` ${m.unit}` : "";
    return `${m.label}: ${m.value}${unit} (tol ${m.threshold}${unit})`;
  }

  function updateHover(id) {
    const d = activeModelData.parts[currentPartIndex].report.defects.find(x=>x.id===id);
    if (!d) return;
    const measure = topMeasurementText(d.measurements);
    hoverInspectPill.textContent = `[#${d.rank}] ${d.defect_type} | ${d.area_mm2?.toFixed(2)}mm² | Sal: ${d.saliency?.toFixed(4)} | (${d.x},${d.y})` +
      (measure ? ` | ${measure}` : "");
  }

  function renderDetectorActivity(stats) {
    const ent = Object.entries(stats);
    if (!ent.length) { detectorContainer.innerHTML=`<div style="color:#64748b;font-size:0.75rem;">No telemetry.</div>`; return; }
    detectorContainer.innerHTML = ent.map(([det,s]) => {
      const pct = Math.min(Math.max(s.pct||0,0),100);
      return `<div class="det-bar-item">
        <div class="det-bar-head"><span class="det-name">D: ${det}</span><span class="det-px">${s.flagged_px?.toLocaleString()} px (${pct.toFixed(2)}%)</span></div>
        <div class="det-track"><div class="det-fill" style="width:${Math.max(pct,2)}%"></div></div>
      </div>`;
    }).join("");
  }

  function applyTransform() {
    canvasStage.style.transform = `translate(${panX}px,${panY}px) scale(${zoom})`;
    zoomLevelText.textContent = `${Math.round(zoom*100)}%`;
  }

  // ═════════════════════════════════════════════════════════════════════════
  //  EVENTS SETUP
  // ═════════════════════════════════════════════════════════════════════════
  function setupEvents() {
    // Gallery open/close
    $("btn-open-gallery").addEventListener("click", showGallery);
    $("btn-close-gallery").addEventListener("click", hideGallery);
    galleryOverlay.addEventListener("click", e => { if (e.target === galleryOverlay) hideGallery(); });

    // Create new model
    $("btn-create-model").addEventListener("click", async () => {
      const name = prompt("Enter a name for the new model / part type:");
      if (!name) return;
      await fetch("/api/models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      await refreshModelsList();
      openModelDetail(name.trim().replace(/ /g, "-"));
    });

    // Detail close / back
    $("btn-close-detail").addEventListener("click", hideDetail);
    $("btn-back-to-gallery").addEventListener("click", () => { hideDetail(); showGallery(); });
    detailOverlay.addEventListener("click", e => { if (e.target === detailOverlay) hideDetail(); });

    // Detail tabs
    document.querySelectorAll(".detail-tab").forEach(tab => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".detail-tab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        const which = tab.dataset.tab;
        $("section-golden").classList.toggle("hidden", which !== "golden");
        $("section-samples").classList.toggle("hidden", which !== "samples");
      });
    });

    // Upload zones
    setupUploadZone("golden-upload-zone", "golden-file-input", "golden");
    setupUploadZone("samples-upload-zone", "samples-file-input", "samples");

    // Select & View Model from detail modal
    const activateBtn = $("btn-activate-model");
    if (activateBtn) {
      activateBtn.addEventListener("click", async () => {
        await activateModel(detailModelName);
        hideDetail();
      });
    }

    // Delete model from detail modal
    const deleteCurrentBtn = $("btn-delete-current-model");
    if (deleteCurrentBtn) {
      deleteCurrentBtn.addEventListener("click", async () => {
        if (detailModelName) {
          await deleteModel(detailModelName);
        }
      });
    }

    // Inspect from detail modal
    $("btn-inspect-model").addEventListener("click", async () => {
      await activateModel(detailModelName);
      hideDetail();
      await runInspection();
    });

    // Re-Run from header
    $("btn-reinspect").addEventListener("click", runInspection);

    // Quick Add Samples from header or left panel
    const mainFileInput = $("main-samples-file-input");
    if (mainFileInput) {
      mainFileInput.addEventListener("change", e => {
        if (e.target.files.length) {
          uploadFiles(e.target.files, "samples", activeModelName);
          e.target.value = "";
        }
      });
    }

    const quickAddBtn = $("btn-quick-add-samples");
    if (quickAddBtn) {
      quickAddBtn.addEventListener("click", () => {
        if (!activeModelName) {
          showToast("Please select an active model first.", "error");
          return;
        }
        if (mainFileInput) mainFileInput.click();
      });
    }

    const panelAddBtn = $("btn-panel-add-samples");
    if (panelAddBtn) {
      panelAddBtn.addEventListener("click", () => {
        if (!activeModelName) {
          showToast("Please select an active model first.", "error");
          return;
        }
        if (mainFileInput) mainFileInput.click();
      });
    }

    // Drag and Drop files directly onto parts list or canvas
    [partsListContainer, canvasViewport].forEach(zone => {
      if (!zone) return;
      zone.addEventListener("dragover", e => {
        e.preventDefault();
        zone.classList.add("drop-target-active");
      });
      zone.addEventListener("dragleave", () => {
        zone.classList.remove("drop-target-active");
      });
      zone.addEventListener("drop", e => {
        e.preventDefault();
        zone.classList.remove("drop-target-active");
        if (!activeModelName) {
          showToast("Please select an active model first.", "error");
          return;
        }
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
          uploadFiles(e.dataTransfer.files, "samples", activeModelName);
        }
      });
    });

    // Layer switcher
    document.querySelectorAll(".layer-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".layer-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentLayer = btn.dataset.layer;
        updateImageLayer();
      });
    });

    // Overlay toggles
    $("chk-show-boxes").addEventListener("change", drawOverlay);
    $("chk-show-rois").addEventListener("change", drawOverlay);

    // Zoom
    $("btn-zoom-in").addEventListener("click", () => { zoom = Math.min(zoom*1.25, 6); applyTransform(); });
    $("btn-zoom-out").addEventListener("click", () => { zoom = Math.max(zoom/1.25, 0.4); applyTransform(); });
    $("btn-zoom-reset").addEventListener("click", () => { zoom=1; panX=0; panY=0; applyTransform(); });

    // Pan
    canvasViewport.addEventListener("mousedown", e => { isDragging=true; dragStartX=e.clientX-panX; dragStartY=e.clientY-panY; });
    window.addEventListener("mousemove", e => { if (!isDragging) return; panX=e.clientX-dragStartX; panY=e.clientY-dragStartY; applyTransform(); });
    window.addEventListener("mouseup", () => isDragging=false);
    canvasViewport.addEventListener("wheel", e => { e.preventDefault(); zoom=Math.min(Math.max(zoom*(e.deltaY<0?1.12:0.88),0.4),6); applyTransform(); }, {passive:false});

    // Severity filters
    document.querySelectorAll(".filter-pill").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".filter-pill").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        activeSeverityFilter = btn.dataset.sev;
        renderDefectsList();
      });
    });
  }

  // ═════════════════════════════════════════════════════════════════════════
  //  INIT
  // ═════════════════════════════════════════════════════════════════════════
  async function init() {
    setupEvents();
    // Load models and auto-select first one with reports
    const resp = await fetch("/api/models");
    const data = await resp.json();
    modelsList = data.models || [];
    const withReports = modelsList.find(m => m.report_count > 0);
    if (withReports) {
      await activateModel(withReports.name);
    } else if (modelsList.length > 0) {
      activeModelName = modelsList[0].name;
      updateActiveModelUI();
    }
  }

  init();
});
