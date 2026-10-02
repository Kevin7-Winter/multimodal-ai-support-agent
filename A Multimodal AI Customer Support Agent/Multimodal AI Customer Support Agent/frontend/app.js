const STORAGE_KEY = "anker-support-case-id";

const state = {
  caseData: null,
  busy: false,
  uploadPreviewUrl: null,
};

const elements = {
  caseId: document.querySelector("#caseId"),
  statusPill: document.querySelector("#statusPill"),
  connectionState: document.querySelector("#connectionState"),
  newCaseButton: document.querySelector("#newCaseButton"),
  handoffButton: document.querySelector("#handoffButton"),
  messageList: document.querySelector("#messageList"),
  chatForm: document.querySelector("#chatForm"),
  messageInput: document.querySelector("#messageInput"),
  sendButton: document.querySelector("#sendButton"),
  uploadButton: document.querySelector("#uploadButton"),
  fileInput: document.querySelector("#fileInput"),
  uploadStatus: document.querySelector("#uploadStatus"),
  uploadPreview: document.querySelector("#uploadPreview"),
  uploadTitle: document.querySelector("#uploadTitle"),
  uploadDetail: document.querySelector("#uploadDetail"),
  uploadSpinner: document.querySelector("#uploadSpinner"),
  riskBanner: document.querySelector("#riskBanner"),
  riskBannerText: document.querySelector("#riskBannerText"),
  caseHeadline: document.querySelector("#caseHeadline"),
  productCategory: document.querySelector("#productCategory"),
  productModel: document.querySelector("#productModel"),
  emotionValue: document.querySelector("#emotionValue"),
  riskLevel: document.querySelector("#riskLevel"),
  nextAction: document.querySelector("#nextAction"),
  intentText: document.querySelector("#intentText"),
  symptomList: document.querySelector("#symptomList"),
  evidenceList: document.querySelector("#evidenceList"),
  completedList: document.querySelector("#completedList"),
  toolResults: document.querySelector("#toolResults"),
  imageEvidence: document.querySelector("#imageEvidence"),
  riskList: document.querySelector("#riskList"),
  forbiddenList: document.querySelector("#forbiddenList"),
  officialSources: document.querySelector("#officialSources"),
  historicalCases: document.querySelector("#historicalCases"),
  handoffDialog: document.querySelector("#handoffDialog"),
  handoffSummary: document.querySelector("#handoffSummary"),
  closeDialogButton: document.querySelector("#closeDialogButton"),
  toast: document.querySelector("#toast"),
};

const statusLabels = {
  collecting: "信息收集",
  diagnosing: "诊断中",
  resolved: "已解决",
  handoff: "转人工",
};

const emotionLabels = {
  low: "平稳",
  medium: "焦虑",
  high: "强烈不满",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function safeUrl(value) {
  if (typeof value !== "string") {
    return null;
  }
  return /^https?:\/\//i.test(value) ? value : null;
}

function basename(path) {
  if (!path) {
    return "";
  }
  return String(path).split(/[\\/]/).pop();
}

function imageUrl(image) {
  const filename = basename(image?.path || "");
  return filename ? `/uploads/${encodeURIComponent(filename)}` : "";
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("is-visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    elements.toast.classList.remove("is-visible");
  }, 3200);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      ...(options.body instanceof FormData
        ? {}
        : { "Content-Type": "application/json" }),
      ...(options.headers || {}),
    },
    ...options,
  });

  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail =
      payload && typeof payload === "object"
        ? payload.detail
        : payload;
    throw new Error(detail || `请求失败：${response.status}`);
  }

  return payload;
}

function setBusy(value) {
  state.busy = value;
  elements.sendButton.disabled = value;
  elements.uploadButton.disabled = value;
  elements.newCaseButton.disabled = value;
  elements.handoffButton.disabled = value;
}

function createListItems(items, emptyText) {
  if (!Array.isArray(items) || !items.length) {
    return `<li>${escapeHtml(emptyText)}</li>`;
  }

  return items
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("");
}

function renderMessage(role, content, citations = []) {
  const isUser = role === "user";
  const article = document.createElement("article");
  article.className = `message ${isUser ? "user" : "assistant"}`;

  const citationsHtml = !isUser && citations.length
    ? `<div class="message-meta">${citations
        .map(
          (item) =>
            `<span class="citation-chip">${escapeHtml(item)}</span>`,
        )
        .join("")}</div>`
    : "";

  article.innerHTML = `
    <span class="message-avatar">
      <i data-lucide="${isUser ? "user-round" : "sparkles"}"></i>
    </span>
    <div class="message-body">
      <p class="message-text">${escapeHtml(content)}</p>
      ${citationsHtml}
    </div>
  `;

  elements.messageList.appendChild(article);
  elements.messageList.scrollTop = elements.messageList.scrollHeight;
  refreshIcons();
}

function renderTranscript() {
  elements.messageList.innerHTML = "";
  const transcript = state.caseData?.transcript || [];

  if (!transcript.length) {
    renderMessage(
      "assistant",
      "您好，请描述产品现象。如果需要核对外观或型号，也可以直接上传图片。",
    );
    return;
  }

  transcript.forEach((turn) => {
    renderMessage(turn.role, turn.content || "");
  });
}

function renderOverview() {
  const data = state.caseData || {};
  const status = data.status || "collecting";

  elements.caseId.textContent = data.case_id || "正在创建";
  elements.statusPill.textContent = statusLabels[status] || status;
  elements.statusPill.className = `status-pill status-${status}`;
  elements.caseHeadline.textContent =
    data.product_model || data.product_category || "售后支持";
  elements.productCategory.textContent =
    data.product_category || "充电宝";
  elements.productModel.textContent =
    data.product_model || "待确认";
  elements.emotionValue.textContent =
    data.emotion || emotionLabels[data.emotion_level] || "未知";
  elements.riskLevel.textContent =
    data.risk_flags?.length ? "高风险" : "未发现";
  elements.nextAction.textContent =
    data.next_action || "请描述产品型号和具体现象";
  elements.intentText.textContent =
    data.intent || "等待用户描述。";

  elements.symptomList.innerHTML = createListItems(
    data.symptoms,
    "暂未记录症状",
  );
  elements.evidenceList.innerHTML = createListItems(
    data.evidence,
    "暂未收集证据",
  );
  elements.completedList.innerHTML = createListItems(
    data.completed_steps,
    "暂无已完成步骤",
  );
  elements.riskList.innerHTML = createListItems(
    data.risk_flags,
    "未发现风险",
  );
  elements.forbiddenList.innerHTML = createListItems(
    data.forbidden_actions,
    "暂无禁止操作",
  );

  const hasRisk = Array.isArray(data.risk_flags) && data.risk_flags.length > 0;
  elements.riskBanner.hidden = !hasRisk;
  elements.riskBannerText.textContent = hasRisk
    ? data.handoff_reason ||
      data.next_action ||
      "请立即停止使用并联系人工售后。"
    : "";
}

function renderImageEvidence() {
  const images = state.caseData?.image_evidence || [];

  if (!images.length) {
    elements.imageEvidence.innerHTML = "";
    return;
  }

  elements.imageEvidence.innerHTML = images
    .map((image) => {
      const analysis = image.analysis || {};
      const riskText =
        analysis.summary ||
        analysis.visible_risk ||
        analysis.description ||
        "暂未生成分析说明";
      const damageText =
        typeof analysis.visible_damage === "string"
          ? analysis.visible_damage
          : Array.isArray(analysis.visible_damage)
            ? analysis.visible_damage.join("、")
            : "未识别";
      const url = imageUrl(image);

      return `
        <article class="image-item">
          ${
            url
              ? `<img src="${escapeHtml(url)}" alt="${escapeHtml(
                  image.filename || "售后图片",
                )}" />`
              : `<div></div>`
          }
          <div class="image-copy">
            <strong>${escapeHtml(image.filename || image.image_id)}</strong>
            <span>状态：${escapeHtml(image.analysis_status || "未知")}</span>
            <span>损坏：${escapeHtml(damageText)}</span>
            <span>${escapeHtml(riskText)}</span>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderOfficialSources() {
  const sources = state.caseData?.official_sources || [];

  if (!sources.length) {
    elements.officialSources.innerHTML = "";
    return;
  }

  elements.officialSources.innerHTML = sources
    .map((source) => {
      const url = safeUrl(source.url);
      return `
        <article class="stack-item">
          <strong>${escapeHtml(source.title || "官方资料")}</strong>
          <p>${escapeHtml(source.content || "")}</p>
          ${
            url
              ? `<a href="${escapeHtml(
                  url,
                )}" target="_blank" rel="noreferrer">查看官方来源</a>`
              : ""
          }
        </article>
      `;
    })
    .join("");
}

function renderHistoricalCases() {
  const cases = state.caseData?.historical_cases || [];

  if (!cases.length) {
    elements.historicalCases.innerHTML = "";
    return;
  }

  elements.historicalCases.innerHTML = cases
    .map(
      (item) => `
        <article class="stack-item">
          <strong>${escapeHtml(
            item["工单号"] || "历史案例",
          )} · ${escapeHtml(item["产品型号"] || "未知型号")}</strong>
          <p>${escapeHtml(item["用户描述"] || "暂无描述")}</p>
          <p>处理：${escapeHtml(
            item["处理结果"] || "暂无处理结果",
          )}</p>
        </article>
      `,
    )
    .join("");
}

function renderToolResults() {
  const results = state.caseData?.tool_results || [];

  if (!results.length) {
    elements.toolResults.innerHTML = "";
    return;
  }

  elements.toolResults.innerHTML = results
    .map(
      (item, index) => `
        <article class="stack-item">
          <strong>工具 ${index + 1}：${escapeHtml(item.tool || "查询")}</strong>
          <details>
            <summary>查看返回结果</summary>
            <pre>${escapeHtml(
              JSON.stringify(item.result, null, 2),
            )}</pre>
          </details>
        </article>
      `,
    )
    .join("");
}

function renderCase() {
  renderOverview();
  renderImageEvidence();
  renderOfficialSources();
  renderHistoricalCases();
  renderToolResults();
  refreshIcons();
}

function refreshIcons() {
  if (window.lucide?.createIcons) {
    window.lucide.createIcons({
      attrs: {
        "stroke-width": 1.9,
      },
    });
  }
}

async function checkConnection() {
  try {
    await api("/health");
    elements.connectionState.classList.add("is-online");
    elements.connectionState.lastChild.textContent = " 后端已连接";
  } catch {
    elements.connectionState.classList.remove("is-online");
    elements.connectionState.lastChild.textContent = " 后端未连接";
  }
}

async function loadOrCreateCase() {
  const existingId = window.localStorage.getItem(STORAGE_KEY);

  if (existingId) {
    try {
      state.caseData = await api(
        `/v1/cases/${encodeURIComponent(existingId)}`,
      );
      renderCase();
      renderTranscript();
      return;
    } catch {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  }

  await createNewCase();
}

async function createNewCase() {
  setBusy(true);

  try {
    state.caseData = await api("/v1/cases", {
      method: "POST",
    });
    window.localStorage.setItem(STORAGE_KEY, state.caseData.case_id);
    renderCase();
    renderTranscript();
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
  }
}

async function sendMessage(event) {
  event.preventDefault();

  const message = elements.messageInput.value.trim();

  if (!message) {
    showToast("请先输入需要咨询的问题");
    elements.messageInput.focus();
    return;
  }

  if (state.busy) {
    showToast("上一条消息仍在处理中，请稍候");
    return;
  }

  if (!state.caseData) {
    showToast("工单连接已断开，请刷新页面后重试");
    await checkConnection();
    return;
  }

  renderMessage("user", message);
  elements.messageInput.value = "";
  elements.messageInput.style.height = "auto";
  setBusy(true);

  try {
    const response = await api("/v1/chat", {
      method: "POST",
      body: JSON.stringify({
        case_id: state.caseData.case_id,
        message,
      }),
    });

    state.caseData = response.case;
    renderCase();
    renderMessage(
      "assistant",
      response.reply || "暂时无法生成回复。",
      response.citations || [],
    );
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
    elements.messageInput.focus();
  }
}

function resetUploadStatus() {
  if (state.uploadPreviewUrl) {
    URL.revokeObjectURL(state.uploadPreviewUrl);
    state.uploadPreviewUrl = null;
  }

  elements.uploadStatus.hidden = true;
  elements.uploadPreview.removeAttribute("src");
  elements.uploadSpinner.hidden = true;
}

async function uploadImage(file) {
  if (!file || !state.caseData || state.busy) {
    return;
  }

  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    showToast("只支持 JPG、PNG 或 WEBP 图片");
    return;
  }

  if (file.size > 8 * 1024 * 1024) {
    showToast("图片不能超过 8 MB");
    return;
  }

  resetUploadStatus();
  state.uploadPreviewUrl = URL.createObjectURL(file);
  elements.uploadPreview.src = state.uploadPreviewUrl;
  elements.uploadTitle.textContent = "正在分析图片";
  elements.uploadDetail.textContent =
    "视觉模型正在读取型号、损坏和风险信息。";
  elements.uploadSpinner.hidden = false;
  elements.uploadStatus.hidden = false;
  setBusy(true);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const result = await api(
      `/v1/cases/${encodeURIComponent(state.caseData.case_id)}/images`,
      {
        method: "POST",
        body: formData,
      },
    );

    state.caseData = await api(
      `/v1/cases/${encodeURIComponent(state.caseData.case_id)}`,
    );
    renderCase();

    const analyzed = result.image?.analysis_status === "analyzed";
    elements.uploadTitle.textContent = analyzed
      ? "图片分析完成"
      : "图片已保存";
    elements.uploadDetail.textContent = analyzed
      ? result.image?.analysis?.summary ||
        "识别结果已经写入当前工单。"
      : "视觉模型未返回结果，图片仍保留在工单中。";
    elements.uploadSpinner.hidden = true;

    renderMessage(
      "assistant",
      analyzed
        ? `图片分析完成：${
            result.image?.analysis?.summary ||
            "结果已经写入当前工单。"
          }`
        : "图片已保存，但暂时没有生成视觉分析结果。",
    );
  } catch (error) {
    elements.uploadTitle.textContent = "图片处理失败";
    elements.uploadDetail.textContent = error.message;
    elements.uploadSpinner.hidden = true;
    showToast(error.message);
  } finally {
    setBusy(false);
    elements.fileInput.value = "";
  }
}

async function handoffCase() {
  if (!state.caseData || state.busy) {
    return;
  }

  setBusy(true);

  try {
    const response = await api(
      `/v1/cases/${encodeURIComponent(state.caseData.case_id)}/handoff`,
      {
        method: "POST",
      },
    );

    state.caseData = await api(
      `/v1/cases/${encodeURIComponent(state.caseData.case_id)}`,
    );
    elements.handoffSummary.textContent = response.summary;
    renderCase();
    elements.handoffDialog.showModal();
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(false);
  }
}

function setupTabs() {
  document.querySelectorAll(".case-tab").forEach((button) => {
    button.addEventListener("click", () => {
      const tab = button.dataset.tab;

      document.querySelectorAll(".case-tab").forEach((item) => {
        item.classList.toggle("is-active", item === button);
      });

      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.classList.toggle(
          "is-active",
          panel.dataset.panel === tab,
        );
      });
    });
  });
}

function setupComposer() {
  elements.messageInput.addEventListener("input", () => {
    elements.messageInput.style.height = "auto";
    elements.messageInput.style.height = `${Math.min(
      elements.messageInput.scrollHeight,
      130,
    )}px`;
  });

  elements.messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      elements.chatForm.requestSubmit();
    }
  });
}

function setupEvents() {
  elements.chatForm.addEventListener("submit", sendMessage);
  elements.newCaseButton.addEventListener("click", createNewCase);
  elements.handoffButton.addEventListener("click", handoffCase);
  elements.uploadButton.addEventListener("click", () => {
    elements.fileInput.click();
  });
  elements.fileInput.addEventListener("change", (event) => {
    const [file] = event.target.files || [];
    uploadImage(file);
  });
  elements.closeDialogButton.addEventListener("click", () => {
    elements.handoffDialog.close();
  });
  elements.handoffDialog.addEventListener("click", (event) => {
    if (event.target === elements.handoffDialog) {
      elements.handoffDialog.close();
    }
  });
}

async function init() {
  refreshIcons();
  setupTabs();
  setupComposer();
  setupEvents();
  await checkConnection();
  await loadOrCreateCase();
  window.setInterval(checkConnection, 30000);
}

document.addEventListener("DOMContentLoaded", init);
