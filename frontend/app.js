const state = {
  previousResponseId: null,
  conversationId: "local",
  selectedFiles: new Set(),
};

const elements = {
  chooseFiles: document.querySelector("#choose-files"),
  clearOutput: document.querySelector("#clear-output"),
  connectionStatus: document.querySelector("#connection-status"),
  fileInput: document.querySelector("#file-input"),
  form: document.querySelector("#query-form"),
  messages: document.querySelector("#messages"),
  outputList: document.querySelector("#output-list"),
  outputCount: document.querySelector("#output-count"),
  queryInput: document.querySelector("#query-input"),
  refreshFiles: document.querySelector("#refresh-files"),
  selectedCount: document.querySelector("#selected-count"),
  sendStatus: document.querySelector("#send-status"),
  sendQuery: document.querySelector("#send-query"),
  uploadBox: document.querySelector("#upload-box"),
  uploadCount: document.querySelector("#upload-count"),
  uploadList: document.querySelector("#upload-list"),
  uploadStatus: document.querySelector("#upload-status"),
};

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function appendMessage(role, body, options = {}) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const roleNode = document.createElement("div");
  roleNode.className = "message-role";
  roleNode.textContent = role === "user" ? "You" : role === "error" ? "Error" : "Milo";

  const bodyNode = document.createElement("div");
  bodyNode.className = "message-body";
  bodyNode.textContent = body;

  article.append(roleNode, bodyNode);

  if (options.outputFile) {
    article.append(createDownloadCard(options.outputFile));
  }

  elements.messages.append(article);
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function createDownloadCard(file) {
  const card = document.createElement("a");
  card.className = "message-file";
  card.href = file.download_url;

  const name = document.createElement("span");
  name.className = "message-file-name";
  name.textContent = file.filename;

  const meta = document.createElement("span");
  meta.className = "message-file-meta";
  meta.textContent = `${formatBytes(file.size_bytes)} · Download`;

  card.append(name, meta);
  return card;
}

function updateSelectedCount() {
  const count = state.selectedFiles.size;
  elements.selectedCount.textContent = `${count} file${count === 1 ? "" : "s"} selected`;
}

async function apiFetch(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch {
      // Keep the HTTP status text when the response is not JSON.
    }
    throw new Error(detail);
  }

  return response.json();
}

async function deleteFile(kind, fileId) {
  try {
    await apiFetch(`/local/files/${kind}/${encodeURIComponent(fileId)}`, {
      method: "DELETE",
    });
    if (kind === "uploads") {
      state.selectedFiles.delete(fileId);
      updateSelectedCount();
    }
    await refreshFiles();
  } catch (error) {
    appendMessage("error", error.message);
  }
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

function renderUploads(files) {
  elements.uploadList.innerHTML = "";
  elements.uploadList.classList.toggle("empty", files.length === 0);
  elements.uploadCount.textContent = files.length;

  if (!files.length) {
    elements.uploadList.textContent = "No input files yet.";
    updateSelectedCount();
    return;
  }

  const selectAllByDefault = state.selectedFiles.size === 0;
  for (const file of files) {
    if (selectAllByDefault) {
      state.selectedFiles.add(file.id);
    }

    const row = document.createElement("div");
    row.className = "file-row";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.selectedFiles.has(file.id);
    checkbox.ariaLabel = `Use ${file.filename}`;
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) state.selectedFiles.add(file.id);
      else state.selectedFiles.delete(file.id);
      updateSelectedCount();
    });

    const info = document.createElement("div");
    const name = document.createElement("div");
    name.className = "file-name";
    name.textContent = file.filename;
    const meta = document.createElement("div");
    meta.className = "file-meta";
    meta.textContent = formatBytes(file.size_bytes);
    info.append(name, meta);

    const link = document.createElement("a");
    link.className = "download-link";
    link.href = file.download_url;
    link.textContent = "Download";

    const actions = document.createElement("div");
    actions.className = "file-actions";
    const deleteButton = document.createElement("button");
    deleteButton.className = "delete-file-button";
    deleteButton.type = "button";
    deleteButton.title = `Delete ${file.filename}`;
    deleteButton.ariaLabel = `Delete ${file.filename}`;
    deleteButton.innerHTML = `
      <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <path d="M3 6h18"></path>
        <path d="M8 6V4h8v2"></path>
        <path d="M19 6l-1 14H6L5 6"></path>
        <path d="M10 11v5"></path>
        <path d="M14 11v5"></path>
      </svg>
    `;
    deleteButton.addEventListener("click", () => deleteFile("uploads", file.id));
    actions.append(link, deleteButton);

    row.append(checkbox, info, actions);
    elements.uploadList.append(row);
  }

  updateSelectedCount();
}

function renderOutputs(files) {
  elements.outputList.innerHTML = "";
  elements.outputList.classList.toggle("empty", files.length === 0);
  elements.outputCount.textContent = files.length;

  if (!files.length) {
    elements.outputList.textContent = "No Milo files yet.";
    return;
  }

  for (const file of files) {
    const row = document.createElement("div");
    row.className = "file-row output";

    const info = document.createElement("div");
    const name = document.createElement("div");
    name.className = "file-name";
    name.textContent = file.filename;
    const meta = document.createElement("div");
    meta.className = "file-meta";
    meta.textContent = formatBytes(file.size_bytes);
    info.append(name, meta);

    const link = document.createElement("a");
    link.className = "download-link";
    link.href = file.download_url;
    link.textContent = "Download";

    const actions = document.createElement("div");
    actions.className = "file-actions";
    const deleteButton = document.createElement("button");
    deleteButton.className = "delete-file-button";
    deleteButton.type = "button";
    deleteButton.title = `Delete ${file.filename}`;
    deleteButton.ariaLabel = `Delete ${file.filename}`;
    deleteButton.innerHTML = `
      <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        <path d="M3 6h18"></path>
        <path d="M8 6V4h8v2"></path>
        <path d="M19 6l-1 14H6L5 6"></path>
        <path d="M10 11v5"></path>
        <path d="M14 11v5"></path>
      </svg>
    `;
    deleteButton.addEventListener("click", () => deleteFile("outputs", file.id));
    actions.append(link, deleteButton);

    row.append(info, actions);
    elements.outputList.append(row);
  }
}

async function refreshFiles() {
  const files = await apiFetch("/local/files");
  renderUploads(files.uploads);
  renderOutputs(files.outputs);
}

async function uploadFiles(files) {
  if (!files.length) return;

  elements.uploadStatus.textContent = `Uploading ${files.length} file${files.length === 1 ? "" : "s"}...`;
  elements.chooseFiles.disabled = true;

  try {
    for (const file of files) {
      const contentBase64 = await fileToDataUrl(file);
      const uploaded = await apiFetch("/local/files/upload", {
        method: "POST",
        body: JSON.stringify({
          filename: file.name,
          content_base64: contentBase64,
        }),
      });
      state.selectedFiles.add(uploaded.id);
    }
    elements.uploadStatus.textContent = "Upload complete.";
    await refreshFiles();
  } catch (error) {
    elements.uploadStatus.textContent = "Upload failed.";
    appendMessage("error", error.message);
  } finally {
    elements.chooseFiles.disabled = false;
    elements.fileInput.value = "";
  }
}

function setBusy(isBusy) {
  elements.sendQuery.disabled = isBusy;
  elements.connectionStatus.textContent = isBusy ? "Thinking" : "Ready";
  document.body.classList.toggle("is-busy", isBusy);
}

function setSendStatus(message, isSuccess = false) {
  elements.sendStatus.textContent = message;
  elements.sendStatus.classList.toggle("success", isSuccess);
}

async function sendQuery(message) {
  setBusy(true);
  setSendStatus("Sending...");
  appendMessage("user", message);

  try {
    const result = await apiFetch("/local/chat", {
      method: "POST",
      body: JSON.stringify({
        message,
        conversation_id: state.conversationId,
        previous_response_id: state.previousResponseId,
        file_ids: [...state.selectedFiles],
      }),
    });

    state.previousResponseId = result.response_id;
    const assistantMessage = result.output_file ? "Your file is ready to download." : result.message;
    appendMessage("assistant", assistantMessage, { outputFile: result.output_file });
    setSendStatus("Query sent and saved to Supabase.", true);
    setBusy(false);
    await refreshFiles();
  } catch (error) {
    setSendStatus("Query failed to send.");
    setBusy(false);
    appendMessage("error", error.message);
  } finally {
    elements.queryInput.focus();
  }
}

elements.chooseFiles.addEventListener("click", () => elements.fileInput.click());
elements.fileInput.addEventListener("change", () => uploadFiles([...elements.fileInput.files]));
elements.refreshFiles.addEventListener("click", () => refreshFiles().catch((error) => appendMessage("error", error.message)));
elements.clearOutput.addEventListener("click", () => {
  elements.messages.innerHTML = "";
  state.previousResponseId = null;
  setSendStatus("");
  setBusy(false);
});

for (const eventName of ["dragenter", "dragover"]) {
  elements.uploadBox.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.uploadBox.classList.add("dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  elements.uploadBox.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.uploadBox.classList.remove("dragging");
  });
}

elements.uploadBox.addEventListener("drop", (event) => {
  uploadFiles([...event.dataTransfer.files]);
});

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = elements.queryInput.value.trim();
  if (!message) return;
  elements.queryInput.value = "";
  setSendStatus("");
  sendQuery(message);
});

elements.queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});

refreshFiles().catch((error) => appendMessage("error", error.message));
updateSelectedCount();
