const fileEl = document.getElementById("file");
const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");
const leftPill = document.getElementById("leftPill");

const origImg = document.getElementById("origImg");
const origEmpty = document.getElementById("origEmpty");

const spinner = document.getElementById("spinner");
const statusText = document.getElementById("statusText");
const rightPill = document.getElementById("rightPill");

const predLabel = document.getElementById("predLabel");
const predRaw = document.getElementById("predRaw");
const confLevel = document.getElementById("confLevel");
const probText = document.getElementById("probText");

const camImg = document.getElementById("camImg");
const camEmpty = document.getElementById("camEmpty");

function setPill(el, text, cls="") {
  el.textContent = text;
  el.className = "pill " + cls;
}

function resetOutput() {
  spinner.style.display = "none";
  statusText.textContent = "No results yet.";
  setPill(rightPill, "Waiting");
  predLabel.textContent = "--";
  predRaw.textContent = "--";
  confLevel.textContent = "--";
  probText.textContent = "--";
  camImg.style.display = "none";
  camImg.removeAttribute("src");
  camEmpty.style.display = "block";
}

function toPercent(prob) {
  const v = Number(prob);
  if (Number.isNaN(v)) return null;
  return (v <= 1) ? v * 100 : v; // supports 0..1 or 0..100
}

function bucketFromPercent(pct) {
  if (pct >= 80) return { level: "High", cls: "ok" };
  if (pct >= 70) return { level: "Medium", cls: "warn" };
  if (pct >= 50) return { level: "Low", cls: "warn" };
  return { level: "Very Low", cls: "danger" };
}

function labelFromPrediction(p) {
  return (Number(p) === 1) ? "Malignant" : "Benign";
}

function setBusy(isBusy) {
  spinner.style.display = isBusy ? "block" : "none";
  fileEl.disabled = isBusy;
  predictBtn.disabled = isBusy || !fileEl.files?.length;
  clearBtn.disabled = isBusy || !fileEl.files?.length;

  if (isBusy) setPill(leftPill, "Uploading…", "warn");
  else if (fileEl.files?.length) setPill(leftPill, "Ready", "ok");
  else setPill(leftPill, "Idle");
}

fileEl.addEventListener("change", () => {
  resetOutput();
  if (fileEl.files && fileEl.files.length) {
    const url = URL.createObjectURL(fileEl.files[0]);
    origImg.src = url;
    origImg.style.display = "block";
    origEmpty.style.display = "none";
    predictBtn.disabled = false;
    clearBtn.disabled = false;
    setPill(leftPill, "Ready", "ok");
  } else {
    origImg.style.display = "none";
    origEmpty.style.display = "block";
    predictBtn.disabled = true;
    clearBtn.disabled = true;
    setPill(leftPill, "Idle");
  }
});

clearBtn.addEventListener("click", () => {
  fileEl.value = "";
  origImg.style.display = "none";
  origEmpty.style.display = "block";
  setPill(leftPill, "Idle");
  predictBtn.disabled = true;
  clearBtn.disabled = true;
  resetOutput();
});

predictBtn.addEventListener("click", async () => {
  if (!fileEl.files || !fileEl.files.length) return;

  setBusy(true);
  statusText.textContent = "Running inference…";
  setPill(rightPill, "Processing", "warn");

  try {
    const fd = new FormData();
    fd.append("request", fileEl.files[0]); // must match FastAPI param name

    const res = await fetch("/predictions", { method: "POST", body: fd });

    if (!res.ok) {
      const t = await res.text();
      throw new Error(`Server error (${res.status}): ${t}`);
    }

    const data = await res.json();

    const pred = Number(data.prediction);
    const label = data.label ? String(data.label) : labelFromPrediction(pred);

    const pct = toPercent(data.predict_proba);
    if (pct === null) throw new Error("Missing or invalid predict_proba in JSON response.");

    const conf = data.confidence_level
      ? { level: String(data.confidence_level), cls: bucketFromPercent(pct).cls }
      : bucketFromPercent(pct);

    predLabel.textContent = label;
    predRaw.textContent = `Raw prediction: ${pred}`;
    confLevel.textContent = conf.level;
    probText.textContent = `Probability: ${pct.toFixed(1)}%`;

    if (data.cam_image) {
      const camUrl = String(data.cam_image) + (String(data.cam_image).includes("?") ? "&" : "?") + "t=" + Date.now();
      camImg.src = camUrl;
      camImg.style.display = "block";
      camEmpty.style.display = "none";
    } else {
      camImg.style.display = "none";
      camEmpty.style.display = "block";
    }

    statusText.textContent = "Done.";
    setPill(rightPill, "Success", conf.cls);

  } catch (err) {
    console.error(err);
    resetOutput();
    statusText.textContent = err.message || "Something went wrong.";
    setPill(rightPill, "Error", "danger");
  } finally {
    setBusy(false);
  }
});

// init
resetOutput();
setPill(leftPill, "Idle");