/**
 * VoxGaze AAC MVP: Frontend Client Controller
 *
 * Manages accessible word board interaction, simulated gaze movement,
 * dwell timing rendering, phrase bar updates, offline speech synthesis,
 * guided demo walkthrough, and honest calibration gate enforcement.
 */

(function () {
  "use strict";

  // State
  const state = {
    mode: "DEMO", // "DEMO" or "LIVE"
    tiles: [],
    phraseWords: [],
    phraseText: "",
    pointer: { x: 0.50, y: 0.50 },
    dwell: { state: "IDLE", progress: 0.0, activeTile: null, candidateTile: null },
    telemetry: {},
    demoTour: {
      active: false,
      stepIndex: 0,
      steps: [
        { target: "I", title: "Move Indicator to 'I'", desc: "Use Arrow keys or WASD to navigate the gaze pointer onto the 'I' tile." },
        { target: "I", title: "Select 'I'", desc: "Press Space or hold the indicator over 'I' to add it to your phrase." },
        { target: "want", title: "Select 'want'", desc: "Move the indicator onto 'want' and select it." },
        { target: "water", title: "Select 'water'", desc: "Move the indicator onto 'water' and select it." },
        { target: "REVIEW", title: "Review Assembled Phrase", desc: "Look at the phrase bar: 'I want water.' Notice how words are verified before speech." },
        { target: "SPEAK", title: "Press 'SPEAK'", desc: "Press the SPEAK button (or press 'S' / Enter) to synthesize offline speech." }
      ]
    },
    camera: {
      active: false,
      stream: null,
      state: "INACTIVE", // INACTIVE, INITIALIZING, AVAILABLE, FACE_DETECTED, FACE_MISSING, ERROR, PERMISSION_DENIED
      errorMessage: null,
    },
    calibration: {
      running: false,
      currentTargetIndex: -1,
      targets: [
        { name: "CENTER", x: 0.50, y: 0.50 },
        { name: "LEFT",   x: 0.15, y: 0.50 },
        { name: "RIGHT",  x: 0.85, y: 0.50 },
        { name: "UP",     x: 0.50, y: 0.15 },
        { name: "DOWN",   x: 0.50, y: 0.85 },
      ],
      phase: "IDLE", // IDLE, SETTLING, RECORDING, TARGET_COMPLETE, SEQUENCE_COMPLETE, ERROR
      settlingMs: 500,
      recordingMs: 1500,
      phaseStartTime: 0,
      samplesCollected: 0,
    },
  };

  // DOM Elements
  const el = {
    tilesGrid: document.getElementById("tiles-grid"),
    phraseText: document.getElementById("phrase-text"),
    phrasePlaceholder: document.getElementById("phrase-placeholder"),
    candidateTag: document.getElementById("phrase-preview-candidate"),
    candidateWord: document.getElementById("candidate-word-text"),
    btnSpeak: document.getElementById("btn-speak"),
    btnBackspace: document.getElementById("btn-backspace"),
    btnClear: document.getElementById("btn-clear"),
    btnModeDemo: document.getElementById("btn-mode-demo"),
    btnModeLive: document.getElementById("btn-mode-live"),
    btnStartDemo: document.getElementById("btn-start-demo"),
    btnExitTour: document.getElementById("btn-exit-tour"),
    tourCard: document.getElementById("demo-tour-card"),
    tourStepBadge: document.getElementById("tour-step-badge"),
    tourStepTitle: document.getElementById("tour-step-title"),
    tourStepDesc: document.getElementById("tour-step-desc"),
    tourProgressFill: document.getElementById("tour-progress-fill"),
    modeNoticeBanner: document.getElementById("mode-notice-banner"),
    bannerTitle: document.getElementById("banner-title"),
    bannerDesc: document.getElementById("banner-desc"),
    bannerIcon: document.getElementById("banner-icon"),
    gateWarningPill: document.getElementById("live-gate-warning"),
    gazeReticle: document.getElementById("gaze-reticle"),
    reticleDwellRing: document.getElementById("reticle-dwell-ring"),
    toastContainer: document.getElementById("toast-container"),
    // Telemetry DOM
    valSelectionStatus: document.getElementById("val-selection-status"),
    valTrackingQuality: document.getElementById("val-tracking-quality"),
    valGazeCoords: document.getElementById("val-gaze-coords"),
    valEar: document.getElementById("val-ear"),
    valHeadPose: document.getElementById("val-head-pose"),
    telemetryPill: document.getElementById("telemetry-status-pill"),
    // Live camera & calib
    liveCameraCard: document.getElementById("live-camera-card"),
    webcamVideo: document.getElementById("webcam-video"),
    cameraStatusLine: document.getElementById("camera-status-line"),
    cameraOverlayMsg: document.getElementById("camera-overlay-msg"),
    camStatusBadge: document.getElementById("cam-status-badge"),
    btnToggleCamera: document.getElementById("btn-toggle-camera"),
    btnOpenCalibration: document.getElementById("btn-open-calibration"),
    calibrationModal: document.getElementById("calibration-modal"),
    btnCloseCalib: document.getElementById("btn-close-calib"),
    btnExitCalib: document.getElementById("btn-exit-calib"),
    btnStartCalibSeq: document.getElementById("btn-start-calib-seq"),
    btnRetryCalib: document.getElementById("btn-retry-calib"),
    calibCanvas: document.getElementById("calib-canvas"),
    calibFeedback: document.getElementById("calib-feedback"),
    calibProgressFill: document.getElementById("calib-progress-fill"),
  };

  // Audio tone synthesizer for local auditory cues
  const audioCtx = (typeof AudioContext !== "undefined" || typeof webkitAudioContext !== "undefined")
    ? new (window.AudioContext || window.webkitAudioContext)()
    : null;

  function playTone(freq = 600, duration = 0.08, type = "sine") {
    if (!audioCtx) return;
    try {
      if (audioCtx.state === "suspended") audioCtx.resume();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
      gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
      // Audio autoplay policy fallback
    }
  }

  // -------------------------------------------------------------
  // Initialization & Fetching
  // -------------------------------------------------------------

  async function init() {
    await fetchTiles();
    await fetchStatus();
    await fetchCalibrationStatus();
    setupEventListeners();
    setupWebSocket();
    updateReticlePosition();
    renderBoard();
  }

  async function fetchCalibrationStatus() {
    try {
      const res = await fetch("/api/calibration/status");
      if (res.ok) {
        const data = await res.json();
        updateGateStatusDisplay(data);
      }
    } catch (e) {}
  }

  function updateGateStatusDisplay(data) {
    const gates = data.gates || {};
    const dirMap = {
      LEFT: document.getElementById("gate-left"),
      RIGHT: document.getElementById("gate-right"),
      UP: document.getElementById("gate-up"),
      DOWN: document.getElementById("gate-down"),
    };
    for (const [dir, elItem] of Object.entries(dirMap)) {
      if (!elItem) continue;
      const g = gates[dir];
      if (g && g.supported) {
        elItem.textContent = `${dir}: SUPPORTED`;
        elItem.className = "gate-item gate-pass";
      } else {
        elItem.textContent = `${dir}: UNSUPPORTED`;
        elItem.className = "gate-item gate-fail";
      }
    }
  }


  async function fetchTiles() {
    try {
      const res = await fetch("/api/board");
      if (res.ok) {
        const data = await res.json();
        state.tiles = data.tiles;
      }
    } catch (e) {
      // Offline fallback tiles if server endpoint is loading
      state.tiles = [
        { id: "I", text: "I", label: "I", icon: "\u{1F464}", category: "PRONOUN" },
        { id: "want", text: "want", label: "Want", icon: "\u{1F932}", category: "ACTION" },
        { id: "water", text: "water", label: "Water", icon: "\u{1F4A7}", category: "NEED" },
        { id: "food", text: "food", label: "Food", icon: "\u{1F372}", category: "NEED" },
        { id: "help", text: "help", label: "Help", icon: "\u{1F198}", category: "ACTION" },
        { id: "yes", text: "yes", label: "Yes", icon: "\u2705", category: "RESPONSE" },
        { id: "no", text: "no", label: "No", icon: "\u274C", category: "RESPONSE" },
        { id: "pain", text: "pain", label: "Pain", icon: "\u26A1", category: "NEED" },
        { id: "rest", text: "rest", label: "Rest", icon: "\u{1F6CF}\uFE0F", category: "NEED" },
        { id: "toilet", text: "toilet", label: "Toilet", icon: "\u{1F6BB}", category: "NEED" },
        { id: "thank you", text: "thank you", label: "Thank you", icon: "\u{1F64F}", category: "SOCIAL" },
      ];
    }
  }

  async function fetchStatus() {
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data = await res.json();
        updateFromStatus(data);
      }
    } catch (e) {}
  }

  function updateFromStatus(data) {
    state.phraseText = data.phrase || "";
    state.phraseWords = data.words || [];
    state.dwell = data.dwell || state.dwell;
    state.telemetry = data.telemetry || {};

    updatePhraseUI();
    updateTelemetryUI(data);
    updateDwellUI();
  }

  // -------------------------------------------------------------
  // UI Rendering
  // -------------------------------------------------------------

  function renderBoard() {
    el.tilesGrid.innerHTML = "";
    state.tiles.forEach((tile) => {
      const btn = document.createElement("button");
      btn.className = "word-tile";
      btn.dataset.id = tile.id;
      btn.dataset.cat = tile.category;
      btn.setAttribute("role", "button");
      btn.setAttribute("aria-label", `${tile.label}, Category: ${tile.category}`);

      btn.innerHTML = `
        <span class="tile-icon" aria-hidden="true">${tile.icon}</span>
        <span class="tile-label">${tile.label}</span>
        <span class="tile-badge">${tile.category}</span>
      `;

      btn.addEventListener("click", () => handleTileClick(tile));
      btn.addEventListener("mouseenter", () => handleTileMouseEnter(tile));
      el.tilesGrid.appendChild(btn);
    });
  }

  function updatePhraseUI() {
    if (state.phraseWords.length > 0) {
      el.phraseText.textContent = state.phraseText;
      el.phraseText.classList.remove("hidden");
      el.phrasePlaceholder.classList.add("hidden");
    } else {
      el.phraseText.textContent = "";
      el.phraseText.classList.add("hidden");
      el.phrasePlaceholder.classList.remove("hidden");
    }

    // Candidate Preview Tag
    if (state.dwell.candidateTile) {
      el.candidateWord.textContent = state.dwell.candidateTile;
      el.candidateTag.classList.remove("hidden");
    } else {
      el.candidateTag.classList.add("hidden");
    }
  }

  function updateTelemetryUI(statusData) {
    const isLive = state.mode === "LIVE";
    const tel = statusData.telemetry || {};

    if (isLive) {
      el.valSelectionStatus.textContent = "DISABLED (GATES INCOMPLETE)";
      el.valSelectionStatus.className = "metric-val text-danger";
      el.valTrackingQuality.textContent = tel.is_tracking_valid ? "VALID (SENSOR ACTIVE)" : "! ABSTAIN (DEGRADED)";
      el.valGazeCoords.textContent = tel.h_ratio !== null ? `${tel.h_ratio}, ${tel.v_ratio}` : "UNCALIBRATED";
      el.valEar.textContent = tel.ear !== null ? `${tel.ear}` : "--";
      el.valHeadPose.textContent = tel.head_yaw_deg !== null ? `${tel.head_yaw_deg}\u00B0, ${tel.head_pitch_deg}\u00B0` : "--";
      el.telemetryPill.textContent = "RESEARCH DIAGNOSTIC";
      el.telemetryPill.className = "badge badge-paused";
    } else {
      el.valSelectionStatus.textContent = "ENABLED (DEMO SIMULATION)";
      el.valSelectionStatus.className = "metric-val text-success";
      el.valTrackingQuality.textContent = "VALID (100% SIMULATED)";
      el.valGazeCoords.textContent = `${state.pointer.x.toFixed(2)}, ${state.pointer.y.toFixed(2)}`;
      el.valEar.textContent = "0.30 (SIM)";
      el.valHeadPose.textContent = "0.0\u00B0, 0.0\u00B0";
      el.telemetryPill.textContent = "DEMO ACTIVE";
      el.telemetryPill.className = "badge badge-active";
    }
  }

  function updateDwellUI() {
    // 1. Dwell Ring on Gaze Reticle
    const circumference = 2 * Math.PI * 20; // 125.66
    const offset = circumference * (1.0 - Math.min(1.0, state.dwell.progress));
    el.reticleDwellRing.style.strokeDashoffset = offset;

    // 2. Tile Visual Highlights
    const tileElements = el.tilesGrid.querySelectorAll(".word-tile");
    tileElements.forEach((btn) => {
      const tileId = btn.dataset.id;
      const isDwelling = state.dwell.activeTile === tileId && state.dwell.state === "DWELLING";
      const isCandidate = state.dwell.candidateTile === tileId || (state.dwell.activeTile === tileId && state.dwell.state === "CANDIDATE_HIGHLIGHTED");

      btn.classList.toggle("is-dwelling", isDwelling);
      btn.classList.toggle("is-candidate", isCandidate);
    });
  }

  function updateReticlePosition() {
    const px = state.pointer.x * window.innerWidth;
    const py = state.pointer.y * window.innerHeight;
    el.gazeReticle.style.left = `${px}px`;
    el.gazeReticle.style.top = `${py}px`;
  }

  // -------------------------------------------------------------
  // Tile Actions & Speech
  // -------------------------------------------------------------

  async function handleTileClick(tile) {
    if (state.mode === "LIVE") {
      showToast("Live word selection is disabled because physical calibration gates have not passed.", "error");
      return;
    }
    await appendWord(tile.text);
    checkDemoTourStep(tile.text);
  }

  async function handleTileMouseEnter(tile) {
    if (state.mode === "DEMO") {
      // Find tile center in normalized coordinates
      const rect = el.tilesGrid.querySelector(`[data-id="${tile.id}"]`).getBoundingClientRect();
      const nx = (rect.left + rect.width / 2) / window.innerWidth;
      const ny = (rect.top + rect.height / 2) / window.innerHeight;
      state.pointer = { x: nx, y: ny };
      updateReticlePosition();
      await sendMove({ x: nx, y: ny });
    }
  }

  async function appendWord(wordText) {
    try {
      const res = await fetch("/api/phrase/append", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tile_id: wordText }),
      });
      if (res.ok) {
        const data = await res.json();
        state.phraseText = data.phrase;
        state.phraseWords = data.words;
        updatePhraseUI();
        playTone(880, 0.08); // confirmation chime
      } else {
        const err = await res.json();
        showToast(err.detail || "Could not add word.", "error");
      }
    } catch (e) {
      showToast("Network error communicating with AAC server.", "error");
    }
  }

  async function removeLastWord() {
    try {
      const res = await fetch("/api/phrase/backspace", { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        state.phraseText = data.phrase;
        state.phraseWords = data.words;
        updatePhraseUI();
        playTone(440, 0.08);
        showToast("Removed last word.", "info");
      }
    } catch (e) {}
  }

  async function clearPhrase() {
    try {
      const res = await fetch("/api/phrase/clear", { method: "POST" });
      if (res.ok) {
        state.phraseText = "";
        state.phraseWords = [];
        state.dwell.candidateTile = null;
        updatePhraseUI();
        playTone(330, 0.12);
        showToast("Cleared phrase.", "info");
      }
    } catch (e) {}
  }

  async function speakPhrase() {
    if (!state.phraseText || state.phraseWords.length === 0) {
      showToast("Please add words to the phrase bar before speaking.", "error");
      playTone(250, 0.15);
      return;
    }

    try {
      const res = await fetch("/api/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phrase: state.phraseText }),
      });
      const data = await res.json();

      if (data.success) {
        showToast(`\u{1F50A} Spoken: "${data.phrase}" (${data.engine_used})`, "success");
        playTone(1050, 0.12);

        // Fallback: If engine used is client signal or local fallback, also speak via browser offline TTS
        if (data.engine_used === "CLIENT_BROWSER_SPEECH_SIGNAL" && "speechSynthesis" in window) {
          const utter = new SpeechSynthesisUtterance(data.phrase);
          utter.rate = 0.95;
          window.speechSynthesis.speak(utter);
        }

        checkDemoTourStep("SPEAK");
      } else {
        showToast(data.error_message || "Speech synthesis failed.", "error");
      }
    } catch (e) {
      showToast("Error requesting speech synthesis.", "error");
    }
  }

  // -------------------------------------------------------------
  // Simulated Gaze Movement (Demo Mode)
  // -------------------------------------------------------------

  async function sendMove(movePayload) {
    try {
      const res = await fetch("/api/demo/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(movePayload),
      });
      if (res.ok) {
        const data = await res.json();
        state.dwell = data.dwell;
        updateDwellUI();
      }
    } catch (e) {}
  }

  async function confirmCandidate() {
    try {
      const res = await fetch("/api/demo/confirm", { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        if (data.success) {
          state.phraseText = data.phrase;
          state.phraseWords = data.words;
          updatePhraseUI();
          playTone(880, 0.1);
          checkDemoTourStep(data.confirmed_tile);
        }
      }
    } catch (e) {}
  }

  // -------------------------------------------------------------
  // Guided Demo Walkthrough
  // -------------------------------------------------------------

  function startDemoTour() {
    setMode("DEMO");
    clearPhrase();
    state.demoTour.active = true;
    state.demoTour.stepIndex = 0;
    renderTourStep();
    el.tourCard.classList.remove("hidden");
    showToast("Starting Guided AAC Demo Walkthrough", "info");
  }

  function renderTourStep() {
    const step = state.demoTour.steps[state.demoTour.stepIndex];
    if (!step) {
      endDemoTour();
      return;
    }
    el.tourStepBadge.textContent = `STEP ${state.demoTour.stepIndex + 1} OF ${state.demoTour.steps.length}`;
    el.tourStepTitle.textContent = step.title;
    el.tourStepDesc.textContent = step.desc;
    const pct = ((state.demoTour.stepIndex + 1) / state.demoTour.steps.length) * 100;
    el.tourProgressFill.style.width = `${pct}%`;

    // Highlight target tile
    el.tilesGrid.querySelectorAll(".word-tile").forEach((btn) => {
      btn.style.outline = (btn.dataset.id === step.target) ? "4px dashed var(--accent-purple)" : "";
    });
  }

  function checkDemoTourStep(actionWord) {
    if (!state.demoTour.active) return;
    const currentStep = state.demoTour.steps[state.demoTour.stepIndex];
    if (!currentStep) return;

    if (currentStep.target === actionWord || (currentStep.target === "REVIEW" && state.phraseWords.length >= 3)) {
      state.demoTour.stepIndex++;
      if (state.demoTour.stepIndex < state.demoTour.steps.length) {
        renderTourStep();
      } else {
        endDemoTour();
        showToast("\u{1F389} Guided Demo Walkthrough Complete!", "success");
      }
    }
  }

  function endDemoTour() {
    state.demoTour.active = false;
    el.tourCard.classList.add("hidden");
    el.tilesGrid.querySelectorAll(".word-tile").forEach((btn) => (btn.style.outline = ""));
  }

  // -------------------------------------------------------------
  // Mode Switching
  // -------------------------------------------------------------

  async function setMode(mode) {
    state.mode = mode;
    await fetch("/api/mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: mode }),
    });

    if (mode === "DEMO") {
      el.btnModeDemo.classList.add("active");
      el.btnModeDemo.setAttribute("aria-checked", "true");
      el.btnModeLive.classList.remove("active");
      el.btnModeLive.setAttribute("aria-checked", "false");

      el.modeNoticeBanner.className = "notice-banner banner-demo";
      el.bannerIcon.textContent = "\u2139\uFE0F";
      el.bannerTitle.textContent = "DEMO / SIMULATED GAZE MODE ACTIVE";
      el.bannerDesc.innerHTML = "Use <strong>Arrow Keys</strong> or <strong>W, A, S, D</strong> to move the gaze indicator. Press <strong>Space</strong> or dwell to select. <em>Input is simulated for reliable presentation and is not live eye tracking.</em>";
      el.gateWarningPill.classList.add("hidden");
      el.gazeReticle.classList.remove("hidden");
      el.liveCameraCard.classList.add("hidden");
    } else {
      el.btnModeLive.classList.add("active");
      el.btnModeLive.setAttribute("aria-checked", "true");
      el.btnModeDemo.classList.remove("active");
      el.btnModeDemo.setAttribute("aria-checked", "false");

      el.modeNoticeBanner.className = "notice-banner banner-live";
      el.bannerIcon.textContent = "\u26A0\uFE0F";
      el.bannerTitle.textContent = "LIVE RESEARCH SENSOR MODE";
      el.bannerDesc.innerHTML = "Webcam perception pipeline active. <strong>Live word selection is DISABLED</strong> because calibration horizontal and vertical directional gates have not passed.";
      el.gateWarningPill.classList.remove("hidden");
      el.gazeReticle.classList.add("hidden");
      el.liveCameraCard.classList.remove("hidden");
      endDemoTour();
    }
    fetchStatus();
  }

  // -------------------------------------------------------------
  // Live Camera Handling
  // -------------------------------------------------------------

  function updateCameraStatus(newState, message) {
    state.camera.state = newState;
    if (!el.cameraStatusLine) return;

    const statusMap = {
      "INACTIVE":          { text: "\u25CF Camera inactive", cls: "cam-status-warning", badge: "STANDBY", badgeCls: "badge badge-paused" },
      "INITIALIZING":      { text: "\u25CF Initializing camera\u2026", cls: "cam-status-warning", badge: "STARTING", badgeCls: "badge badge-paused" },
      "AVAILABLE":         { text: "\u25CF Camera active \u2014 full frame preview", cls: "cam-status-ok", badge: "ACTIVE", badgeCls: "badge badge-active" },
      "FACE_DETECTED":     { text: "\u25CF Face detected", cls: "cam-status-ok", badge: "TRACKING", badgeCls: "badge badge-active" },
      "FACE_MISSING":      { text: "\u25CF No face detected", cls: "cam-status-warning", badge: "NO FACE", badgeCls: "badge badge-paused" },
      "PERMISSION_DENIED": { text: "\u25CF Camera permission denied", cls: "cam-status-error", badge: "DENIED", badgeCls: "badge badge-danger" },
      "ERROR":             { text: "\u25CF Camera error", cls: "cam-status-error", badge: "ERROR", badgeCls: "badge badge-danger" },
    };

    const info = statusMap[newState] || statusMap["ERROR"];
    el.cameraStatusLine.textContent = message || info.text;
    el.cameraStatusLine.className = "camera-status-indicator " + info.cls;
    if (el.camStatusBadge) {
      el.camStatusBadge.textContent = info.badge;
      el.camStatusBadge.className = info.badgeCls;
    }
  }

  async function toggleCamera() {
    if (state.camera.active) {
      // Stop camera
      if (state.camera.stream) {
        state.camera.stream.getTracks().forEach((t) => t.stop());
        state.camera.stream = null;
      }
      state.camera.active = false;
      el.webcamVideo.srcObject = null;
      el.webcamVideo.classList.add("hidden");
      el.cameraOverlayMsg.classList.remove("hidden");
      el.cameraOverlayMsg.querySelector("span").textContent = "Camera Paused";
      el.btnToggleCamera.textContent = "Start Camera";
      updateCameraStatus("INACTIVE", "\u25CF Camera paused");
    } else {
      // Start camera
      updateCameraStatus("INITIALIZING");
      el.cameraOverlayMsg.querySelector("span").textContent = "Requesting camera access\u2026";

      try {
        // Request full-frame capture (not face-cropped); mirroring is CSS-only
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: "user",
          }
        });
        state.camera.stream = stream;
        state.camera.active = true;

        el.webcamVideo.srcObject = stream;
        el.webcamVideo.classList.remove("hidden");
        el.cameraOverlayMsg.classList.add("hidden");
        el.btnToggleCamera.textContent = "Stop Camera";
        updateCameraStatus("AVAILABLE");
        showToast("Webcam stream active \u2014 full-frame preview (mirrored for natural feedback).", "info");

        // Monitor video track state changes
        const videoTrack = stream.getVideoTracks()[0];
        if (videoTrack) {
          videoTrack.addEventListener("ended", () => {
            state.camera.active = false;
            el.webcamVideo.classList.add("hidden");
            el.cameraOverlayMsg.classList.remove("hidden");
            el.cameraOverlayMsg.querySelector("span").textContent = "Camera stream ended";
            el.btnToggleCamera.textContent = "Start Camera";
            updateCameraStatus("INACTIVE", "\u25CF Camera stream ended");
          });
        }
      } catch (err) {
        state.camera.active = false;
        state.camera.errorMessage = err.message || "Unknown camera error";

        if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
          updateCameraStatus("PERMISSION_DENIED");
          el.cameraOverlayMsg.querySelector("span").textContent = "Camera permission denied. Please allow camera access in browser settings.";
          showToast("Camera permission denied. Grant access in browser settings to use live mode.", "error");
        } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
          updateCameraStatus("ERROR", "\u25CF No camera found");
          el.cameraOverlayMsg.querySelector("span").textContent = "No webcam detected. Connect a camera and try again.";
          showToast("No webcam detected.", "error");
        } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
          updateCameraStatus("ERROR", "\u25CF Camera busy or unavailable");
          el.cameraOverlayMsg.querySelector("span").textContent = "Camera is in use by another application.";
          showToast("Camera is busy. Close other applications using the camera.", "error");
        } else {
          updateCameraStatus("ERROR");
          el.cameraOverlayMsg.querySelector("span").textContent = `Camera error: ${err.message}`;
          showToast(`Camera error: ${err.message}`, "error");
        }
      }
    }
  }

  // -------------------------------------------------------------
  // Calibration Flow
  // -------------------------------------------------------------

  function openCalibrationModal() {
    el.calibrationModal.classList.remove("hidden");
    resetCalibrationUI();
    drawCalibrationFixationTargets(-1); // No active target initially
  }

  function closeCalibrationModal() {
    // Abort any running calibration
    if (state.calibration.running) {
      if (state.calibration.animFrameId) {
        cancelAnimationFrame(state.calibration.animFrameId);
        state.calibration.animFrameId = null;
      }
      state.calibration.running = false;
      state.calibration.phase = "IDLE";
      fetch("/api/calibration/session/abort", { method: "POST" }).catch(() => {});
    }
    el.calibrationModal.classList.add("hidden");
  }

  function resetCalibrationUI() {
    if (state.calibration.animFrameId) {
      cancelAnimationFrame(state.calibration.animFrameId);
      state.calibration.animFrameId = null;
    }
    state.calibration.running = false;
    state.calibration.currentTargetIndex = -1;
    state.calibration.phase = "IDLE";
    state.calibration.samplesCollected = 0;

    el.calibFeedback.textContent = "Status: Ready. Gate validation requires minimum 0.06 H and 0.05 V separation.";
    el.calibFeedback.className = "calib-feedback-banner";
    if (el.calibProgressFill) el.calibProgressFill.style.width = "0%";

    el.btnStartCalibSeq.classList.remove("hidden");
    el.btnStartCalibSeq.disabled = false;
    el.btnRetryCalib.classList.add("hidden");
  }

  function drawCalibrationFixationTargets(activeIndex) {
    const canvas = el.calibCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = "#090d16";
    ctx.fillRect(0, 0, w, h);

    const targets = state.calibration.targets;

    targets.forEach((t, idx) => {
      const tx = t.x * w;
      const ty = t.y * h;
      const isActive = (idx === activeIndex);
      const isCompleted = (idx < activeIndex);

      // Draw target circle
      ctx.beginPath();
      ctx.arc(tx, ty, isActive ? 28 : 14, 0, 2 * Math.PI);

      if (isActive) {
        // Active target: pulsing cyan/green
        ctx.fillStyle = (state.calibration.phase === "RECORDING") ? "#22c55e" : "#38bdf8";
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 3;
        ctx.stroke();

        // Inner fixation dot
        ctx.beginPath();
        ctx.arc(tx, ty, 5, 0, 2 * Math.PI);
        ctx.fillStyle = "#000";
        ctx.fill();
      } else if (isCompleted) {
        // Completed target: green check
        ctx.fillStyle = "rgba(34, 197, 94, 0.4)";
        ctx.fill();
        ctx.strokeStyle = "#22c55e";
        ctx.lineWidth = 2;
        ctx.stroke();

        // Checkmark
        ctx.fillStyle = "#22c55e";
        ctx.font = "bold 16px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("\u2713", tx, ty);
      } else {
        // Future target: dim
        ctx.fillStyle = "#334155";
        ctx.fill();
        ctx.strokeStyle = "#475569";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Label
      ctx.fillStyle = isActive ? "#ffffff" : "#cbd5e1";
      ctx.font = isActive ? "bold 14px sans-serif" : "12px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(t.name, tx, ty + (isActive ? 36 : 22));
    });

    // Phase instruction for active target
    if (activeIndex >= 0 && activeIndex < targets.length) {
      const target = targets[activeIndex];
      const phase = state.calibration.phase;
      let instruction = "";
      if (phase === "SETTLING") {
        instruction = `\u23F3 SETTLING \u2014 ${target.instruction || "Hold your gaze steady on the target\u2026"}`;
      } else if (phase === "RECORDING") {
        instruction = `\u{1F534} RECORDING \u2014 Keep looking steadily at ${target.name}!`;
      } else if (phase === "TARGET_COMPLETE") {
        instruction = "\u2705 Target complete. Advancing\u2026";
      }

      if (instruction) {
        ctx.fillStyle = "#f8fafc";
        ctx.font = "bold 15px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.fillText(instruction, w / 2, h - 14);
      }
    }
  }

  /**
   * Start the 5-point calibration sequence.
   * Invokes backend /api/calibration/session/start to initialize protocol parameters.
   */
  async function startCalibrationSequence() {
    // Prevent duplicate submissions
    if (state.calibration.running) return;

    // Verify camera is active
    if (!state.camera.active) {
      el.calibFeedback.textContent = "Error: Camera must be active before starting calibration. Please start the camera first.";
      el.calibFeedback.className = "calib-feedback-banner is-error";
      showToast("Start the webcam camera before calibrating.", "error");
      return;
    }

    try {
      const res = await fetch("/api/calibration/session/start", { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        if (data.targets && data.targets.length > 0) {
          state.calibration.targets = data.targets;
        }
        state.calibration.settlingMs = data.settling_ms || 500;
        state.calibration.recordingMs = data.recording_ms || 1500;
      }
    } catch (e) {
      // Network error fallback
    }

    state.calibration.running = true;
    state.calibration.currentTargetIndex = 0;
    state.calibration.phase = "SETTLING";
    state.calibration.phaseStartTime = performance.now();
    state.calibration.samplesCollected = 0;

    el.btnStartCalibSeq.disabled = true;
    el.btnStartCalibSeq.classList.add("hidden");
    el.btnRetryCalib.classList.add("hidden");

    showToast("Calibration started. Fix your gaze on each target.", "info");
    playTone(800, 0.1);

    runCalibrationLoop();
  }

  function runCalibrationLoop() {
    if (!state.calibration.running) return;

    const now = performance.now();
    const cal = state.calibration;
    const elapsed = now - cal.phaseStartTime;
    const totalTargets = cal.targets.length;
    const target = cal.targets[cal.currentTargetIndex];

    if (cal.phase === "SETTLING") {
      const settlingProgress = Math.min(1.0, elapsed / cal.settlingMs);

      el.calibFeedback.textContent = `\u23F3 Target ${cal.currentTargetIndex + 1}/${totalTargets}: ${target.name} \u2014 Settling (${Math.round(settlingProgress * 100)}%)`;
      el.calibFeedback.className = "calib-feedback-banner is-settling";

      drawCalibrationFixationTargets(cal.currentTargetIndex);
      updateCalibProgress(cal.currentTargetIndex, "settling", settlingProgress);

      if (elapsed >= cal.settlingMs) {
        cal.phase = "RECORDING";
        cal.phaseStartTime = now;
        playTone(600, 0.06);
      }

      cal.animFrameId = requestAnimationFrame(runCalibrationLoop);

    } else if (cal.phase === "RECORDING") {
      const recordingProgress = Math.min(1.0, elapsed / cal.recordingMs);

      el.calibFeedback.textContent = `\u{1F534} Target ${cal.currentTargetIndex + 1}/${totalTargets}: ${target.name} \u2014 Recording (${Math.round(recordingProgress * 100)}%)`;
      el.calibFeedback.className = "calib-feedback-banner is-recording";

      drawCalibrationFixationTargets(cal.currentTargetIndex);
      updateCalibProgress(cal.currentTargetIndex, "recording", recordingProgress);

      if (elapsed >= cal.recordingMs) {
        cal.phase = "TARGET_COMPLETE";
        cal.phaseStartTime = now;
        playTone(1000, 0.08);
      }

      cal.animFrameId = requestAnimationFrame(runCalibrationLoop);

    } else if (cal.phase === "TARGET_COMPLETE") {
      drawCalibrationFixationTargets(cal.currentTargetIndex);

      if (elapsed >= 200) {
        cal.currentTargetIndex++;

        if (cal.currentTargetIndex >= totalTargets) {
          completeCalibrationSequence();
        } else {
          cal.phase = "SETTLING";
          cal.phaseStartTime = performance.now();
          cal.animFrameId = requestAnimationFrame(runCalibrationLoop);
        }
      } else {
        cal.animFrameId = requestAnimationFrame(runCalibrationLoop);
      }
    }
  }

  function updateCalibProgress(targetIndex, subPhase, subProgress) {
    const totalTargets = state.calibration.targets.length;
    const targetPortion = 1.0 / totalTargets;
    let overallProgress;

    if (subPhase === "settling") {
      overallProgress = (targetIndex * targetPortion) + (subProgress * 0.25 * targetPortion);
    } else {
      overallProgress = (targetIndex * targetPortion) + (0.25 + subProgress * 0.75) * targetPortion;
    }

    if (el.calibProgressFill) {
      el.calibProgressFill.style.width = `${Math.round(overallProgress * 100)}%`;
    }
  }

  async function completeCalibrationSequence() {
    state.calibration.running = false;
    state.calibration.phase = "SEQUENCE_COMPLETE";
    state.calibration.animFrameId = null;

    if (el.calibProgressFill) el.calibProgressFill.style.width = "100%";

    drawCalibrationFixationTargets(state.calibration.targets.length); // All completed

    let summaryMsg =
      "Fixation sequence complete. " +
      "NOTE: Physical calibration gates (H \u2265 0.06, V \u2265 0.05 separation) have NOT passed. " +
      "Live word activation remains DISABLED.";

    try {
      const res = await fetch("/api/calibration/session/complete", { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        if (data.summary_message) summaryMsg = data.summary_message;
      }
    } catch (e) {}

    el.calibFeedback.textContent = summaryMsg;
    el.calibFeedback.className = "calib-feedback-banner is-complete";

    el.btnRetryCalib.classList.remove("hidden");
    el.btnStartCalibSeq.classList.add("hidden");

    playTone(1200, 0.15);
    showToast("Fixation sequence complete. Live word activation remains disabled.", "info");
  }

  function retryCalibration() {
    resetCalibrationUI();
    drawCalibrationFixationTargets(-1);
    showToast("Calibration reset. Press Start to begin again.", "info");
  }


  // -------------------------------------------------------------
  // Keyboard Navigation
  // -------------------------------------------------------------

  function setupEventListeners() {
    // Buttons
    el.btnSpeak.addEventListener("click", speakPhrase);
    el.btnBackspace.addEventListener("click", removeLastWord);
    el.btnClear.addEventListener("click", clearPhrase);
    el.btnModeDemo.addEventListener("click", () => setMode("DEMO"));
    el.btnModeLive.addEventListener("click", () => setMode("LIVE"));
    el.btnStartDemo.addEventListener("click", startDemoTour);
    el.btnExitTour.addEventListener("click", endDemoTour);
    el.btnToggleCamera.addEventListener("click", toggleCamera);
    el.btnOpenCalibration.addEventListener("click", openCalibrationModal);
    el.btnCloseCalib.addEventListener("click", closeCalibrationModal);
    el.btnExitCalib.addEventListener("click", closeCalibrationModal);
    // FIX: Wire the Start 5-Point Calibration button to the actual sequence handler
    el.btnStartCalibSeq.addEventListener("click", startCalibrationSequence);
    el.btnRetryCalib.addEventListener("click", retryCalibration);

    // Keyboard controls
    window.addEventListener("keydown", (e) => {
      // Ignore if user is inside an input
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

      const step = 0.08;
      let handled = false;

      if (state.mode === "DEMO") {
        switch (e.key) {
          case "ArrowLeft":
          case "a":
          case "A":
            state.pointer.x = Math.max(0.05, state.pointer.x - step);
            handled = true;
            break;
          case "ArrowRight":
          case "d":
          case "D":
            state.pointer.x = Math.min(0.95, state.pointer.x + step);
            handled = true;
            break;
          case "ArrowUp":
          case "w":
          case "W":
            state.pointer.y = Math.max(0.05, state.pointer.y - step);
            handled = true;
            break;
          case "ArrowDown":
          case "s":
          case "S":
            state.pointer.y = Math.min(0.95, state.pointer.y + step);
            handled = true;
            break;
          case " ":
            // Space: Confirm candidate or select tile directly under reticle
            if (state.dwell.candidateTile) {
              confirmCandidate();
            } else {
              // Direct hit-test
              const tileAtPos = findTileUnderPointer(state.pointer.x, state.pointer.y);
              if (tileAtPos) appendWord(tileAtPos.text);
            }
            handled = true;
            break;
        }

        if (handled) {
          updateReticlePosition();
          sendMove({ x: state.pointer.x, y: state.pointer.y });
        }
      }

      // Universal shortcuts
      if (e.key === "Backspace") {
        removeLastWord();
        e.preventDefault();
      } else if (e.key === "c" || e.key === "C") {
        if (!e.ctrlKey && !e.metaKey) {
          clearPhrase();
          e.preventDefault();
        }
      } else if (e.key === "Enter" || ((e.key === "s" || e.key === "S") && state.mode !== "DEMO")) {
        speakPhrase();
        e.preventDefault();
      }
    });
  }

  function findTileUnderPointer(nx, ny) {
    const px = nx * window.innerWidth;
    const py = ny * window.innerHeight;
    const element = document.elementFromPoint(px, py);
    if (!element) return null;
    const tileBtn = element.closest(".word-tile");
    if (!tileBtn) return null;
    return state.tiles.find((t) => t.id === tileBtn.dataset.id);
  }

  // -------------------------------------------------------------
  // WebSocket Telemetry
  // -------------------------------------------------------------

  function setupWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
    let ws = null;

    try {
      ws = new WebSocket(wsUrl);
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          updateFromStatus(data);
        } catch (e) {}
      };
      ws.onclose = () => {
        // Reconnect after delay
        setTimeout(setupWebSocket, 2000);
      };
    } catch (e) {}
  }

  // -------------------------------------------------------------
  // Toast Notifications
  // -------------------------------------------------------------

  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    el.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transition = "opacity 0.3s";
      setTimeout(() => toast.remove(), 300);
    }, 3200);
  }

  // Kickoff on DOM loaded
  document.addEventListener("DOMContentLoaded", init);
})();
