/**
 * VoxGaze AAC Experimental Demo: High-Contrast Gaze-Tree Client (demo.js)
 *
 * Implements:
 * 1. 10 Hz polling loop for /api/demo/state.
 * 2. Real-time rendering of screen nodes, candidate preview, and hold progress.
 * 3. Multi-modal state cues (ARMED vs LOOK TO CENTRE, Tracking OK vs PAUSED).
 * 4. Keyboard shortcuts (Esc=Reset, P=Pause/Resume, C=Recalibrate, Arrows=Key Nav).
 * 5. Strictly ZERO browser camera requests (Backend maintains exclusive camera control).
 */

(function () {
  "use strict";

  // DOM Elements
  const bannerText = document.getElementById("banner-text");
  const phraseDisplay = document.getElementById("phrase-display");
  const speakingIndicator = document.getElementById("speaking-indicator");

  const screenTitle = document.getElementById("screen-title");
  const screenLevelTag = document.getElementById("screen-level-tag");
  const confirmPreviewBox = document.getElementById("confirm-preview-box");
  const confirmPrompt = document.getElementById("confirm-prompt");
  const confirmCandidate = document.getElementById("confirm-candidate");

  const panelLeft = document.getElementById("panel-left");
  const panelRight = document.getElementById("panel-right");
  const labelLeft = document.getElementById("label-left");
  const labelRight = document.getElementById("label-right");
  const dirNameLeft = document.getElementById("dir-name-left");
  const dirNameRight = document.getElementById("dir-name-right");
  const progressLeft = document.getElementById("progress-left");
  const progressRight = document.getElementById("progress-right");
  const progressBarFill = document.getElementById("progress-bar-fill");

  const centerPill = document.getElementById("center-pill");
  const centerDot = document.getElementById("center-dot");
  const armCueBadge = document.getElementById("arm-cue-badge");
  const armExplanation = document.getElementById("arm-explanation");

  const valDirection = document.getElementById("val-direction");
  const valTracking = document.getElementById("val-tracking");
  const valFps = document.getElementById("val-fps");
  const valLatency = document.getElementById("val-latency");
  const valSpeakingContainer = document.getElementById("val-speaking-container");

  // State Tracking
  let isPolling = true;
  let pollIntervalId = null;
  let isRequestInFlight = false;

  /**
   * Main state rendering routine.
   */
  function renderState(state) {
    if (!state) return;

    // 1. Permanent Banner (if present)
    if (bannerText && state.banner_text && bannerText.textContent !== state.banner_text) {
      bannerText.textContent = state.banner_text;
    }

    // 2. Phrase Display & Speaking Badge
    if (state.phrase && state.phrase.length > 0) {
      phraseDisplay.innerHTML = "";
      phraseDisplay.textContent = state.phrase.join(" ");
    } else {
      phraseDisplay.innerHTML = `<span class="phrase-placeholder">(No words yet &mdash; look left or right to start)</span>`;
    }

    if (state.speaking) {
      speakingIndicator.style.display = "flex";
    } else {
      speakingIndicator.style.display = "none";
    }

    // 3. Screen Mode & Hierarchy Context
    const screen = state.screen || {};
    const isConfirm = screen.kind === "confirm";

    if (isConfirm) {
      document.body.classList.add("confirm-mode");
      screenLevelTag.textContent = "CONFIRM";
      screenTitle.style.display = "none";
      confirmPreviewBox.style.display = "flex";

      if (screen.candidate_type === "action") {
        confirmPrompt.textContent = screen.candidate === "SPEAK" ? "SPEAK PHRASE?" : "UNDO LAST WORD?";
      } else {
        confirmPrompt.textContent = "ADD WORD?";
      }
      confirmCandidate.textContent = screen.candidate || "";

      dirNameLeft.textContent = "YES";
      dirNameRight.textContent = "NO";
      labelLeft.textContent = screen.left_label || "YES";
      labelRight.textContent = screen.right_label || "NO";
    } else {
      document.body.classList.remove("confirm-mode");
      screenLevelTag.textContent = "LEVEL";
      screenTitle.style.display = "block";
      screenTitle.textContent = screen.title || "ROOT";
      confirmPreviewBox.style.display = "none";

      dirNameLeft.textContent = "LEFT";
      dirNameRight.textContent = "RIGHT";
      labelLeft.textContent = screen.left_label || "";
      labelRight.textContent = screen.right_label || "";
    }

    // 4. Hold Progress & Panel Highlights
    const hold = state.hold || {};
    const holdDir = (hold.direction || "").toUpperCase();
    const holdProg = Math.max(0.0, Math.min(1.0, state.hold_progress !== undefined ? state.hold_progress : (hold.progress || 0.0)));
    const holdPct = `${Math.round(holdProg * 100)}%`;

    if (progressBarFill) {
      progressBarFill.style.width = holdPct;
    }

    if (holdDir === "LEFT") {
      panelLeft.classList.add("panel-holding");
      panelRight.classList.remove("panel-holding");
      progressLeft.style.width = holdPct;
      progressRight.style.width = "0%";
    } else if (holdDir === "RIGHT") {
      panelRight.classList.add("panel-holding");
      panelLeft.classList.remove("panel-holding");
      progressRight.style.width = holdPct;
      progressLeft.style.width = "0%";
    } else {
      panelLeft.classList.remove("panel-holding");
      panelRight.classList.remove("panel-holding");
      progressLeft.style.width = "0%";
      progressRight.style.width = "0%";
    }

    // 5. Center & Arming Status
    const isArmed = Boolean(state.armed);
    const detDir = (state.detected_direction || "CENTER").toUpperCase();

    if (detDir === "CENTER") {
      centerPill.classList.add("is-holding-center");
    } else {
      centerPill.classList.remove("is-holding-center");
    }

    if (isArmed) {
      centerPill.classList.add("is-armed");
      centerDot.textContent = "●";
      armCueBadge.textContent = "ARMED / LOOK TO CENTRE";
      armExplanation.textContent = "Ready: Look LEFT or RIGHT and hold to select";
    } else {
      centerPill.classList.remove("is-armed");
      centerDot.textContent = "○";
      armCueBadge.textContent = "RE-ARMING AT CENTRE...";
      armExplanation.textContent = "Hold eyes at CENTER (500 ms) to arm next decision";
    }

    // 6. Bottom Telemetry
    valDirection.textContent = detDir;

    const trackingStatus = (state.tracking || state.tracking_status || "ok").toLowerCase();
    if (trackingStatus === "ok") {
      valTracking.textContent = "● OK";
      valTracking.className = "item-value status-ok";
    } else if (trackingStatus === "paused") {
      valTracking.textContent = "! PAUSED";
      valTracking.className = "item-value status-warn";
    } else {
      valTracking.textContent = "✕ LOST";
      valTracking.className = "item-value status-danger";
    }

    const fpsVal = state.fps !== undefined ? state.fps : state.measured_fps;
    if (fpsVal !== undefined && fpsVal !== null) {
      valFps.textContent = `${Math.round(fpsVal)} fps`;
    } else {
      valFps.textContent = "-- fps";
    }

    if (valLatency) {
      if (state.per_frame_latency_ms !== undefined && state.per_frame_latency_ms !== null) {
        valLatency.textContent = `${Math.round(state.per_frame_latency_ms)} ms`;
      } else {
        valLatency.textContent = "-- ms";
      }
    }

    if (valSpeakingContainer) {
      valSpeakingContainer.style.display = state.speaking ? "flex" : "none";
    }
  }

  /**
   * 10 Hz Polling function.
   */
  async function pollState() {
    if (isRequestInFlight) return;
    isRequestInFlight = true;

    try {
      const response = await fetch("/api/demo/state");
      if (response.ok) {
        const data = await response.json();
        renderState(data);
      } else if (response.status === 404) {
        valTracking.textContent = "✕ DEMO INACTIVE";
        valTracking.className = "item-value status-danger";
        armExplanation.textContent = "Server running without --experimental-demo flag";
      }
    } catch (err) {
      valTracking.textContent = "! DISCONNECTED";
      valTracking.className = "item-value status-danger";
    } finally {
      isRequestInFlight = false;
    }
  }

  /**
   * Send control actions to backend.
   */
  async function sendAction(endpoint, payload) {
    try {
      const opts = { method: "POST" };
      if (payload) {
        opts.headers = { "Content-Type": "application/json" };
        opts.body = JSON.stringify(payload);
      }
      await fetch(endpoint, opts);
      // Immediately trigger poll to reflect state
      pollState();
    } catch (err) {
      console.error(`Failed action to ${endpoint}:`, err);
    }
  }

  /**
   * Keyboard shortcuts.
   */
  function handleKeyDown(event) {
    if (event.key === "Escape") {
      event.preventDefault();
      sendAction("/api/demo/reset");
    } else if (event.key === "p" || event.key === "P") {
      event.preventDefault();
      sendAction("/api/demo/pause");
    } else if (event.key === "c" || event.key === "C") {
      event.preventDefault();
      sendAction("/api/demo/recalibrate");
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      sendAction("/api/demo/key", { key: "left" });
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      sendAction("/api/demo/key", { key: "right" });
    }
  }

  // Initialize
  window.addEventListener("DOMContentLoaded", () => {
    document.addEventListener("keydown", handleKeyDown);
    pollIntervalId = setInterval(pollState, 100); // 10 Hz polling
    pollState();
  });
})();
