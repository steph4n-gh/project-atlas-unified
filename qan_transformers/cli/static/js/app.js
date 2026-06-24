/**
 * GossetGate: Main Application Orchestrator
 */
import { loadModelStream, ingestCodebase, fetchContextFiles } from './api.js';
import { SidebarPanel } from './components/sidebar.js';
import { TelemetryPanel } from './components/telemetry.js';
import { ChatConsole } from './components/chat.js';
import { E8Visualizer3D } from './components/e8_viz_3d.js';
import { AuditPanel } from './components/audit.js';

document.addEventListener('DOMContentLoaded', () => {
    // Add decorative screws to all cards and panels for rack-mount look
    const panels = document.querySelectorAll('.control-card, .panel-card, .tour-panel, .audit-panel, .self-improve-panel');
    panels.forEach(p => {
        // Ensure position relative for absolute positioning of screws
        p.style.position = 'relative';
        const screws = ['top-left', 'top-right', 'bottom-left', 'bottom-right'];
        screws.forEach(pos => {
            const screw = document.createElement('div');
            screw.className = `rack-screw ${pos}`;
            p.appendChild(screw);
        });
    });

    // --- Global State ---
    let isModelLoaded = false;
    const activeIndices = new Set();

    // --- Instantiate Telemetry & Logging console ---
    const telemetry = new TelemetryPanel();
    const appendLog = (msg) => telemetry.appendLog(msg);

    // --- Instantiate Sidebar configuration manager ---
    const sidebar = new SidebarPanel(appendLog);

    // --- Instantiate E8 WebGL Visualizer ---
    const visualizer = new E8Visualizer3D();

    // Callback when chat streams return active memory grid points
    const onGridPointsUpdated = (gridPoints) => {
        if (!visualizer) return;
        activeIndices.clear();
        gridPoints.forEach(pt => {
            let closestIdx = -1;
            let minDistance = 1e9;
            visualizer.e8CoordinatesList.forEach((item) => {
                const dx = item.pt3d[0] - pt[0];
                const dy = item.pt3d[1] - pt[1];
                const dz = item.pt3d[2] - pt[2];
                const dist = dx*dx + dy*dy + dz*dz;
                if (dist < minDistance) {
                    minDistance = dist;
                    closestIdx = item.idx;
                }
            });
            if (closestIdx !== -1 && minDistance < 0.25) {
                activeIndices.add(closestIdx);
            }
        });
        visualizer.updateActiveCoordinates(activeIndices);
    };

    // Callback when chat streams telemetry metrics
    const updateTelemetry = (data) => telemetry.update(data);

    // --- Instantiate Chat Console ---
    const chat = new ChatConsole(appendLog, onGridPointsUpdated, updateTelemetry);

    // --- Instantiate Code Audit & Self-Improvement loop tab ---
    const audit = new AuditPanel(appendLog);

    // --- Tab Switcher Logic ---
    const chatTabBtn = document.getElementById('tab-chat-btn');
    const optimizeTabBtn = document.getElementById('tab-optimize-btn');
    const chatPanel = document.querySelector('.main-layout');
    const auditPanelEl = document.querySelector('.main-layout-audit'); // new audit layout selector

    if (chatTabBtn && optimizeTabBtn) {
        chatTabBtn.addEventListener('click', () => {
            chatTabBtn.classList.add('active');
            optimizeTabBtn.classList.remove('active');
            if (chatPanel) chatPanel.classList.remove('hidden');
            const auditSec = document.getElementById('self-improvement-layout');
            if (auditSec) auditSec.classList.add('hidden');
        });

        optimizeTabBtn.addEventListener('click', () => {
            optimizeTabBtn.classList.add('active');
            chatTabBtn.classList.remove('active');
            if (chatPanel) chatPanel.classList.add('hidden');
            const auditSec = document.getElementById('self-improvement-layout');
            if (auditSec) {
                auditSec.classList.remove('hidden');
                // Trigger canvas resize
                if (audit) audit.resizeCanvas();
            }
        });
    }

    // --- Model Ingestion and Load Action Handlers ---
    const loadModelBtn = document.getElementById('load-model-btn');
    const targetModelInput = document.getElementById('target-model-input');
    const draftModelInput = document.getElementById('draft-model-input');
    const draftStrategy = document.getElementById('draft-strategy');
    const mockModelCheckbox = document.getElementById('mock-model-checkbox');
    const modelPrecision = document.getElementById('model-precision');
    const modelFramework = document.getElementById('model-framework');

    // Modal Progress elements
    const loadModal = document.getElementById('load-modal');
    const loadProgressFill = document.getElementById('load-progress-fill');
    const loadModalLogs = document.getElementById('load-modal-logs');
    const modalCloseBtn = document.getElementById('modal-close-btn');

    if (loadModelBtn) {
        loadModelBtn.addEventListener('click', () => {
            const targetModel = targetModelInput.value.trim();
            const draftModel = draftModelInput.value.trim();
            const useSpeculative = (draftStrategy.value === 'speculative');
            const sparseRatio = parseFloat(sidebar.sparseRatioSlider.value);
            const lightweightMock = mockModelCheckbox.checked;
            const precisionValue = modelPrecision.value;
            const frameworkValue = modelFramework ? modelFramework.value : 'mlx';

            if (!targetModel) {
                alert("Please input a valid Hugging Face target model.");
                return;
            }

            // Setup loading modal UI
            loadModalLogs.textContent = "Connecting to initialization stream...\n";
            loadProgressFill.style.width = "0%";
            loadModal.classList.remove('hidden');
            modalCloseBtn.disabled = true;

            const checklistItems = ['device', 'tokenizer', 'target_model', 'swap_db', 'firewall', 'draft_model'];
            checklistItems.forEach(id => {
                const el = document.getElementById(`chk-${id}`);
                if (el) {
                    el.className = "checklist-item";
                    el.querySelector('.status-icon').innerHTML = "&#x25CB;";
                }
            });

            const draftCheckEl = document.getElementById('chk-draft_model');
            if (useSpeculative) {
                draftCheckEl.classList.remove('hidden');
            } else {
                draftCheckEl.classList.add('hidden');
            }

            const stepsToComplete = useSpeculative ? 6 : 5;
            let completedSteps = 0;

            const onStep = (data) => {
                const stepEl = document.getElementById(`chk-${data.step}`);
                if (stepEl) {
                    const iconEl = stepEl.querySelector('.status-icon');
                    if (data.status === 'running') {
                        stepEl.className = "checklist-item running";
                        iconEl.innerHTML = "&#x21BB;";
                        iconEl.classList.add('running');
                    } else if (data.status === 'success') {
                        stepEl.className = "checklist-item success";
                        iconEl.innerHTML = "&#x2713;";
                        iconEl.classList.remove('running');
                        completedSteps++;
                        loadProgressFill.style.width = `${(completedSteps / stepsToComplete) * 100}%`;
                    }
                }
                loadModalLogs.textContent += `[INFO] ${data.message}\n`;
                loadModalLogs.scrollTop = loadModalLogs.scrollHeight;
            };

            const onComplete = (data) => {
                if (useSpeculative) {
                    const draftEl = document.getElementById('chk-draft_model');
                    if (draftEl) {
                        draftEl.className = "checklist-item success";
                        draftEl.querySelector('.status-icon').innerHTML = "&#x2713;";
                        draftEl.querySelector('.status-icon').classList.remove('running');
                    }
                }
                loadProgressFill.style.width = "100%";
                loadModalLogs.textContent += `\n[SUCCESS] Model engine ready: ${data.message}\n`;
                loadModalLogs.scrollTop = loadModalLogs.scrollHeight;

                isModelLoaded = true;
                loadModelBtn.textContent = "Re-initialize Model Engine";
                chat.enable(true);
                
                const ingestBtn = document.getElementById('ingest-context-btn');
                if (ingestBtn) ingestBtn.disabled = false;
                modalCloseBtn.disabled = false;

                // Update system headers
                const cleanName = targetModel.split('/').pop();
                document.getElementById('status-model-lbl').textContent = `Model: ${cleanName}` + (useSpeculative ? ` + Spec` : ``);
                document.getElementById('status-model-dot').className = "status-dot green";

                document.getElementById('status-db-lbl').textContent = "E8 Grid: Standby";
                document.getElementById('status-db-dot').className = "status-dot green";

                setTimeout(() => {
                    loadModal.classList.add('hidden');
                }, 800);
            };

            const onError = (data) => {
                loadModalLogs.textContent += `\n[ERROR] Load Failed: ${data.message}\n`;
                loadModalLogs.scrollTop = loadModalLogs.scrollHeight;
                modalCloseBtn.disabled = false;
                
                // Show diagnostics
                const diagnosticsModal = document.getElementById('diagnostics-modal');
                const diagErrorMsg = document.getElementById('diag-error-msg');
                const diagTraceback = document.getElementById('diag-traceback');
                const guideHfAuth = document.getElementById('guide-hf-auth');
                const guideOom = document.getElementById('guide-oom');

                if (diagnosticsModal) {
                    diagnosticsModal.classList.remove('hidden');
                    if (diagErrorMsg) diagErrorMsg.textContent = data.message;
                    if (diagTraceback) diagTraceback.textContent = data.traceback || "No python traceback available.";

                    const isGated = data.message.includes("gated") || data.message.includes("401") || data.message.includes("unauthorized");
                    const isOom = data.message.includes("out of memory") || data.message.includes("allocation") || data.message.includes("137");

                    if (guideHfAuth) guideHfAuth.style.display = isGated ? "block" : "none";
                    if (guideOom) guideOom.style.display = isOom ? "block" : "none";
                }
            };

            loadModelStream({
                target_model: targetModel,
                draft_model: draftModel,
                use_speculative: useSpeculative,
                sparse_ratio: sparseRatio,
                lightweight_mock: lightweightMock,
                precision: precisionValue,
                framework: frameworkValue
            }, onStep, onComplete, onError);
        });
    }

    if (modalCloseBtn) {
        modalCloseBtn.addEventListener('click', () => loadModal.classList.add('hidden'));
    }

    const diagCloseBtn = document.getElementById('diag-close-btn');
    if (diagCloseBtn) {
        diagCloseBtn.addEventListener('click', () => {
            const diagnosticsModal = document.getElementById('diagnostics-modal');
            if (diagnosticsModal) diagnosticsModal.classList.add('hidden');
        });
    }

    // --- Context Ingestion Action Handlers ---
    const ingestContextBtn = document.getElementById('ingest-context-btn');
    const ingestStats = document.getElementById('ingest-stats');
    const ingestTokensLbl = document.getElementById('ingest-tokens-lbl');
    const ingestTimeLbl = document.getElementById('ingest-time-lbl');
    const contextInput = document.getElementById('context-input');

    if (ingestContextBtn) {
        ingestContextBtn.addEventListener('click', async () => {
            const folderPath = contextInput.value.trim();
            if (!folderPath) {
                alert("Please specify a valid folder path for indexing.");
                return;
            }

            ingestContextBtn.disabled = true;
            ingestContextBtn.textContent = "Ingesting...";
            appendLog(`Indexing folder contents recursively: ${folderPath}`);

            try {
                const data = await ingestCodebase(folderPath);
                
                if (ingestStats) ingestStats.classList.remove('hidden');
                if (ingestTokensLbl) ingestTokensLbl.textContent = data.tokens.toLocaleString();
                if (ingestTimeLbl) ingestTimeLbl.textContent = data.time_seconds.toFixed(2);

                appendLog(`Ingestion successful. Ingested ${data.tokens} tokens in ${data.time_seconds.toFixed(2)}s.`);
                
                // Refresh files list
                loadCodebaseFileList();
            } catch (err) {
                console.error("Ingestion failed", err);
                appendLog(`[ERROR] Ingestion failed: ${err.message}`);
                alert(`Ingestion failed: ${err.message}`);
            } finally {
                ingestContextBtn.disabled = false;
                ingestContextBtn.textContent = "Ingest Folder Context";
            }
        });
    }

    // Ingestion files list update
    async function loadCodebaseFileList() {
        const fileListContainer = document.getElementById('ingested-files-list');
        if (!fileListContainer) return;
        
        try {
            const data = await fetchContextFiles();
            fileListContainer.innerHTML = "";
            if (data.files && data.files.length > 0) {
                data.files.forEach(f => {
                    const li = document.createElement('li');
                    li.className = "file-item";
                    const sizeKb = (f.char_count || 0) / 1024;
                    li.innerHTML = `<span>📄 ${f.filename}</span><span class="file-size">${sizeKb.toFixed(1)} KB</span>`;
                    fileListContainer.appendChild(li);
                });
            } else {
                fileListContainer.innerHTML = `<li class="no-files">No files ingested. Specify a folder to start.</li>`;
            }
        } catch (err) {
            console.error("Failed loading context files", err);
        }
    }

    // Load initial file list
    loadCodebaseFileList();

    appendLog("GossetGate Interface initialized. WebGL 3D radar active.");
});
