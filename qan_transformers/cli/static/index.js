/**
 * GossetGate: Quasicrystalline Attention Portal Interaction Logic
 *
 * Implements:
 * 1. 3D projection, depth sorting, and rendering of the E8 Root Lattice.
 * 2. Canvas-based Coxeter Projected Concentric Shell [2, 30, 64, 64, 80] grouping.
 * 3. Interactive HUD overlay tracking node properties on hover.
 * 4. Step-by-step SSE model loader parsing fetch ReadableStreams.
 * 5. Diagnostic panel for gated repos and out-of-memory errors.
 * 6. File ingestion and chat query SSE telemetries.
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- Global State ---
    let activeIndices = new Set();
    let currentEventSource = null;
    let isModelLoaded = false;
    
    // Canvas rotation parameters
    let rotationAngleX = 0.002;
    let rotationAngleY = 0.004;
    let rotationAngleZ = 0.001;
    let currentRotX = 0;
    let currentRotY = 0;
    let currentRotZ = 0;
    let animationFrameId = null;
    
    // Cache render list for hover calculations
    let lastRenderList = [];

    // --- DOM Elements ---
    // Tour Toggle
    const tourToggleBtn = document.getElementById('tour-toggle-btn');
    const tourContent = document.getElementById('tour-content');
    const tourChevron = document.getElementById('tour-chevron');
    const tourPanel = document.querySelector('.tour-panel');

    // Load Inputs
    const targetModelInput = document.getElementById('target-model-input');
    const draftStrategy = document.getElementById('draft-strategy');
    const draftModelGroup = document.getElementById('draft-model-group');
    const draftModelInput = document.getElementById('draft-model-input');
    const modelPrecision = document.getElementById('model-precision');
    const modelFramework = document.getElementById('model-framework');
    const mockModelCheckbox = document.getElementById('mock-model-checkbox');
    const loadModelBtn = document.getElementById('load-model-btn');

    // Modal Progress Elements
    const loadModal = document.getElementById('load-modal');
    const loadProgressFill = document.getElementById('load-progress-fill');
    const loadModalLogs = document.getElementById('load-modal-logs');
    const modalCloseBtn = document.getElementById('modal-close-btn');

    // Diagnostics Elements
    const diagnosticsModal = document.getElementById('diagnostics-modal');
    const diagErrorMsg = document.getElementById('diag-error-msg');
    const diagTraceback = document.getElementById('diag-traceback');
    const guideHfAuth = document.getElementById('guide-hf-auth');
    const guideOom = document.getElementById('guide-oom');
    const diagCloseBtn = document.getElementById('diag-close-btn');

    // Controls
    const sparseRatioSlider = document.getElementById('sparse-ratio-slider');
    const sparseRatioVal = document.getElementById('sparse-ratio-val');
    const firewallThresholdSlider = document.getElementById('firewall-threshold-slider');
    const firewallThresholdVal = document.getElementById('firewall-threshold-val');
    const firewallToggle = document.getElementById('firewall-toggle');
    const reviewToggle = document.getElementById('review-toggle');
    const thinkingToggle = document.getElementById('thinking-toggle');
    const telemetryToggle = document.getElementById('telemetry-toggle');
    const maxTokensSlider = document.getElementById('max-tokens-slider');
    const maxTokensVal = document.getElementById('max-tokens-val');

    // Ingestion
    const contextInput = document.getElementById('context-input');
    const contextFile = document.getElementById('context-file');
    const uploadedFileName = document.getElementById('uploaded-file-name');
    const ingestContextBtn = document.getElementById('ingest-context-btn');
    const ingestStats = document.getElementById('ingest-stats');
    const ingestTokensLbl = document.getElementById('ingest-tokens-lbl');
    const ingestTimeLbl = document.getElementById('ingest-time-lbl');

    // Chat
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendChatBtn = document.getElementById('send-chat-btn');

    // Telemetry
    const metricSpeed = document.getElementById('metric-speed');
    const metricVram = document.getElementById('metric-vram');
    const metricCells = document.getElementById('metric-cells');
    const metricL2 = document.getElementById('metric-l2');
    const metricContext = document.getElementById('metric-context');
    const metricAcceptance = document.getElementById('metric-acceptance');
    const logConsole = document.getElementById('log-console');

    // Canvas & HUD
    const canvas = document.getElementById('e8-canvas');
    const ctx = canvas.getContext('2d');
    const e8Hud = document.getElementById('e8-hud');
    const hudRoot = document.getElementById('hud-root');
    const hudProj = document.getElementById('hud-proj');
    const hudShell = document.getElementById('hud-shell');
    const hudStatus = document.getElementById('hud-status');

    // Layout Elements
    const toggleSidebarBtn = document.getElementById('toggle-sidebar-btn');
    const toggleVizBtn = document.getElementById('toggle-viz-btn');
    const sidebarPanel = document.getElementById('sidebar-panel');
    const visualizationPanel = document.getElementById('visualization-panel');

    // --- Onboarding & Tour Panel Toggle ---
    tourToggleBtn.addEventListener('click', () => {
        const isCollapsed = tourContent.classList.contains('hidden');
        if (isCollapsed) {
            tourContent.classList.remove('hidden');
            tourPanel.classList.add('active');
            tourChevron.innerHTML = "&#x25B2;"; // Up Chevron
        } else {
            tourContent.classList.add('hidden');
            tourPanel.classList.remove('active');
            tourChevron.innerHTML = "&#x25BC;"; // Down Chevron
        }
    });

    // --- Sidebar & Telemetry Panel Toggles ---
    function updateLayoutButton(btn, panel) {
        if (panel.classList.contains('collapsed')) {
            btn.classList.remove('active');
        } else {
            btn.classList.add('active');
        }
    }

    if (toggleSidebarBtn && sidebarPanel) {
        toggleSidebarBtn.addEventListener('click', () => {
            sidebarPanel.classList.toggle('collapsed');
            updateLayoutButton(toggleSidebarBtn, sidebarPanel);
        });
    }

    if (toggleVizBtn && visualizationPanel) {
        toggleVizBtn.addEventListener('click', () => {
            visualizationPanel.classList.toggle('collapsed');
            updateLayoutButton(toggleVizBtn, visualizationPanel);
            
            // Trigger canvas resize once transition finishes
            setTimeout(() => {
                if (typeof handleResize === 'function') {
                    handleResize();
                }
            }, 305);
        });
    }

    // Keyboard Shortcuts (Ctrl+B / Ctrl+Shift+V)
    // H21: Changed telemetry toggle from Ctrl+V to Ctrl+Shift+V to avoid
    // hijacking the system paste shortcut.
    window.addEventListener('keydown', (e) => {
        // Ignore if typing in text fields
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            return;
        }
        
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b') {
            e.preventDefault();
            if (toggleSidebarBtn) toggleSidebarBtn.click();
        }
        if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'v') {
            e.preventDefault();
            if (toggleVizBtn) toggleVizBtn.click();
        }
    });

    // --- Vertical Collapsible Card Accordions ---
    document.addEventListener('click', (e) => {
        const header = e.target.closest('.collapsible-card .card-header');
        if (!header) return;
        
        // Skip toggling if clicking inside an interactive element inside the header
        if (e.target.closest('button') || e.target.closest('a') || e.target.closest('input') || e.target.closest('select')) {
            return;
        }
        
        const card = header.closest('.collapsible-card');
        if (card) {
            card.classList.toggle('collapsed');
        }
    });

    // --- Strategy Selector Toggle ---
    draftStrategy.addEventListener('change', () => {
        if (draftStrategy.value === 'single') {
            draftModelGroup.classList.add('hidden');
        } else {
            draftModelGroup.classList.remove('hidden');
        }
    });

    // --- Range Sliders & Config syncing ---
    sparseRatioSlider.addEventListener('input', (e) => {
        sparseRatioVal.textContent = parseFloat(e.target.value).toFixed(2);
    });
    sparseRatioSlider.addEventListener('change', syncBackendConfiguration);

    firewallThresholdSlider.addEventListener('input', (e) => {
        firewallThresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
    });
    firewallThresholdSlider.addEventListener('change', syncBackendConfiguration);

    firewallToggle.addEventListener('click', () => {
        firewallToggle.classList.toggle('active');
        syncBackendConfiguration();
    });

    reviewToggle.addEventListener('click', () => {
        reviewToggle.classList.toggle('active');
        syncBackendConfiguration();
    });

    thinkingToggle.addEventListener('click', () => {
        thinkingToggle.classList.toggle('active');
        syncBackendConfiguration();
    });

    telemetryToggle.addEventListener('click', () => {
        telemetryToggle.classList.toggle('active');
        syncBackendConfiguration();
    });

    maxTokensSlider.addEventListener('input', (e) => {
        maxTokensVal.textContent = parseInt(e.target.value);
    });
    maxTokensSlider.addEventListener('change', syncBackendConfiguration);

    async function syncBackendConfiguration() {
        const payload = {
            sparse_ratio: parseFloat(sparseRatioSlider.value),
            firewall_enabled: firewallToggle.classList.contains('active'),
            review_mode: reviewToggle.classList.contains('active'),
            threshold: parseFloat(firewallThresholdSlider.value),
            thinking_mode: thinkingToggle.classList.contains('active') ? "thinking" : "direct",
            max_new_tokens: parseInt(maxTokensSlider.value),
            optimize_telemetry: telemetryToggle.classList.contains('active')
        };
        try {
            const res = await fetch('/api/config/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (res.ok) {
                appendLog(`Config synchronized: sparse=${payload.sparse_ratio}, firewall=${payload.firewall_enabled}, threshold=${payload.threshold}, thinking=${payload.thinking_mode}, max_tokens=${payload.max_new_tokens}, low_telemetry=${payload.optimize_telemetry}`);
            }
        } catch (err) {
            console.error("Failed syncing configs to backend", err);
        }
    }

    // --- Helper function: Append Logs ---
    function appendLog(msg) {
        const now = new Date().toLocaleTimeString();
        logConsole.textContent += `\n[${now}] ${msg}`;
        // H24: Cap log console to prevent unbounded memory growth
        if (logConsole.textContent.length > 50000) {
            logConsole.textContent = logConsole.textContent.slice(-25000);
        }
        logConsole.scrollTop = logConsole.scrollHeight;
    }

    // --- SSE-based Model Loading Checklist ---
    loadModelBtn.addEventListener('click', async () => {
        const targetModel = targetModelInput.value.trim();
        const draftModel = draftModelInput.value.trim();
        const useSpeculative = (draftStrategy.value === 'speculative');
        const sparseRatio = parseFloat(sparseRatioSlider.value);
        const lightweightMock = mockModelCheckbox.checked;
        const precisionValue = modelPrecision.value;
        const frameworkValue = modelFramework ? modelFramework.value : 'mlx';

        if (!targetModel) {
            alert("Please input a valid Hugging Face target model.");
            return;
        }

        // Initialize Loading Panel Checklist UI
        loadModalLogs.textContent = "Connecting to initialization stream...\n";
        loadProgressFill.style.width = "0%";
        loadModal.classList.remove('hidden');
        modalCloseBtn.disabled = true;

        // Reset checkmark items
        const checklistItems = ['device', 'tokenizer', 'target_model', 'swap_db', 'firewall', 'draft_model'];
        checklistItems.forEach(id => {
            const el = document.getElementById(`chk-${id}`);
            if (el) {
                el.className = "checklist-item";
                el.querySelector('.status-icon').innerHTML = "&#x25CB;"; // Pending circle
            }
        });

        // Toggle draft checklist visibility
        const draftCheckEl = document.getElementById('chk-draft_model');
        if (useSpeculative) {
            draftCheckEl.classList.remove('hidden');
        } else {
            draftCheckEl.classList.add('hidden');
        }

        const stepsToComplete = useSpeculative ? 6 : 5;
        let completedSteps = 0;

        try {
            const response = await fetch('/api/model/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    target_model: targetModel,
                    draft_model: draftModel,
                    use_speculative: useSpeculative,
                    sparse_ratio: sparseRatio,
                    lightweight_mock: lightweightMock,
                    precision: precisionValue,
                    framework: frameworkValue
                })
            });

            if (!response.body) {
                throw new Error("No readable body stream returned by model initialization API.");
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop(); // Hold onto partial line chunk

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const payload = JSON.parse(line.substring(6));

                        if (payload.type === 'step') {
                            const stepEl = document.getElementById(`chk-${payload.step}`);
                            const iconEl = stepEl.querySelector('.status-icon');
                            
                            if (payload.status === 'running') {
                                stepEl.className = "checklist-item running";
                                iconEl.innerHTML = "&#x21BB;"; // Spinning arrows indicator
                                iconEl.classList.add('running');
                            } else if (payload.status === 'success') {
                                stepEl.className = "checklist-item success";
                                iconEl.innerHTML = "&#x2713;"; // Green checkmark
                                iconEl.classList.remove('running');
                                completedSteps++;
                                loadProgressFill.style.width = `${(completedSteps / stepsToComplete) * 100}%`;
                            }
                            
                            loadModalLogs.textContent += `[INFO] ${payload.message}\n`;
                            loadModalLogs.scrollTop = loadModalLogs.scrollHeight;
                        } 
                        else if (payload.type === 'complete') {
                            // Finished spec
                            if (useSpeculative) {
                                const draftEl = document.getElementById('chk-draft_model');
                                draftEl.className = "checklist-item success";
                                draftEl.querySelector('.status-icon').innerHTML = "&#x2713;";
                                draftEl.querySelector('.status-icon').classList.remove('running');
                            }
                            loadProgressFill.style.width = "100%";
                            loadModalLogs.textContent += `\n[SUCCESS] Model engine ready: ${payload.message}\n`;
                            loadModalLogs.scrollTop = loadModalLogs.scrollHeight;
                            
                            // Enable controls
                            isModelLoaded = true;
                            loadModelBtn.textContent = "Re-initialize Model Engine";
                            chatInput.disabled = false;
                            sendChatBtn.disabled = false;
                            ingestContextBtn.disabled = false;
                            modalCloseBtn.disabled = false;

                            // Update Header labels
                            const cleanName = targetModel.split('/').pop();
                            document.getElementById('status-model-lbl').textContent = `Model: ${cleanName}` + (useSpeculative ? ` + Speculative` : ``);
                            document.getElementById('status-model-dot').className = "status-dot green";

                            document.getElementById('status-db-lbl').textContent = "E8 Grid: Standby";
                            document.getElementById('status-db-dot').className = "status-dot green";

                            // Auto-close modal
                            setTimeout(() => {
                                loadModal.classList.add('hidden');
                            }, 800);
                        } 
                        else if (payload.type === 'error') {
                            // Show error checklist item
                            checklistItems.forEach(id => {
                                const el = document.getElementById(`chk-${id}`);
                                if (el.classList.contains('running')) {
                                    el.className = "checklist-item failed";
                                    el.querySelector('.status-icon').innerHTML = "&#x2717;";
                                    el.querySelector('.status-icon').classList.remove('running');
                                }
                            });

                            loadModalLogs.textContent += `\n[ERROR] Model loading failed: ${payload.message}\n`;
                            loadModalLogs.scrollTop = loadModalLogs.scrollHeight;
                            modalCloseBtn.disabled = false;

                            // Trigger Diagnostics overlay
                            setTimeout(() => {
                                loadModal.classList.add('hidden');
                                showDiagnosticsOverlay(payload.error_type, payload.message, payload.traceback);
                            }, 1000);
                        }
                    }
                }
            }
        } catch (err) {
            loadModalLogs.textContent += `\n[CRITICAL] Stream connection error: ${err.message}\n`;
            modalCloseBtn.disabled = false;
            showDiagnosticsOverlay("UNKNOWN_LOAD_ERROR", err.message, "Network or server connection failed.");
        }
    });

    modalCloseBtn.addEventListener('click', () => {
        loadModal.classList.add('hidden');
    });

    // --- Diagnostics Overlay Actions ---
    function showDiagnosticsOverlay(errType, message, traceback) {
        diagErrorMsg.textContent = message;
        diagTraceback.textContent = traceback || "No stack trace returned.";

        // Toggle helpful advice boxes
        guideHfAuth.classList.add('hidden');
        guideOom.classList.add('hidden');

        if (errType === 'HF_AUTH_ERROR') {
            guideHfAuth.classList.remove('hidden');
        } else if (errType === 'OUT_OF_MEMORY') {
            guideOom.classList.remove('hidden');
        }

        diagnosticsModal.classList.remove('hidden');
    }

    diagCloseBtn.addEventListener('click', () => {
        diagnosticsModal.classList.add('hidden');
    });

    // --- File Ingestion CRUD ---
    const ingestedFilesSection = document.getElementById('ingested-files-section');
    const ingestedFileList = document.getElementById('ingested-file-list');

    function renderFileList(files) {
        if (!ingestedFileList) return;
        ingestedFileList.innerHTML = "";
        
        if (!files || files.length === 0) {
            ingestedFilesSection.classList.add('hidden');
            document.getElementById('status-db-lbl').textContent = "E8 Grid: Empty (0 tokens)";
            document.getElementById('status-db-dot').className = "status-dot red";
            if (metricContext) metricContext.textContent = "0";
            return;
        }

        ingestedFilesSection.classList.remove('hidden');
        let totalTokens = files.reduce((acc, f) => acc + f.tokens, 0);
        document.getElementById('status-db-lbl').textContent = `E8 Grid: Locked (${totalTokens} tokens)`;
        document.getElementById('status-db-dot').className = "status-dot cyan";
        if (metricContext) metricContext.textContent = totalTokens;

        files.forEach(f => {
            const li = document.createElement('li');
            li.className = 'file-item';

            const metaDiv = document.createElement('div');
            metaDiv.className = 'file-meta';
            
            const icon = document.createElement('span');
            icon.className = 'file-meta-icon';
            icon.textContent = '📄';
            
            const name = document.createElement('span');
            name.className = 'file-meta-name';
            name.textContent = f.filename;
            name.title = f.filename;

            metaDiv.appendChild(icon);
            metaDiv.appendChild(name);
            li.appendChild(metaDiv);

            const badge = document.createElement('span');
            badge.className = 'file-tokens-badge';
            badge.textContent = `${f.tokens} T`;
            li.appendChild(badge);

            const removeBtn = document.createElement('button');
            removeBtn.type = 'button';
            removeBtn.className = 'remove-file-btn';
            removeBtn.innerHTML = '&times;';
            removeBtn.title = 'Remove Document';
            removeBtn.setAttribute('aria-label', `Remove document ${f.filename}`);
            removeBtn.addEventListener('click', async () => {
                await deleteIngestedFile(f.filename);
            });
            li.appendChild(removeBtn);

            ingestedFileList.appendChild(li);
        });
    }

    function renderFileTree(tree, container) {
        if (!container) return;
        container.innerHTML = "";
        
        function buildTreeDOM(node, name, depth) {
            const wrapper = document.createElement('div');
            wrapper.style.paddingLeft = `${depth * 12}px`;
            wrapper.style.marginTop = '2px';
            
            if (node.type === "file") {
                const item = document.createElement('div');
                item.className = 'tree-file-item';
                item.style.display = 'flex';
                item.style.justifyContent = 'space-between';
                item.style.alignItems = 'center';
                
                const spanName = document.createElement('span');
                spanName.textContent = `📄 ${name}`;
                spanName.style.color = '#e2e2e2';
                spanName.style.overflow = 'hidden';
                spanName.style.textOverflow = 'ellipsis';
                spanName.style.whiteSpace = 'nowrap';
                
                const metaSpan = document.createElement('span');
                metaSpan.textContent = `(${node.tokens} T, ${(node.size / 1024).toFixed(1)} KB)`;
                metaSpan.style.color = '#7e8e9f';
                metaSpan.style.fontSize = '0.7rem';
                metaSpan.style.marginLeft = '8px';
                
                item.appendChild(spanName);
                item.appendChild(metaSpan);
                wrapper.appendChild(item);
            } else {
                // Directory node
                const folderHeader = document.createElement('div');
                folderHeader.className = 'tree-folder-header';
                folderHeader.style.cursor = 'pointer';
                folderHeader.style.color = '#ffd700'; // gold directories
                folderHeader.style.userSelect = 'none';
                folderHeader.style.fontWeight = 'bold';
                
                const folderChevron = document.createElement('span');
                folderChevron.textContent = '▼ ';
                folderChevron.style.fontSize = '0.7rem';
                folderChevron.style.marginRight = '4px';
                
                const folderName = document.createElement('span');
                folderName.textContent = `📁 ${name}`;
                
                folderHeader.appendChild(folderChevron);
                folderHeader.appendChild(folderName);
                wrapper.appendChild(folderHeader);
                
                const childrenContainer = document.createElement('div');
                childrenContainer.className = 'tree-folder-children';
                
                const children = node.children || {};
                Object.keys(children).sort().forEach(childName => {
                    childrenContainer.appendChild(buildTreeDOM(children[childName], childName, depth + 1));
                });
                
                wrapper.appendChild(childrenContainer);
                
                folderHeader.addEventListener('click', () => {
                    const isCollapsed = childrenContainer.style.display === 'none';
                    childrenContainer.style.display = isCollapsed ? 'block' : 'none';
                    folderChevron.textContent = isCollapsed ? '▼ ' : '▶ ';
                });
            }
            return wrapper;
        }
        
        Object.keys(tree).sort().forEach(name => {
            container.appendChild(buildTreeDOM(tree[name], name, 0));
        });
    }

    async function fetchIngestedFiles() {
        try {
            const res = await fetch('/api/context/files');
            if (res.ok) {
                const data = await res.json();
                renderFileList(data.files);
                if (data.tree) {
                    renderFileTree(data.tree, document.getElementById('file-tree-root'));
                }
            }
        } catch (err) {
            console.error("Error fetching ingested files list:", err);
        }
    }

    async function deleteIngestedFile(filename) {
        appendLog(`Removing document context '${filename}'...`);
        try {
            const res = await fetch(`/api/context/files/${encodeURIComponent(filename)}`, {
                method: 'DELETE'
            });
            const data = await res.json();
            if (data.status === 'success') {
                appendLog(`Successfully removed document context '${filename}'. Remaining tokens: ${data.remaining_tokens}`);
                fetchIngestedFiles();
            } else {
                appendLog(`ERROR removing context: ${data.message}`);
            }
        } catch (err) {
            appendLog(`CRITICAL connection error during deletion: ${err.message}`);
        }
    }

    // Call on load to retrieve active files
    fetchIngestedFiles();

    contextFile.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            uploadedFileName.textContent = file.name;
            const reader = new FileReader();
            reader.onload = (evt) => {
                contextInput.value = evt.target.result;
                appendLog(`Selected local file loaded: ${file.name} (${evt.target.result.length} characters)`);
            };
            reader.readAsText(file);
        } else {
            uploadedFileName.textContent = "No file loaded";
        }
    });

    ingestContextBtn.addEventListener('click', async () => {
        const text = contextInput.value.trim();
        if (!text) {
            appendLog("Warning: Cannot ingest empty text document.");
            return;
        }

        const filename = uploadedFileName.textContent !== "No file loaded" 
            ? uploadedFileName.textContent 
            : `document_${ingestedFileList.children.length + 1}.txt`;

        ingestContextBtn.disabled = true;
        ingestContextBtn.textContent = "Ingesting...";
        appendLog(`Ingesting context '${filename}' (${text.length} chars) into Conway-Sloane grid swap cache...`);

        try {
            const res = await fetch('/api/context/ingest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, filename })
            });

            const data = await res.json();
            if (data.status === 'success') {
                appendLog(`Prefill finished: Ingested '${filename}' (${data.tokens} tokens) into E8 lattice db cache.`);
                
                // Show Ingest metrics
                ingestStats.classList.remove('hidden');
                ingestTokensLbl.textContent = `Tokens Ingested: ${data.tokens}`;
                ingestTimeLbl.textContent = `Prefill Time: ${data.prefill_time_sec.toFixed(2)}s`;

                // Update documents list
                fetchIngestedFiles();
                
                // Clear inputs
                contextInput.value = "";
                uploadedFileName.textContent = "No file loaded";
                contextFile.value = "";
            } else {
                appendLog(`ERROR ingesting context: ${data.message}`);
            }
        } catch (err) {
            appendLog(`CRITICAL connection error during ingestion: ${err.message}`);
        } finally {
            ingestContextBtn.disabled = false;
            ingestContextBtn.textContent = "Ingest Document Context";
        }
    });

    // --- Prompt Template Ingest controls ---
    const templateHeader = document.getElementById('template-collapsible-header');
    const templateChevron = document.getElementById('template-chevron');
    const templateContent = document.getElementById('template-collapsible-content');
    const templateInput = document.getElementById('template-input');
    const saveTemplateBtn = document.getElementById('save-template-btn');

    if (templateHeader) {
        templateHeader.addEventListener('click', () => {
            const isHidden = templateContent.classList.contains('hidden');
            if (isHidden) {
                templateContent.classList.remove('hidden');
                templateChevron.textContent = '▼';
            } else {
                templateContent.classList.add('hidden');
                templateChevron.textContent = '▸';
            }
        });
    }

    async function fetchPromptTemplate() {
        try {
            const res = await fetch('/api/context/template');
            if (res.ok) {
                const data = await res.json();
                if (templateInput) {
                    templateInput.value = data.template;
                }
            }
        } catch (err) {
            console.error("Error fetching prompt template:", err);
        }
    }

    if (saveTemplateBtn) {
        saveTemplateBtn.addEventListener('click', async () => {
            const updatedTemplate = templateInput.value;
            saveTemplateBtn.disabled = true;
            saveTemplateBtn.textContent = "Saving...";
            try {
                const res = await fetch('/api/context/template', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ template: updatedTemplate })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    appendLog("System prompt template updated successfully.");
                    if (data.prefill_time_sec) {
                        appendLog(`Prefilled XML context under new template in ${data.prefill_time_sec.toFixed(2)}s (${data.total_tokens} tokens).`);
                        fetchIngestedFiles();
                    }
                } else {
                    appendLog(`Error updating template: ${data.message}`);
                }
            } catch (err) {
                appendLog(`Error updating template: ${err.message}`);
            } finally {
                saveTemplateBtn.disabled = false;
                saveTemplateBtn.textContent = "Save Template";
            }
        });
    }

    // Call on load to retrieve active template
    fetchPromptTemplate();
 
    // --- Hugging Face Downloader controls ---
    const hfRepoInput = document.getElementById('hf-repo-input');
    const downloadModelBtn = document.getElementById('download-model-btn');
    const downloadProgressContainer = document.getElementById('download-progress-container');
    const downloadFileName = document.getElementById('download-file-name');
    const downloadPercentLbl = document.getElementById('download-percent-lbl');
    const downloadProgressFill = document.getElementById('download-progress-fill');
    const downloadLogs = document.getElementById('download-logs');
    const targetModelsList = document.getElementById('target-models-list');
    let downloadEventSource = null;

    if (downloadModelBtn) {
        downloadModelBtn.addEventListener('click', () => {
            const repoId = hfRepoInput.value.trim();
            if (!repoId) {
                alert("Please enter a valid Hugging Face Repository ID.");
                return;
            }

            // Disable downloader controls
            downloadModelBtn.disabled = true;
            downloadModelBtn.textContent = "Downloading...";
            downloadProgressContainer.classList.remove('hidden');
            downloadFileName.textContent = "Connecting to download stream...";
            downloadPercentLbl.textContent = "0%";
            downloadProgressFill.style.width = "0%";
            downloadProgressFill.style.background = "linear-gradient(90deg, #b026ff, #00e6ff)";
            downloadLogs.textContent = `[INFO] Initializing download stream for ${repoId}...\n`;

            if (downloadEventSource) {
                downloadEventSource.close();
            }

            downloadEventSource = new EventSource(`/api/model/download/stream?repo_id=${encodeURIComponent(repoId)}`);

            downloadEventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'ping') {
                    return;
                }

                if (data.type === 'info') {
                    downloadLogs.textContent += `[INFO] ${data.message}\n`;
                    downloadLogs.scrollTop = downloadLogs.scrollHeight;
                    downloadFileName.textContent = data.message;
                }
                else if (data.type === 'progress') {
                    downloadFileName.textContent = `${data.file} (${data.speed})`;
                    downloadPercentLbl.textContent = `${data.percent}%`;
                    downloadProgressFill.style.width = `${data.percent}%`;
                }
                else if (data.type === 'success') {
                    downloadLogs.textContent += `\n[SUCCESS] ${data.message}\nPath: ${data.path}\n`;
                    downloadLogs.scrollTop = downloadLogs.scrollHeight;
                    downloadFileName.textContent = "Download complete!";
                    downloadPercentLbl.textContent = "100%";
                    downloadProgressFill.style.width = "100%";
                    
                    appendLog(`Hugging Face model '${repoId}' successfully downloaded.`);
                    
                    // Add to options datalist if not already there
                    let exists = false;
                    for (const option of targetModelsList.options) {
                        if (option.value === repoId) {
                            exists = true;
                            break;
                        }
                    }
                    if (!exists) {
                        const opt = document.createElement('option');
                        opt.value = repoId;
                        targetModelsList.appendChild(opt);
                    }
                    
                    // Autofill model selector
                    targetModelInput.value = repoId;
                    
                    // Reset UI
                    downloadEventSource.close();
                    downloadModelBtn.disabled = false;
                    downloadModelBtn.textContent = "Download Model";
                }
                else if (data.type === 'error') {
                    downloadLogs.textContent += `\n[ERROR] ${data.message}\n${data.traceback || ''}\n`;
                    downloadLogs.scrollTop = downloadLogs.scrollHeight;
                    downloadFileName.textContent = "Download failed.";
                    downloadPercentLbl.textContent = "Failed";
                    downloadProgressFill.style.background = "#ff4a4a";
                    downloadProgressFill.style.width = "100%";
                    
                    appendLog(`ERROR downloading Hugging Face model '${repoId}': ${data.message}`);
                    
                    downloadEventSource.close();
                    downloadModelBtn.disabled = false;
                    downloadModelBtn.textContent = "Download Model";
                }
            };

            downloadEventSource.onerror = (err) => {
                console.error("Download SSE connection error:", err);
                downloadLogs.textContent += `\n[CRITICAL] Connection lost or stream error. Check terminal logs.\n`;
                downloadLogs.scrollTop = downloadLogs.scrollHeight;
                
                downloadEventSource.close();
                downloadModelBtn.disabled = false;
                downloadModelBtn.textContent = "Download Model";
            };
        });
    }

    // --- Interactive Chat Stream ---
    function appendChatBubble(sender, contentText) {
        const bubble = document.createElement('div');
        bubble.className = `message ${sender.toLowerCase()}`;
        
        const senderSpan = document.createElement('span');
        senderSpan.className = "msg-sender";
        senderSpan.textContent = `${sender}:`;
        bubble.appendChild(senderSpan);

        let thoughtDiv = null;
        let thoughtHeader = null;
        let thoughtBody = null;
        let responseTextSpan = null;

        if (sender === "Assistant") {
            // Pre-create thought-container structures
            thoughtDiv = document.createElement('div');
            thoughtDiv.className = 'thought-container hidden'; // hidden by default

            thoughtHeader = document.createElement('div');
            thoughtHeader.className = 'thought-header';
            thoughtHeader.innerHTML = `<span>Thinking Process</span><span class="thought-toggle-icon">⚡</span>`;

            thoughtBody = document.createElement('div');
            thoughtBody.className = 'thought-body';

            thoughtDiv.appendChild(thoughtHeader);
            thoughtDiv.appendChild(thoughtBody);
            bubble.appendChild(thoughtDiv);

            // Toggle listener
            thoughtHeader.addEventListener('click', () => {
                const icon = thoughtHeader.querySelector('.thought-toggle-icon');
                if (bubble.classList.contains('collapsed-thought')) {
                    bubble.classList.remove('collapsed-thought');
                    thoughtBody.classList.remove('hidden');
                    icon.textContent = '▼';
                } else {
                    bubble.classList.add('collapsed-thought');
                    thoughtBody.classList.add('hidden');
                    icon.textContent = '▶';
                }
            });

            // Pre-create responseTextSpan
            responseTextSpan = document.createElement('span');
            responseTextSpan.className = 'msg-text';
            responseTextSpan.textContent = contentText;
            bubble.appendChild(responseTextSpan);
        } else {
            responseTextSpan = document.createElement('span');
            responseTextSpan.className = 'msg-text';
            responseTextSpan.textContent = contentText;
            bubble.appendChild(responseTextSpan);
        }

        chatMessages.appendChild(bubble);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        return {
            bubble: bubble,
            thoughtDiv: thoughtDiv,
            thoughtHeader: thoughtHeader,
            thoughtBody: thoughtBody,
            textSpan: responseTextSpan
        };
    }

    sendChatBtn.addEventListener('click', () => {
        const prompt = chatInput.value.trim();
        if (!prompt) return;

        chatInput.value = "";
        chatInput.disabled = true;
        sendChatBtn.disabled = true;

        appendChatBubble("User", prompt);
        const assistantBubbleObj = appendChatBubble("Assistant", "Thinking...");

        if (currentEventSource) {
            currentEventSource.close();
        }

        appendLog("Connecting SSE stream for response tokens...");
        currentEventSource = new EventSource(`/api/chat/stream?prompt=${encodeURIComponent(prompt)}`);

        let responseText = "";

        currentEventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.error) {
                assistantBubbleObj.textSpan.textContent = `[Error]: ${data.token}`;
                currentEventSource.close();
                chatInput.disabled = false;
                sendChatBtn.disabled = false;
                return;
            }

            // Clean initial placeholder
            if (responseText === "") {
                assistantBubbleObj.textSpan.textContent = "";
            }

            responseText += data.token;
            
            // Normalize various model thinking/text channel tags for unified parsing
            let normalizedText = responseText
                .replace(/<think>/gi, "<|channel>thought")
                .replace(/<\/think>/gi, "<channel|>text")
                .replace(/<thought>/gi, "<|channel>thought")
                .replace(/<\/thought>/gi, "<channel|>text");
            
            // Parse thinking process vs response using regex
            let thoughtContent = "";
            let textContent = "";
            let hasThought = false;
            let isThinking = false;

            const thoughtTag = /<\|channel>\s*thought/gi;
            if (normalizedText.match(thoughtTag)) {
                hasThought = true;
                let remaining = normalizedText;
                let thoughtParts = [];
                let textParts = [];
                
                while (true) {
                    const thoughtMatch = remaining.match(/<\|channel>\s*thought/i);
                    if (!thoughtMatch) {
                        if (remaining.trim()) {
                            textParts.push(remaining);
                        }
                        break;
                    }
                    
                    const beforeThought = remaining.substring(0, thoughtMatch.index);
                    if (beforeThought.trim()) {
                        textParts.push(beforeThought);
                    }
                    
                    const startIndex = thoughtMatch.index + thoughtMatch[0].length;
                    const afterThought = remaining.substring(startIndex);
                    
                    const textMatch = afterThought.match(/<channel\|>|<\|channel>\s*text/i);
                    if (textMatch) {
                        const thoughtVal = afterThought.substring(0, textMatch.index);
                        thoughtParts.push(thoughtVal);
                        remaining = afterThought.substring(textMatch.index + textMatch[0].length);
                        isThinking = false;
                    } else {
                        thoughtParts.push(afterThought);
                        isThinking = true;
                        remaining = "";
                        break;
                    }
                }
                
                thoughtContent = thoughtParts.join("\n\n")
                    .replace(/<bos>|<eos>/gi, "")
                    .trim();
                textContent = textParts.join("\n\n")
                    .replace(/<bos>|<eos>/gi, "")
                    .replace(/<turn\|>|<\|turn>/gi, "")
                    .replace(/<\|channel>\s*thought/gi, "")
                    .replace(/<channel\|>|<\|channel>\s*text/gi, "")
                    .trim();
            } else {
                textContent = normalizedText
                    .replace(/<bos>|<eos>/gi, "")
                    .replace(/<turn\|>|<\|turn>/gi, "")
                    .trim();
            }

            if (hasThought) {
                if (assistantBubbleObj.thoughtDiv) {
                    assistantBubbleObj.thoughtDiv.classList.remove('hidden');
                    assistantBubbleObj.thoughtBody.textContent = thoughtContent;
                    const icon = assistantBubbleObj.thoughtHeader.querySelector('.thought-toggle-icon');
                    if (icon) {
                        if (isThinking) {
                            icon.textContent = '⚡';
                        } else if (assistantBubbleObj.bubble.classList.contains('collapsed-thought')) {
                            icon.textContent = '▶';
                        } else {
                            icon.textContent = '▼';
                        }
                    }
                }
            }

            if (hasThought && isThinking && !textContent) {
                if (data.done) {
                    assistantBubbleObj.textSpan.textContent = "[Generation stopped during thinking process]";
                    assistantBubbleObj.textSpan.style.fontStyle = "italic";
                    assistantBubbleObj.textSpan.style.opacity = "0.5";
                } else {
                    assistantBubbleObj.textSpan.textContent = "Thinking...";
                    assistantBubbleObj.textSpan.style.fontStyle = "italic";
                    assistantBubbleObj.textSpan.style.opacity = "0.6";
                }
            } else {
                assistantBubbleObj.textSpan.textContent = textContent || (normalizedText ? "" : "Thinking...");
                assistantBubbleObj.textSpan.style.fontStyle = "";
                assistantBubbleObj.textSpan.style.opacity = "";
            }

            // C8: Close EventSource when generation is complete to prevent
            // the chat from locking permanently on a finished stream.
            if (data.done === true) {
                currentEventSource.close();
                currentEventSource = null;
                chatInput.disabled = false;
                sendChatBtn.disabled = false;

                // Soft-clear highlighted paged elements after completion
                setTimeout(() => {
                    activeIndices.clear();
                }, 1500);
                return;
            }
            
            chatMessages.scrollTop = chatMessages.scrollHeight;

            // Stream telemetries
            metricSpeed.textContent = data.speed.toFixed(1);
            metricVram.textContent = `${data.vram_saved.toFixed(0)}%`;
            metricCells.textContent = data.active_cells;
            metricL2.textContent = isNaN(data.lambda_2) ? "---" : data.lambda_2.toFixed(3);
            if (data.context_size !== undefined) {
                metricContext.textContent = data.context_size;
            }
            if (data.speculative && data.acceptance_rate !== undefined) {
                metricAcceptance.textContent = `${data.acceptance_rate.toFixed(1)}%`;
            } else {
                metricAcceptance.textContent = "N/A";
            }

            // Connective alert
            if (data.is_fractured) {
                metricL2.style.color = "#ff3b6f";
                metricL2.style.textShadow = "0 0 8px #ff3b6f";
            } else {
                metricL2.style.color = "";
                metricL2.style.textShadow = "";
            }

            // Sync active E8 visualizer coordinates
            if (data.grid_points) {
                activeIndices.clear();
                data.grid_points.forEach(pt => {
                    let closestIdx = -1;
                    let minDistance = 1e9;

                    e8CoordinatesList.forEach((item) => {
                        const dx = item.pt3d[0] - pt[0];
                        const dy = item.pt3d[1] - pt[1];
                        const dz = item.pt3d[2] - pt[2];
                        const dist = dx*dx + dy*dy + dz*dz;
                        if (dist < minDistance) {
                            minDistance = dist;
                            closestIdx = item.idx;
                        }
                    });

                    // Tag active page mapping
                    if (closestIdx !== -1 && minDistance < 0.25) {
                        activeIndices.add(closestIdx);
                    }
                });
            }

            // Print Phason logging ticks
            if (data.logs && data.logs.length > 0) {
                data.logs.forEach(line => appendLog(line));
            }
        };

        currentEventSource.onerror = (err) => {
            appendLog("SSE chat response stream finished.");
            currentEventSource.close();
            currentEventSource = null;
            chatInput.disabled = false;
            sendChatBtn.disabled = false;

            // If still in default 'Thinking...' placeholder, mark it as stopped
            if (assistantBubbleObj.textSpan.textContent === "Thinking...") {
                assistantBubbleObj.textSpan.textContent = "[Generation stopped during thinking process]";
                assistantBubbleObj.textSpan.style.fontStyle = "italic";
                assistantBubbleObj.textSpan.style.opacity = "0.5";
            }

            // Soft-clear highlighted paged elements
            setTimeout(() => {
                activeIndices.clear();
            }, 1500);
        };
    });

    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendChatBtn.click();
        }
    });


    // --- 3D Concentric E8 Coxeter Projection Engine ---
    
    // Generates the 240 root vectors of E8 root system
    function getE8Roots() {
        const roots = [];
        
        // 1. Permutations of (+-1, +-1, 0, 0, 0, 0, 0, 0)
        for (let i = 0; i < 8; i++) {
            for (let j = i + 1; j < 8; j++) {
                for (const s1 of [-1, 1]) {
                    for (const s2 of [-1, 1]) {
                        const v = new Array(8).fill(0);
                        v[i] = s1;
                        v[j] = s2;
                        roots.push(v);
                    }
                }
            }
        }
        
        // 2. (+-1/2, ..., +-1/2) with an even number of minus signs
        for (let bits = 0; bits < 256; bits++) {
            const signs = [];
            let negatives = 0;
            for (let i = 0; i < 8; i++) {
                const s = (bits & (1 << i)) !== 0 ? 1 : -1;
                signs.push(s);
                if (s === -1) negatives++;
            }
            if (negatives % 2 === 0) {
                const v = signs.map(s => s * 0.5);
                roots.push(v);
            }
        }
        return roots;
    }

    // Projects 8D coordinates to 3D via Coxeter-Icosian matrix math
    function getIcosianProjection3D(roots) {
        const phi = (1.0 + Math.sqrt(5.0)) / 2.0;
        const scale = 1.0 / Math.sqrt(1.0 + phi * phi);
        const projected = [];
        
        roots.forEach((v, idx) => {
            const x = (v[0] * phi + v[4]) * scale;
            const y = (v[1] * phi + v[5]) * scale;
            const z = (v[2] * phi + v[6]) * scale;
            projected.push({ pt3d: [x, y, z], original: v, idx: idx });
        });
        return projected;
    }

    const rawE8Roots = getE8Roots();
    const e8CoordinatesList = getIcosianProjection3D(rawE8Roots);

    // Group roots into concentric shells based on Euclidean norm radius
    e8CoordinatesList.forEach(item => {
        const [x, y, z] = item.pt3d;
        item.norm = Math.sqrt(x*x + y*y + z*z);
    });
    // Sort by radius for concentric shell grouping.
    // M25: This sort reorders the original E8 root indices, which could destroy
    // index stability. However, the nearest-neighbor matching used in the active
    // coordinate sync (minDistance < 0.25) mitigates this instability by matching
    // on 3D projected position rather than relying on index identity.
    e8CoordinatesList.sort((a, b) => a.norm - b.norm);

    // Replicate concentric shells counts: [2, 30, 64, 64, 80]
    const shellGroups = [[], [], [], [], []];
    e8CoordinatesList.forEach((node, index) => {
        let shIdx = 0;
        if (index < 2) shIdx = 0;
        else if (index < 32) shIdx = 1;
        else if (index < 96) shIdx = 2;
        else if (index < 160) shIdx = 3;
        else shIdx = 4;
        
        node.shellIdx = shIdx;
        shellGroups[shIdx].push(node);
    });

    // Rotation helper
    function rotatePoint(x, y, z, ax, ay, az) {
        // Rot X
        let cos = Math.cos(ax), sin = Math.sin(ax);
        let y1 = y * cos - z * sin;
        let z1 = y * sin + z * cos;
        // Rot Y
        cos = Math.cos(ay); sin = Math.sin(ay);
        let x2 = x * cos + z1 * sin;
        let z2 = -x * sin + z1 * cos;
        // Rot Z
        cos = Math.cos(az); sin = Math.sin(az);
        let x3 = x2 * cos - y1 * sin;
        let y3 = x2 * sin + y1 * cos;
        
        return [x3, y3, z2];
    }

    // Color schema for E8 shell nodes
    const shellColors = [
        'rgba(255, 0, 102, 0.85)',   // Shell 0 (Pink poles)
        'rgba(0, 243, 255, 0.85)',   // Shell 1 (Cyan)
        'rgba(255, 170, 0, 0.85)',   // Shell 2 (Gold)
        'rgba(138, 43, 226, 0.85)',  // Shell 3 (Purple)
        'rgba(0, 255, 136, 0.85)'    // Shell 4 (Emerald green)
    ];

    function drawCanvasFrame() {
        const dpr = window.devicePixelRatio || 1;
        const w = canvas.width / dpr;
        const h = canvas.height / dpr;
        
        ctx.clearRect(0, 0, w, h);
        
        const centerX = w / 2;
        const centerY = h / 2;
        const fitScale = Math.min(w, h) * 0.40;
        const fov = 350;

        currentRotX += rotationAngleX;
        currentRotY += rotationAngleY;
        currentRotZ += rotationAngleZ;

        const renderList = [];

        e8CoordinatesList.forEach(node => {
            const [x, y, z] = node.pt3d;
            const [rx, ry, rz] = rotatePoint(x, y, z, currentRotX, currentRotY, currentRotZ);
            
            const persp = fov / (fov + rz);
            const sx = centerX + rx * persp * fitScale;
            const sy = centerY + ry * persp * fitScale;
            
            const isActive = activeIndices.has(node.idx);

            renderList.push({
                sx: sx,
                sy: sy,
                sz: rz,
                persp: persp,
                isActive: isActive,
                nodeRef: node
            });
        });

        // Depth sorting (painter's algorithm)
        renderList.sort((a, b) => b.sz - a.sz);
        lastRenderList = renderList; // Cache for hover interaction

        // 1. Draw connecting Coxeter lines
        ctx.lineWidth = 0.4;
        for (let i = 0; i < renderList.length; i++) {
            const p1 = renderList[i];
            let connections = 0;
            
            for (let j = i + 1; j < renderList.length; j++) {
                if (connections > 2) break;
                const p2 = renderList[j];
                
                if (p1.nodeRef.shellIdx !== p2.nodeRef.shellIdx) continue;

                // Calculate distance on 3D space first to draw only local geometric links
                const dx3 = p1.nodeRef.pt3d[0] - p2.nodeRef.pt3d[0];
                const dy3 = p1.nodeRef.pt3d[1] - p2.nodeRef.pt3d[1];
                const dz3 = p1.nodeRef.pt3d[2] - p2.nodeRef.pt3d[2];
                const dist3 = Math.sqrt(dx3*dx3 + dy3*dy3 + dz3*dz3);

                if (dist3 < 1.1) {
                    ctx.strokeStyle = p1.isActive || p2.isActive 
                        ? 'rgba(0, 243, 255, 0.35)' 
                        : 'rgba(255, 255, 255, 0.05)';
                    
                    ctx.beginPath();
                    ctx.moveTo(p1.sx, p1.sy);
                    ctx.lineTo(p2.sx, p2.sy);
                    ctx.stroke();
                    connections++;
                }
            }
        }

        // 2. Draw nodes
        renderList.forEach(p => {
            const size = p.isActive ? 6.5 : (p.nodeRef.shellIdx === 0 ? 5 : 3.5);
            const radius = Math.max(1.2, size * p.persp);
            
            ctx.beginPath();
            ctx.arc(p.sx, p.sy, radius, 0, 2 * Math.PI);
            
            if (p.isActive) {
                const glowPulse = 6 + Math.sin(Date.now() / 60) * 3;
                ctx.shadowColor = 'rgba(0, 243, 255, 1)';
                ctx.shadowBlur = glowPulse;
                ctx.fillStyle = '#ffffff';
            } else {
                ctx.shadowBlur = 0;
                ctx.fillStyle = shellColors[p.nodeRef.shellIdx];
            }
            
            ctx.fill();
            ctx.shadowBlur = 0;
        });

        animationFrameId = requestAnimationFrame(drawCanvasFrame);
    }

    // Initialize Canvas drawing loop
    drawCanvasFrame();

    // Resize handlers
    function handleResize() {
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        // C9: Use setTransform instead of scale to prevent DPR accumulation on resize.
        // ctx.scale(dpr, dpr) was additive and would compound on every resize call.
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    handleResize();
    window.addEventListener('resize', handleResize);

    // L15: Cancel animation frame on page unload to prevent orphaned rAF loops
    window.addEventListener('beforeunload', () => {
        if (animationFrameId !== null) {
            cancelAnimationFrame(animationFrameId);
        }
    });

    // --- Interactive 3D Canvas Hover HUD handler ---
    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        let closestItem = null;
        let minDist = 12; // Search radius (in pixels)

        lastRenderList.forEach(item => {
            const dx = item.sx - mouseX;
            const dy = item.sy - mouseY;
            const dist = Math.sqrt(dx*dx + dy*dy);
            if (dist < minDist) {
                minDist = dist;
                closestItem = item;
            }
        });

        if (closestItem) {
            const node = closestItem.nodeRef;
            
            // Format 8D root vector beautifully
            const formatted8D = '[' + node.original.map(n => {
                if (n === 0.5) return '½';
                if (n === -0.5) return '-½';
                return n;
            }).join(', ') + ']';

            const formatted3D = `(${node.pt3d[0].toFixed(2)}, ${node.pt3d[1].toFixed(2)}, ${node.pt3d[2].toFixed(2)})`;
            
            // Populate HUD
            hudRoot.textContent = formatted8D;
            hudProj.textContent = formatted3D;
            hudShell.textContent = `Shell ${node.shellIdx} (r = ${node.norm.toFixed(2)})`;
            
            const isPaged = activeIndices.has(node.idx);
            hudStatus.textContent = isPaged ? "ACTIVE (VRAM Cache)" : "INACTIVE (Host RAM)";
            hudStatus.style.color = isPaged ? "var(--neon-cyan)" : "var(--text-muted)";

            // Position HUD overlay box
            e8Hud.style.left = `${mouseX + 12}px`;
            e8Hud.style.top = `${mouseY + 12}px`;
            e8Hud.classList.remove('hidden');
        } else {
            e8Hud.classList.add('hidden');
        }
    });

    canvas.addEventListener('mouseleave', () => {
        e8Hud.classList.add('hidden');
    });

    // --- Tab Switching Logic ---
    const tabChatBtn = document.getElementById('tab-chat-btn');
    const tabOptimizeBtn = document.getElementById('tab-optimize-btn');
    const mainLayout = document.querySelector('.main-layout');
    const selfImprovementLayout = document.getElementById('self-improvement-layout');

    if (tabChatBtn && tabOptimizeBtn && mainLayout && selfImprovementLayout) {
        tabChatBtn.addEventListener('click', () => {
            tabChatBtn.classList.add('active');
            tabOptimizeBtn.classList.remove('active');
            mainLayout.classList.remove('hidden');
            selfImprovementLayout.classList.add('hidden');
            if (tourPanel) tourPanel.classList.remove('hidden');
        });

        tabOptimizeBtn.addEventListener('click', () => {
            tabOptimizeBtn.classList.add('active');
            tabChatBtn.classList.remove('active');
            mainLayout.classList.add('hidden');
            selfImprovementLayout.classList.remove('hidden');
            if (tourPanel) tourPanel.classList.add('hidden');
            
            // Resize audit canvas
            resizeAuditCanvas();
        });
    }

    // --- Code Audit & AST Canvas Visualizer ---
    const auditCodeInput = document.getElementById('audit-code-input');
    const auditTauSlider = document.getElementById('audit-tau-slider');
    const auditTauVal = document.getElementById('audit-tau-val');
    const runAuditBtn = document.getElementById('run-audit-btn');
    const auditResultsContainer = document.getElementById('audit-results-container');
    const auditStatusBadge = document.getElementById('audit-status-badge');
    const auditLambda2Val = document.getElementById('audit-lambda2-val');
    const auditMsg = document.getElementById('audit-msg');
    const auditCanvas = document.getElementById('ast-graph-canvas');
    const auditCtx = auditCanvas ? auditCanvas.getContext('2d') : null;

    if (auditTauSlider && auditTauVal) {
        auditTauSlider.addEventListener('input', (e) => {
            auditTauVal.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    function resizeAuditCanvas() {
        if (!auditCanvas) return;
        const container = auditCanvas.parentElement;
        auditCanvas.width = container.clientWidth;
        auditCanvas.height = container.clientHeight || 300;
        drawAuditGraph([], []);
    }

    if (runAuditBtn) {
        runAuditBtn.addEventListener('click', async () => {
            const code = auditCodeInput.value.trim();
            const tau = parseFloat(auditTauSlider.value);
            if (!code) {
                alert("Please paste some Python source code first.");
                return;
            }

            runAuditBtn.disabled = true;
            runAuditBtn.textContent = "Analyzing...";
            appendLog("Sending AST code audit request to cohomology validator...");

            try {
                const res = await fetch('/api/audit', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ code, tau })
                });
                
                if (res.ok) {
                    const data = await res.json();
                    
                    // Show results
                    auditResultsContainer.classList.remove('hidden');
                    if (data.approved) {
                        auditStatusBadge.textContent = "Approved";
                        auditStatusBadge.className = "audit-status-badge approved";
                    } else {
                        auditStatusBadge.textContent = "Fractured";
                        auditStatusBadge.className = "audit-status-badge fractured";
                    }
                    auditLambda2Val.textContent = data.lambda_2.toFixed(4);
                    auditMsg.textContent = data.message;
                    
                    appendLog(`Audit complete: approved=${data.approved}, lambda2=${data.lambda_2.toFixed(4)}`);
                    
                    // Draw Graph
                    drawAuditGraph(data.nodes, data.edges, data.undefined_nodes || []);
                } else {
                    alert("Failed to analyze connectivity.");
                }
            } catch (err) {
                console.error(err);
                alert("Error during audit execution.");
            } finally {
                runAuditBtn.disabled = false;
                runAuditBtn.textContent = "Analyze Connectivity";
            }
        });
    }

    function drawAuditGraph(nodes, edges, undefinedNodes) {
        if (!auditCtx || !auditCanvas) return;
        const W = auditCanvas.width;
        const H = auditCanvas.height;
        auditCtx.clearRect(0, 0, W, H);

        if (!nodes || nodes.length === 0) {
            // Draw empty state message
            auditCtx.fillStyle = "rgba(255, 255, 255, 0.3)";
            auditCtx.font = "14px 'Outfit', sans-serif";
            auditCtx.textAlign = "center";
            auditCtx.fillText("No graph computed. Paste code and analyze to visualize AST structure.", W / 2, H / 2);
            return;
        }

        // Layout nodes in a circle
        const cx = W / 2;
        const cy = H / 2;
        const radius = Math.min(W, H) * 0.35;
        const nodePositions = {};

        nodes.forEach((node, idx) => {
            const angle = (idx * 2 * Math.PI) / nodes.length;
            nodePositions[node] = {
                x: cx + radius * Math.cos(angle),
                y: cy + radius * Math.sin(angle)
            };
        });

        // Draw edges
        auditCtx.strokeStyle = "rgba(255, 255, 255, 0.15)";
        auditCtx.lineWidth = 1.5;
        edges.forEach(([src, dst]) => {
            const p1 = nodePositions[src];
            const p2 = nodePositions[dst];
            if (p1 && p2) {
                auditCtx.beginPath();
                auditCtx.moveTo(p1.x, p1.y);
                auditCtx.lineTo(p2.x, p2.y);
                auditCtx.stroke();
            }
        });

        // Draw nodes
        nodes.forEach((node) => {
            const pos = nodePositions[node];
            const isUndefined = undefinedNodes.includes(node);
            
            // Outer glow
            auditCtx.shadowBlur = 10;
            auditCtx.shadowColor = isUndefined ? "rgba(255, 75, 75, 0.8)" : "rgba(0, 242, 254, 0.8)";
            
            // Draw circle
            auditCtx.fillStyle = isUndefined ? "rgba(255, 75, 75, 0.9)" : "rgba(0, 242, 254, 0.9)";
            auditCtx.beginPath();
            auditCtx.arc(pos.x, pos.y, 6, 0, 2 * Math.PI);
            auditCtx.fill();
            
            // Label
            auditCtx.shadowBlur = 0; // reset
            auditCtx.fillStyle = "rgba(255, 255, 255, 0.8)";
            auditCtx.font = "10px 'Fira Code', monospace";
            auditCtx.textAlign = "center";
            auditCtx.fillText(node, pos.x, pos.y - 10);
        });
    }

    // --- Self-Improvement Loop ---
    const siBackendSelect = document.getElementById('si-backend-select');
    const siTargetSelect = document.getElementById('si-target-select');
    const siGenerationsInput = document.getElementById('si-generations-input');
    const startSiBtn = document.getElementById('start-si-btn');
    const siStatusIndicator = document.getElementById('si-status-indicator');
    const siLogConsole = document.getElementById('si-log-console');
    let siEventSource = null;

    if (startSiBtn) {
        startSiBtn.addEventListener('click', async () => {
            const backend = siBackendSelect.value;
            const target = siTargetSelect.value;
            const generations = parseInt(siGenerationsInput.value) || 3;

            startSiBtn.disabled = true;
            startSiBtn.textContent = "Starting...";
            siLogConsole.textContent = "Connecting to background execution thread...\n";
            siStatusIndicator.textContent = "Status: Initializing";
            
            try {
                const url = `/api/self-improve/start?backend=${backend}&generations=${generations}&target=${target}`;
                const res = await fetch(url, { method: 'POST' });
                const data = await res.json();
                
                if (data.status === 'started' || data.status === 'already_running') {
                    siStatusIndicator.textContent = "Status: Executing";
                    siStatusIndicator.style.color = "var(--neon-cyan)";
                    startSiBtn.textContent = "Running Loop";
                    
                    // Connect stream
                    if (siEventSource) {
                        siEventSource.close();
                    }
                    
                    siEventSource = new EventSource('/api/self-improve/stream');
                    siEventSource.onmessage = (e) => {
                        siLogConsole.textContent += `\n${e.data}`;
                        siLogConsole.scrollTop = siLogConsole.scrollHeight;
                        
                        if (e.data.includes("=== Self-Improvement Loop Complete ===")) {
                            siStatusIndicator.textContent = "Status: Complete";
                            siStatusIndicator.style.color = "var(--neon-green)";
                            startSiBtn.disabled = false;
                            startSiBtn.textContent = "Start Self-Improvement Loop";
                            siEventSource.close();
                        }
                    };
                } else {
                    alert("Could not start loop.");
                    startSiBtn.disabled = false;
                    startSiBtn.textContent = "Start Self-Improvement Loop";
                }
            } catch (err) {
                console.error(err);
                alert("Error calling self-improve API.");
                startSiBtn.disabled = false;
                startSiBtn.textContent = "Start Self-Improvement Loop";
            }
        });
    }

    // Auto-check status on load
    async function checkSiStatus() {
        try {
            const res = await fetch('/api/self-improve/status');
            const data = await res.json();
            if (data.running) {
                siStatusIndicator.textContent = "Status: Executing";
                siStatusIndicator.style.color = "var(--neon-cyan)";
                if (startSiBtn) {
                    startSiBtn.disabled = true;
                    startSiBtn.textContent = "Running Loop";
                }
                
                siEventSource = new EventSource('/api/self-improve/stream');
                siEventSource.onmessage = (e) => {
                    siLogConsole.textContent += `\n${e.data}`;
                    siLogConsole.scrollTop = siLogConsole.scrollHeight;
                    if (e.data.includes("=== Self-Improvement Loop Complete ===")) {
                        siStatusIndicator.textContent = "Status: Complete";
                        siStatusIndicator.style.color = "var(--neon-green)";
                        if (startSiBtn) {
                            startSiBtn.disabled = false;
                            startSiBtn.textContent = "Start Self-Improvement Loop";
                        }
                        siEventSource.close();
                    }
                };
            }
        } catch (err) {
            console.error("Failed checking SI status", err);
        }
    }
    
    checkSiStatus();

    appendLog("GossetGate Interface initial completion successful. WebGL-emulated canvas root projector online.");
});
