document.addEventListener('DOMContentLoaded', () => {
    // ---------------------------------------------------------
    // Tab Navigation
    // ---------------------------------------------------------
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            
            // Deactivate all
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanels.forEach(p => p.classList.remove('active'));
            
            // Activate target
            btn.classList.add('active');
            document.getElementById(`tab-${tabId}`).classList.add('active');
            
            // Re-render canvas if auditor tab is selected
            if (tabId === 'auditor') {
                resizeCanvas();
                drawGraph();
            }
        });
    });

    // ---------------------------------------------------------
    // Topological Call-Graph Visualizer (HTML5 Canvas)
    // ---------------------------------------------------------
    const canvas = document.getElementById('graph-canvas');
    const ctx = canvas.getContext('2d');
    
    let graphNodes = [];
    let graphEdges = [];
    let undefinedNodes = [];
    let nodePositions = {};
    let animationFrameId = null;
    let angleOffset = 0;

    function resizeCanvas() {
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
    }

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    function setupNodePositions() {
        nodePositions = {};
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        const radius = Math.min(canvas.width, canvas.height) * 0.3;
        
        const N = graphNodes.length;
        graphNodes.forEach((node, idx) => {
            // Distribute nodes evenly in a circle
            const angle = (idx * 2 * Math.PI) / N;
            nodePositions[node] = {
                x: cx + radius * Math.cos(angle),
                y: cy + radius * Math.sin(angle),
                angle: angle,
                radius: radius,
                name: node
            };
        });
    }

    function drawGraph() {
        if (!canvas.width || graphNodes.length === 0) return;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        
        // Update animated positions with small orbit wobble
        angleOffset += 0.003;
        
        const currentPositions = {};
        graphNodes.forEach(node => {
            const pos = nodePositions[node];
            if (!pos) return;
            const animatedAngle = pos.angle + angleOffset;
            const wobbleRadius = pos.radius + Math.sin(angleOffset * 5 + pos.angle) * 8;
            currentPositions[node] = {
                x: cx + wobbleRadius * Math.cos(animatedAngle),
                y: cy + wobbleRadius * Math.sin(animatedAngle)
            };
        });

        // 1. Draw Edges
        ctx.lineWidth = 1.5;
        graphEdges.forEach(([u, v]) => {
            const posU = currentPositions[u];
            const posV = currentPositions[v];
            if (posU && posV) {
                // If either node is undefined, draw as dashed red
                const isFractured = undefinedNodes.includes(u) || undefinedNodes.includes(v);
                if (isFractured) {
                    ctx.strokeStyle = 'rgba(255, 0, 0, 0.6)';
                    ctx.setLineDash([4, 4]);
                } else {
                    ctx.strokeStyle = 'rgba(0, 255, 255, 0.25)';
                    ctx.setLineDash([]);
                }
                ctx.beginPath();
                ctx.moveTo(posU.x, posU.y);
                ctx.lineTo(posV.x, posV.y);
                ctx.stroke();
            }
        });
        ctx.setLineDash([]); // Reset

        // 2. Draw Nodes
        graphNodes.forEach(node => {
            const pos = currentPositions[node];
            if (!pos) return;
            
            const isUndefined = undefinedNodes.includes(node);
            const isImport = !isUndefined && node.match(/^[a-z_][a-z0-9_]*$/) && !graphEdges.some(([u, v]) => u === node);
            
            let color = 'hsl(190, 100%, 50%)'; // default defined node (cyan)
            let shadowColor = 'rgba(0, 255, 255, 0.4)';
            
            if (isUndefined) {
                color = 'hsl(0, 100%, 55%)'; // Undefined logic fracture (neon red)
                shadowColor = 'rgba(255, 0, 0, 0.8)';
            } else if (isImport) {
                color = 'hsl(260, 100%, 65%)'; // Import node (neon purple)
                shadowColor = 'rgba(138, 43, 226, 0.5)';
            }
            
            // Draw glowing halo
            ctx.shadowBlur = 12;
            ctx.shadowColor = shadowColor;
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 8, 0, 2 * Math.PI);
            ctx.fill();
            
            // Reset shadow
            ctx.shadowBlur = 0;
            
            // Draw inner node core
            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 3.5, 0, 2 * Math.PI);
            ctx.fill();
            
            // Label
            ctx.fillStyle = 'hsl(210, 20%, 95%)';
            ctx.font = '11px Outfit, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(node, pos.x, pos.y - 14);
        });
    }

    function animateGraph() {
        drawGraph();
        animationFrameId = requestAnimationFrame(animateGraph);
    }

    // ---------------------------------------------------------
    // Run Cohomology Audit Action
    // ---------------------------------------------------------
    const runAuditBtn = document.getElementById('run-audit-btn');
    const codeInput = document.getElementById('code-input');
    const tauInput = document.getElementById('tau-input');
    const auditResults = document.getElementById('audit-results');
    const auditStatusBadge = document.getElementById('audit-status-badge');
    const auditScore = document.getElementById('audit-score');
    const auditMessage = document.getElementById('audit-message');

    runAuditBtn.addEventListener('click', async () => {
        runAuditBtn.disabled = true;
        runAuditBtn.innerText = 'Auditing...';
        
        try {
            const res = await fetch('/api/audit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    code: codeInput.value,
                    tau: parseFloat(tauInput.value) || 0.05
                })
            });
            
            const data = await res.json();
            
            // Update UI elements
            auditResults.classList.remove('hidden');
            if (data.approved) {
                auditStatusBadge.innerText = 'Approved';
                auditStatusBadge.className = 'badge approved';
            } else {
                auditStatusBadge.innerText = 'Rejected';
                auditStatusBadge.className = 'badge rejected';
            }
            auditScore.innerText = `Algebraic Connectivity (λ₂): ${data.lambda_2.toFixed(4)}`;
            auditMessage.innerText = data.message;
            
            // Update call-graph visualizer
            graphNodes = data.nodes;
            graphEdges = data.edges;
            undefinedNodes = data.undefined_nodes;
            
            setupNodePositions();
            
            if (animationFrameId) cancelAnimationFrame(animationFrameId);
            animateGraph();
            
        } catch (err) {
            console.error(err);
            alert('Error running cohomology audit backend.');
        } finally {
            runAuditBtn.disabled = false;
            runAuditBtn.innerText = 'Run Cohomology Audit';
        }
    });

    // Run initial audit on load to display graph
    runAuditBtn.click();

    // ---------------------------------------------------------
    // Self-Improvement Swarm Controller
    // ---------------------------------------------------------
    const startSiBtn = document.getElementById('start-si-btn');
    const siBackend = document.getElementById('si-backend');
    const siTarget = document.getElementById('si-target');
    const siGenerations = document.getElementById('si-generations');
    const consoleOutput = document.getElementById('console-output');
    
    const statBaseline = document.getElementById('stat-baseline');
    const statBest = document.getElementById('stat-best');
    const statSpeedup = document.getElementById('stat-speedup');
    
    let eventSource = null;

    startSiBtn.addEventListener('click', async () => {
        if (eventSource) {
            eventSource.close();
        }
        
        consoleOutput.innerText = 'Booting optimization loop server thread...\n';
        startSiBtn.disabled = true;
        startSiBtn.innerText = 'Loop Running...';
        
        try {
            // Start process
            const backend = siBackend.value;
            const target = siTarget.value;
            const generations = parseInt(siGenerations.value) || 3;
            
            await fetch(`/api/self-improve/start?backend=${backend}&generations=${generations}&target=${target}`, {
                method: 'POST'
            });
            
            // Stream logs
            eventSource = new EventSource('/api/self-improve/stream');
            eventSource.onmessage = (event) => {
                const line = event.data;
                consoleOutput.innerText += line + '\n';
                consoleOutput.scrollTop = consoleOutput.scrollHeight;
                
                // Parse stats dynamically from logs
                if (line.includes('Baseline mean latency:')) {
                    const match = line.match(/Baseline mean latency:\s+([0-9.]+)\s+ms/);
                    if (match) statBaseline.innerText = `${parseFloat(match[1]).toFixed(2)} ms`;
                }
                
                if (line.includes('SUCCESS: Latency reduced') || line.includes('Final best latency:')) {
                    const match = line.match(/latency:\s+([0-9.]+)\s+ms/i) || line.match(/reduced from [0-9.]+ ms to\s+([0-9.]+)\s+ms/);
                    if (match) {
                        const bestVal = parseFloat(match[1]);
                        statBest.innerText = `${bestVal.toFixed(2)} ms`;
                        
                        const baseVal = parseFloat(statBaseline.innerText);
                        if (!isNaN(baseVal) && baseVal > 0) {
                            const speedup = ((baseVal - bestVal) / baseVal) * 100;
                            statSpeedup.innerText = `${speedup.toFixed(2)}%`;
                        }
                    }
                }
                
                if (line.includes('Self-Improvement Loop complete')) {
                    eventSource.close();
                    startSiBtn.disabled = false;
                    startSiBtn.innerText = 'Launch Optimization Loop';
                }
            };
            
            eventSource.onerror = () => {
                // If disconnected/ended
                eventSource.close();
                startSiBtn.disabled = false;
                startSiBtn.innerText = 'Launch Optimization Loop';
            };
            
        } catch (err) {
            console.error(err);
            consoleOutput.innerText += `\nERROR: Could not establish SSE connection to loop controller.\n`;
            startSiBtn.disabled = false;
            startSiBtn.innerText = 'Launch Optimization Loop';
        }
    });

    // ---------------------------------------------------------
    // Interactive Chat Controller
    // ---------------------------------------------------------
    const chatInput = document.getElementById('chat-input');
    const sendChatBtn = document.getElementById('send-chat-btn');
    const chatMessages = document.getElementById('chat-messages');

    function appendMessage(sender, text, type) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${type}`;
        
        const senderSpan = document.createElement('span');
        senderSpan.className = 'msg-sender';
        senderSpan.innerText = `${sender}:`;
        
        const textSpan = document.createElement('span');
        textSpan.className = 'msg-text';
        textSpan.innerText = text;
        
        msgDiv.appendChild(senderSpan);
        msgDiv.appendChild(textSpan);
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return textSpan;
    }

    async function sendQuery() {
        const query = chatInput.value.trim();
        if (!query) return;
        
        chatInput.value = '';
        appendMessage('User', query, 'user');
        
        // Append placeholder for QAN streaming response
        const textPlaceholder = appendMessage('QAN-ATLAS', 'Connecting to E8 search index...', 'qan');
        
        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: query })
            });
            
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            
            textPlaceholder.innerText = '';
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const content = line.substring(6);
                        textPlaceholder.innerText += content + '\n';
                    }
                }
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
        } catch (err) {
            console.error(err);
            textPlaceholder.innerText = 'Error querying the model backend.';
        }
    }

    sendChatBtn.addEventListener('click', sendQuery);
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') sendQuery();
    });
});
