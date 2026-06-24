/**
 * GossetGate: Chat Component
 */

export class ChatConsole {
    constructor(appendLog, onGridPointsUpdated, updateTelemetry) {
        this.appendLog = appendLog;
        this.onGridPointsUpdated = onGridPointsUpdated;
        this.updateTelemetry = updateTelemetry;
        this.currentEventSource = null;
        this.initDOMElements();
        this.bindEvents();
    }

    initDOMElements() {
        this.chatMessages = document.getElementById('chat-messages');
        this.chatInput = document.getElementById('chat-input');
        this.sendChatBtn = document.getElementById('send-chat-btn');
    }

    bindEvents() {
        if (this.sendChatBtn) {
            this.sendChatBtn.addEventListener('click', () => this.sendPrompt());
        }
        if (this.chatInput) {
            this.chatInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendPrompt();
                }
            });
        }
    }

    enable(loaded) {
        if (this.chatInput && this.sendChatBtn) {
            this.chatInput.disabled = !loaded;
            this.sendChatBtn.disabled = !loaded;
        }
    }

    sendPrompt() {
        const prompt = this.chatInput.value.trim();
        if (!prompt) return;

        this.chatInput.value = "";
        this.chatInput.disabled = true;
        this.sendChatBtn.disabled = true;

        this.appendChatBubble("User", prompt);
        const assistantBubbleObj = this.appendChatBubble("Assistant", "Thinking...");

        if (this.currentEventSource) {
            this.currentEventSource.close();
        }

        this.appendLog("Connecting SSE stream for response tokens...");
        this.currentEventSource = new EventSource(`/api/chat/stream?prompt=${encodeURIComponent(prompt)}`);

        let responseText = "";

        this.currentEventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.error) {
                assistantBubbleObj.textSpan.textContent = `[Error]: ${data.token}`;
                this.closeStream();
                return;
            }

            if (responseText === "") {
                assistantBubbleObj.textSpan.textContent = "";
            }

            responseText += data.token;
            
            // Normalize thought channel tags
            let normalizedText = responseText
                .replace(/<think>/gi, "<|channel>thought")
                .replace(/<\/think>/gi, "<channel|>text")
                .replace(/<thought>/gi, "<|channel>thought")
                .replace(/<\/thought>/gi, "<channel|>text");
            
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
                
                thoughtContent = thoughtParts.join("\n\n").replace(/<bos>|<eos>/gi, "").trim();
                textContent = textParts.join("\n\n")
                    .replace(/<bos>|<eos>/gi, "")
                    .replace(/<turn\|>|<\|turn>/gi, "")
                    .replace(/<\|channel>\s*thought/gi, "")
                    .replace(/<channel\|>|<\|channel>\s*text/gi, "")
                    .trim();
            } else {
                textContent = normalizedText.replace(/<bos>|<eos>/gi, "").replace(/<turn\|>|<\|turn>/gi, "").trim();
            }

            if (hasThought && assistantBubbleObj.thoughtDiv) {
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

            if (data.done === true) {
                this.closeStream();
                return;
            }
            
            if (this.chatMessages) {
                this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
            }

            // Sync telemetry metrics
            if (this.updateTelemetry) {
                this.updateTelemetry(data);
            }

            // Update E8 active grid coordinates
            if (data.grid_points && this.onGridPointsUpdated) {
                this.onGridPointsUpdated(data.grid_points);
            }

            // Print Phason logging
            if (data.logs && data.logs.length > 0) {
                data.logs.forEach(line => this.appendLog(line));
            }
        };

        this.currentEventSource.onerror = (err) => {
            console.error("SSE stream error", err);
            this.closeStream();
        };
    }

    closeStream() {
        if (this.currentEventSource) {
            this.currentEventSource.close();
            this.currentEventSource = null;
        }
        this.chatInput.disabled = false;
        this.sendChatBtn.disabled = false;
        this.appendLog("SSE Stream closed.");
    }

    appendChatBubble(sender, contentText) {
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
            thoughtDiv = document.createElement('div');
            thoughtDiv.className = 'thought-container hidden';

            thoughtHeader = document.createElement('div');
            thoughtHeader.className = 'thought-header';
            thoughtHeader.innerHTML = `<span>Thinking Process</span><span class="thought-toggle-icon">⚡</span>`;

            thoughtBody = document.createElement('div');
            thoughtBody.className = 'thought-body';

            thoughtDiv.appendChild(thoughtHeader);
            thoughtDiv.appendChild(thoughtBody);
            bubble.appendChild(thoughtDiv);

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

        if (this.chatMessages) {
            this.chatMessages.appendChild(bubble);
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }

        return {
            bubble: bubble,
            thoughtDiv: thoughtDiv,
            thoughtHeader: thoughtHeader,
            thoughtBody: thoughtBody,
            textSpan: responseTextSpan
        };
    }
}
