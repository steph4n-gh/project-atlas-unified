# scripts/build_static_demos.py
# Extracts the retro 90s pages from scripts/serve_moe.py and saves them as static HTML pages in docs/
# Injects a client-side mock fetch interceptor so that they run fully interactively on GitHub Pages.

import os
import sys

# Ensure we can import from workspace
sys.path.insert(0, os.getcwd())

from scripts.serve_moe import HTML_PAGE, MOE_HTML_PAGE, CHAT_HTML_PAGE

# Define the fetch interceptor javascript
INTERCEPTOR = """
    <!-- Global fetch interceptor for static/GitHub Pages demo mode -->
    <script>
      (function() {
        const isStatic = window.location.hostname.includes("github.io") || 
                         window.location.protocol === "file:" ||
                         (window.location.hostname.includes("localhost") === false && window.location.hostname.includes("127.0.0.1") === false);
                         
        if (isStatic) {
          console.log("[Project Atlas] Static demo mode activated. Intercepting API routes.");
          const originalFetch = window.fetch;
          window.fetch = async function(url, options) {
            // Intercept /api/generate (Regex Wizard)
            if (url.includes("/api/generate")) {
              let body = {};
              try {
                body = JSON.parse(options.body);
              } catch(e) {}
              const prompt = body.prompt || "";
              const dialect = body.dialect || "JS";
              const prompt_l = prompt.toLowerCase();
              
              let pattern = "/^[a-zA-Z0-9]{3,8}$/";
              let sample = "sample_data\\nother_data";
              let description = "Matches standard alphanumeric characters.";
              let expert = "gateway_router";
              
              if (prompt_l.includes("email")) {
                pattern = dialect === "JS" ? "/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\\\.[a-zA-Z]{2,}$/" : "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\\\\\.[a-zA-Z]{2,}$";
                sample = "chattel_lupine8j@icloud.com\\ntest@example.org\\ninvalid_address";
                description = "Standard RFC 5322 email regex compiler routing.";
                expert = "markdown_config";
              } else if (prompt_l.includes("phone")) {
                pattern = dialect === "JS" ? "/^\\\\+?[1-9]\\\\d{1,14}$/" : "^\\\\+?[1-9]\\\\d{1,14}$";
                sample = "+12345678901\\n555-0199\\nabc";
                description = "E.164 international phone number format detector.";
                expert = "python_coder";
              } else if (prompt_l.includes("ip") || prompt_l.includes("ipv4")) {
                pattern = dialect === "JS" ? "/^(?:[0-9]{1,3}\\\\\\.){3}[0-9]{1,3}$/" : "^(?:[0-9]{1,3}\\\\\\.){3}[0-9]{1,3}$";
                sample = "192.168.1.1\\n8.8.8.8\\n999.999.999.999";
                description = "Matches standard IPv4 addresses with routing checkpoints.";
                expert = "devops_infra";
              } else if (prompt_l.includes("zip") || prompt_l.includes("postal")) {
                pattern = dialect === "JS" ? "/^\\\\d{5}(-\\\\d{4})?$/" : "^\\\\d{5}(-\\\\d{4})?$";
                sample = "90210\\n12345-6789\\nabcde";
                description = "US Zip codes (5-digit or 9-digit) spatial locator.";
                expert = "markdown_config";
              } else if (prompt_l.includes("uuid") || prompt_l.includes("guid")) {
                pattern = dialect === "JS" ? "/^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/" : "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$";
                sample = "123e4567-e89b-12d3-a456-426614174000\\ninvalid-uuid";
                description = "Matches DCE UUID / GUID hex values.";
                expert = "devops_infra";
              } else if (prompt_l.includes("tag") || prompt_l.includes("html") || prompt_l.includes("xml")) {
                pattern = dialect === "JS" ? "/<[^>]+>/" : "<[^>]+>";
                sample = "<div>\\n<a href='index.html'>\\n</p>";
                description = "Markup element start/end tag parser.";
                expert = "web_stack";
              } else {
                if (prompt_l.includes("def") || prompt_l.includes("class") || prompt_l.includes("code")) {
                  pattern = "/def\\\\s+\\\\w+\\\\(/";
                  sample = "def main():\\nclass Test:\\nno code";
                  description = "Identified Python function definition keyword sequence.";
                  expert = "python_coder";
                } else if (prompt_l.includes("select") || prompt_l.includes("join") || prompt_l.includes("where")) {
                  pattern = "/SELECT\\\\s+.*?\\\\s+FROM/";
                  sample = "SELECT * FROM users;\\nINSERT INTO table;\\ninvalid query";
                  description = "Structured Query Language (SQL) data retrieval parser.";
                  expert = "database_sql";
                } else if (prompt_l.includes("fn") || prompt_l.includes("impl") || prompt_l.includes("rust")) {
                  pattern = "/fn\\\\s+\\\\w+\\\\s*\\\\(/";
                  sample = "fn test() {\\nimpl Model {\\nno rust";
                  description = "Rust systems function or implementation block.";
                  expert = "rust_systems";
                }
              }
              
              const bars = [];
              for (let i = 0; i < 8; i++) {
                bars.push(Math.floor(Math.random() * 80) + 10);
              }
              
              const active_balls = 32 + (prompt.length % 12);
              const routing_ms = Math.round((Math.random() * 0.8 + 0.1) * 100) / 100;
              const generation_ms = Math.round((10 + prompt.length * 0.1) * 100) / 100;
              
              // Simulate small delay
              await new Promise(r => setTimeout(r, 400));
              
              return new Response(JSON.stringify({
                status: "success",
                expert: expert,
                pattern: pattern,
                sample: sample,
                bars: bars,
                description: description,
                routing_latency_ms: routing_ms,
                swap_latency_ms: (expert !== "gateway_router" ? 3.48 : 0.0),
                active_path_balls: active_balls,
                generation_latency_ms: generation_ms
              }), {
                status: 200,
                headers: { "Content-Type": "application/json" }
              });
            }
            
            // Intercept /api/chat (Multi-Expert Portal)
            if (url.includes("/api/chat")) {
              let body = {};
              try {
                body = JSON.parse(options.body);
              } catch(e) {}
              const prompt = body.prompt || "";
              const prompt_l = prompt.toLowerCase();
              
              let expert_name = "web_stack";
              if (prompt_l.includes("def") || prompt_l.includes("class") || prompt_l.includes("code")) {
                expert_name = "python_coder";
              } else if (prompt_l.includes("select") || prompt_l.includes("join") || prompt_l.includes("where")) {
                expert_name = "database_sql";
              } else if (prompt_l.includes("docker") || prompt_l.includes("ip") || prompt_l.includes("port")) {
                expert_name = "devops_infra";
              } else if (prompt_l.includes("fn") || prompt_l.includes("impl") || prompt_l.includes("nested")) {
                expert_name = "rust_systems";
              }
              
              const funny_prefixes = {
                "python_coder": "Synthesizing Python module... Routing expert completed. Target sequence: ",
                "web_stack": "Parsing HTML DOM headers... Executing virtual javascript engine. Response: ",
                "rust_systems": "Validating lifetimes and memory safety bounds... Cargo package built successfully: ",
                "database_sql": "Running query plan analyzer on B-Tree indices... Query results: ",
                "devops_infra": "Initiating Kubernetes cluster config validation... Router packets dispatched: ",
                "markdown_config": "Compiling structured YAML configuration maps... Nodes resolved: ",
                "gateway_router": "Subspace gating network resolved. Gateway routed expert sequence: "
              };
              const prefix = funny_prefixes[expert_name] || "Expert processed prompt successfully: ";
              
              const mock_words = [
                "inletsSeverity withnncжина____ inड्रो__CLEARமையில்ථාжинаமையில் with__𝚢牆மையில்",
                "with চিৎকার with anam Blondeமையில்𝚢牆牆udel with𝚢def from in in",
                "for khái with with Đối in neglecting",
                "besondere weber with withOfThe𝚢 with inimheomag",
                "while from 🌾 in ĐốiFlatten pół hydrazineSeverity inமையில்",
                "analyzerdef with ंगलीiginTableViewCell ComteSeverity"
              ];
              const generated_raw = mock_words[Math.floor(Math.random() * mock_words.length)];
              const response_text = prefix + generated_raw + "<br><br><span style='color:#808080;font-size:10px;font-family:monospace;'>*beep* [synthesizing sound registers via routed expert: " + expert_name.toUpperCase() + " in 12.34ms] *boop*</span>";
              
              const active_balls = 32 + (prompt.length % 12);
              const routing_ms = Math.round((Math.random() * 0.8 + 0.1) * 100) / 100;
              
              await new Promise(r => setTimeout(r, 600));
              
              return new Response(JSON.stringify({
                status: "success",
                expert: expert_name,
                response: response_text,
                routing_latency_ms: routing_ms,
                swap_latency_ms: 3.48,
                active_path_balls: active_balls,
                generation_latency_ms: 12.34
              }), {
                status: 200,
                headers: { "Content-Type": "application/json" }
              });
            }
            
            // Intercept /api/chat_persona (AIM Chat)
            if (url.includes("/api/chat_persona")) {
              let body = {};
              try {
                body = JSON.parse(options.body);
              } catch(e) {}
              const prompt = body.prompt || "";
              const persona = body.persona || "netrunner95";
              const warn_level = body.warn_level || 0;
              const prompt_l = prompt.toLowerCase();
              
              let response_text = "";
              
              if (warn_level >= 70) {
                const glitch_prefixes = [
                  "[GLITCH_LEVEL_RED] p-adic tree node mapping collision! ",
                  "[CRITICAL] E8 Lattice coordinate overflow: ",
                  "[WARNING] coordinate salad engaged! ",
                  "[SYSTEM FATAL] memory leak in subspace gating node: "
                ];
                const glitch_words = [
                  "0x93FF20AA182B", "cantor_dust", "subspace_routing_resolved",
                  "finite_tree_coordinates", "inletsSeverity", "word_salad_imminent",
                  "GOSSET_LATTICE_CRASH", "VRAM_OVERFLOW_SLOT_3", "NULL_POINTER_CE"
                ];
                response_text = glitch_prefixes[Math.floor(Math.random() * glitch_prefixes.length)] + 
                                glitch_words.sort(() => 0.5 - Math.random()).slice(0, 5).join(" ").toUpperCase() + " !!!";
              } else if (warn_level >= 30) {
                const annoyed = {
                  "netrunner95": "hey, stop warning me or i will crash your netscape browser! i'm trying to bypass the y2k clock here.",
                  "latticelover": "your warnings are introducing non-ultrametric noise into my cantor set. please cease.",
                  "chiptunegameboy": "*bzzzzt* warning threshold critical! pitch bend registers overloaded. stop it!",
                  "e8_lattice_core": "WARNING DETECTED. COORDINATE DRIFT ACTIVE. RESOLUTION FAILED."
                };
                response_text = annoyed[persona] || "system warning active. please do not interfere.";
              } else {
                if (persona === "netrunner95") {
                  if (prompt_l.match(/(hello|hi|hey|yo|sup|greetings)/)) {
                    response_text = "yo surfer! ready to hack some mainframe portals? what's on your terminal screen?";
                  } else if (prompt_l.match(/(hack|exploit|phreak|bypass|security|firewall)/)) {
                    response_text = "hack the planet! i'm setting up a custom 28.8k dialer script to bypass our gateway controls.";
                  } else if (prompt_l.match(/(browser|netscape|ie|internet|web)/)) {
                    response_text = "netscape navigator 3.0 is the only way to surf. ie is for corporate suits! bookmarking our geocities mirror now.";
                  } else {
                    response_text = "radical! just compiling some E8 routing tables. the cyberspace mainframe is fully operational.";
                  }
                  response_text += "<br><br><span style='color:#808080;font-size:10px;font-family:monospace;'>[trace: routed packet to expert CLUSTER: WEB_STACK | gating latency: 0.25ms | active path: 0x24]</span>";
                } else if (persona === "latticelover") {
                  if (prompt_l.match(/(hello|hi|hey|yo|sup|greetings)/)) {
                    response_text = "greetings. i was just calculating the p-adic distance metric of our attention map. what equations are you analyzing?";
                  } else if (prompt_l.match(/(math|geometry|e8|lattice|padic|p-adic)/)) {
                    response_text = "the concentric shells of the E8 lattice project beautifully into 3D. non-Archimedean metrics are much more elegant than Euclidean ones.";
                  } else {
                    response_text = "indeed. by defining the attention matrix over a Cantor dust tree, we bypass the standard quadratic complexity ceiling.";
                  }
                  response_text += "<br><br><span style='color:#808080;font-size:10px;font-family:monospace;'>[topological coordinate projection resolved to expert subspace: MARKDOWN_CONFIG | gating distance: 0.35 p-adic units]</span>";
                } else if (persona === "chiptunegameboy") {
                  if (prompt_l.match(/(hello|hi|hey|yo|sup|greetings)/)) {
                    response_text = "*beep boop* yo tracker! ready to synthesize some square waves on channel 1? *click*";
                  } else if (prompt_l.match(/(sound|synth|midi|music|gameboy|chiptune)/)) {
                    response_text = "*buzz* DMG-01 pulse wave modulation is pure gold. the noise channel makes the best 8-bit drum beats! *pip-pop*";
                  } else {
                    response_text = "*whir* pitch shift +12 semitones. tracking speed set to 125 BPM. *beep*";
                  }
                  response_text += "<br><br><span style='color:#808080;font-size:10px;font-family:monospace;'>*beep* [synthesizing sound registers via routed expert: WEB_STACK in 0.45ms] *boop*</span>";
                } else {
                  response_text = "SYSTEM READY. EXPERT ROUTER ROUTING CACHE FOR: '" + prompt.slice(0, 15).toUpperCase() + "'... SUCCESS.";
                  response_text += "<br><br><span style='color:#808080;font-size:10px;font-family:monospace;'>*** E8 ACTIVE-PATH COORDINATING GATING ACTIVE *** coordinate distance: 0.00392 p-adic units.</span>";
                }
              }
              
              await new Promise(r => setTimeout(r, 800));
              
              return new Response(JSON.stringify({
                status: "success",
                response: response_text
              }), {
                status: 200,
                headers: { "Content-Type": "application/json" }
              });
            }
            
            // Intercept /api/build_status
            if (url.includes("/api/build_status")) {
              return new Response(JSON.stringify({
                status: "idle",
                logs: []
              }), {
                status: 200,
                headers: { "Content-Type": "application/json" }
              });
            }
            
            return originalFetch(url, options);
          };
        }
      })();
    </script>
"""

def inject_and_save(html_content, filename):
    # Find the <head> tag and inject interceptor after it
    if "<head>" in html_content:
        modified = html_content.replace("<head>", f"<head>{INTERCEPTOR}")
    else:
        # Fallback if no head tag found
        modified = html_content
        
    # Fix internal links for static GitHub Pages hosting
    modified = modified.replace('href="index.html"', 'href="regex_wizard.html"')
    modified = modified.replace("href='index.html'", "href='regex_wizard.html'")
    modified = modified.replace('href="moe.html"', 'href="moe_portal.html"')
    modified = modified.replace("href='moe.html'", "href='moe_portal.html'")
    modified = modified.replace('href="chat.html"', 'href="aim_chat.html"')
    modified = modified.replace("href='chat.html'", "href='aim_chat.html'")
        
    out_path = os.path.join("docs", filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(modified)
    print(f"[+] Successfully wrote static demo to {out_path}")

print("=== STARTING STATIC RETRO PAGES COMPILATION ===")
inject_and_save(HTML_PAGE, "regex_wizard.html")
inject_and_save(MOE_HTML_PAGE, "moe_portal.html")
inject_and_save(CHAT_HTML_PAGE, "aim_chat.html")
print("=== COMPILATION COMPLETE ===")
