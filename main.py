import sys
import os
import json
import time

# System Path Adjustment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from brain_engine import CogniPro
    from registry.config import SystemConfig
except ImportError as e:
    print(f"System Linkage Error: {e}")
    sys.exit(1)

def run_repl():
    platform = CogniPro()
    platform.train_knowledge(["System startup sequences running natively"])
    
    # v5.5 Startup
    platform.display_neural_report()
    print("  Commands: /ingest <file>, /web_learn <url>, /upgrade, /status, /reset, /rebuild, /help, /exit")
    print("  Type anything to chat.\n")

    while True:
        try:
            user_input = input("You > ").strip()
            if not user_input: continue
            
            if user_input.startswith("/"):
                parts = user_input.split()
                cmd = parts[0].lower()
                
                if cmd == "/exit": break
                elif cmd == "/reset":
                    confirm = input("  [CAUTION] Resetting manifold centroids. Weight essence will remain. Proceed? (y/n): ").strip().lower()
                    if confirm == 'y':
                        platform.router.reset_centroids()
                        print("  [System] Expert centroids reset. Epistemic weights preserved in Hebbian mode.")
                    else:
                        print("  Reset aborted.")
                elif cmd == "/rebuild":
                    if len(parts) > 1:
                        target = parts[1]
                        print(f"  [System] Initiating cumulative distillation from {target}...")
                        platform.ingestion_engine.distill_json(target)
                    else:
                        print("  Usage: /rebuild <json>")
                elif cmd == "/web_learn":
                    if len(parts) > 1:
                        url = parts[1]
                        report = platform.ingestion_engine.scrape_url(url)
                        print(f"\n  [Cogni-Sense] {report}\n")
                    else:
                        print("  Usage: /web_learn <url>")
                elif cmd == "/status":
                    platform.display_neural_report()
                elif cmd == "/clean":
                    print("  [System] Cleaning redundant synapses and optimizing manifold...")
                    # Remove anchors with very low activation weight or redundant content
                    count = 0
                    keys = list(platform.long_term.knowledge_base.keys())
                    for k in keys:
                        v = platform.long_term.knowledge_base[k]
                        if v.get("activation_weight", 1.0) < 0.1:
                            del platform.long_term.knowledge_base[k]
                            count += 1
                    print(f"  [System] Cleaned {count} weak synapses.")
                    import gc
                    gc.collect()
                elif cmd == "/upgrade":
                    print("\n  [/upgrade] Cogni Pro Brain Upgrade Protocol — v5.5")
                    print("  " + "="*45)
                    
                    # Step 1: Expand anchors toward 1024 saturation
                    print("  [Step 1/4] Autonomous Anchor Expansion...")
                    platform._promote_tokens_to_anchors()
                    
                    # Step 2: Re-balance expert shards
                    print("  [Step 2/4] Expert Shard Re-balancing (Coding → Computing)...")
                    platform.rebalance_shards("Coding", "Computing")
                    platform.rebalance_shards("Science", "Computing")
                    
                    # Step 3: Prune noise tokens
                    print("  [Step 3/4] Dictionary Pruning (noise token removal)...")
                    platform.prune_dictionary()
                    
                    # Step 4: Hebbian consistency audit
                    print("  [Step 4/4] Hebbian Manifold Consistency Audit...")
                    platform.ingestion_engine.verify_manifold_consistency()
                    
                    print("  [/upgrade COMPLETE] Brain upgrade successful!")
                    platform.display_neural_report()
                elif cmd == "/help":
                    print("\n  === Cogni Pro v5.5 — Command Reference ===")
                    print("  /ingest <file>    — Load a local JSON knowledge file")
                    print("  /web_learn <url>  — Non-destructive web ingestion (v5.5 Delta-Weight)")
                    print("  /upgrade          — Full brain upgrade: expand anchors, rebalance, prune")
                    print("  /status           — Live Neural Report (anchors, health, vocab)")
                    print("  /reset            — Reset expert centroids (weights preserved)")
                    print("  /rebuild <json>   — Rebuild knowledge from a JSON source")
                    print("  /exit             — Exit Cogni Pro\n")
                continue

            # Wrap user input with a hidden instruction frame to prime the generative manifold
            primed_input = f"[System: Expert Mode] [Input: {user_input}] [Output: Start Generating Text...]\n{user_input}"
            response, conf = platform.process(primed_input)
            print(f"\n{response}\n")
        except KeyboardInterrupt: break
        except Exception as e: print(f"Error: {e}")

if __name__ == "__main__":
    run_repl()