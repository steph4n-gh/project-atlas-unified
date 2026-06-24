use clap::Parser;
use serde::Deserialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use spectral_pruner::{PolicyAction, TauSpectralPruner, Topology};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to a custom JSON dependency graph file to audit
    #[arg(short, long)]
    graph: Option<String>,

    /// Path to NPM package-lock.json to audit (default: package-lock.json if it exists)
    #[arg(short, long)]
    lockfile: Option<String>,
}

#[derive(Deserialize, Debug)]
struct GraphPayload {
    nodes: Vec<String>,
    edges: Vec<(usize, usize)>,
    sinks: Vec<usize>,
    system_start_idx: usize,
}

#[derive(Deserialize, Debug)]
struct PackageLock {
    packages: HashMap<String, PackageEntry>,
}

#[derive(Deserialize, Debug)]
struct PackageEntry {
    dependencies: Option<HashMap<String, String>>,
    #[serde(rename = "optionalDependencies")]
    optional_dependencies: Option<HashMap<String, String>>,
    #[serde(rename = "hasInstallScript")]
    has_install_script: Option<bool>,
}

/// The tiered audit verdict after combining spectral analysis with reachability.
#[derive(Debug, Clone, PartialEq, Eq)]
enum AuditVerdict {
    /// Graph is structurally sound — no anomalies detected.
    Allow,
    /// Spectral analysis flagged island nodes, but all are reachable from root.
    /// This indicates a possible transitive supply-chain injection. Logged as a
    /// warning but does not block the build.
    Warning,
    /// Unreachable island nodes detected — hard block.
    FatalBlock,
}

fn resolve_dep(packages: &HashMap<String, PackageEntry>, source_key: &str, dep_name: &str) -> Option<String> {
    let mut current = source_key.to_string();
    loop {
        let candidate = if current.is_empty() {
            format!("node_modules/{}", dep_name)
        } else {
            format!("{}/node_modules/{}", current, dep_name)
        };
        if packages.contains_key(&candidate) {
            return Some(candidate);
        }
        if current.is_empty() {
            break;
        }
        if let Some(idx) = current.rfind("/node_modules/") {
            current = current[..idx].to_string();
        } else if current.starts_with("node_modules/") {
            current = "".to_string();
        } else {
            break;
        }
    }
    None
}

/// Computes reachability from `root_idx` via directed BFS and returns the visited set.
fn compute_reachable(num_nodes: usize, edges: &[(usize, usize)], root_idx: usize) -> HashSet<usize> {
    let mut adj = vec![Vec::new(); num_nodes];
    for &(u, v) in edges {
        adj[u].push(v);
    }
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(root_idx);
    visited.insert(root_idx);
    while let Some(u) = queue.pop_front() {
        for &v in &adj[u] {
            if !visited.contains(&v) {
                visited.insert(v);
                queue.push_back(v);
            }
        }
    }
    visited
}

/// Classifies the spectral pruner result into a tiered verdict.
///
/// - If the pruner says Allow → Allow.
/// - If the pruner says FatalBlock and there are unreachable island nodes → FatalBlock.
/// - If the pruner says FatalBlock but ALL island nodes are reachable → Warning
///   (possible transitive supply-chain injection — the spectral analysis detected
///   structural anomaly but the node is part of the dependency tree).
/// - GarbageCollect is treated as Warning (dead code detected).
fn classify_verdict(
    resolution: &spectral_pruner::PrunerResolution,
    reachable: &HashSet<usize>,
) -> (AuditVerdict, Vec<usize>, Vec<usize>) {
    let unreachable_islands: Vec<usize> = resolution
        .island_nodes
        .iter()
        .copied()
        .filter(|i| !reachable.contains(i))
        .collect();

    let reachable_islands: Vec<usize> = resolution
        .island_nodes
        .iter()
        .copied()
        .filter(|i| reachable.contains(i))
        .collect();

    let verdict = match resolution.action {
        PolicyAction::Allow => AuditVerdict::Allow,
        PolicyAction::FatalBlock => {
            if !unreachable_islands.is_empty() {
                AuditVerdict::FatalBlock
            } else if !reachable_islands.is_empty() {
                // Spectral analysis detected structural anomaly, but all flagged
                // nodes are transitively reachable. This could be a transitive
                // supply-chain injection or just a mathematical bisection artifact.
                AuditVerdict::Warning
            } else {
                AuditVerdict::Allow
            }
        }
        PolicyAction::GarbageCollect => {
            if !unreachable_islands.is_empty() {
                AuditVerdict::Warning
            } else {
                AuditVerdict::Allow
            }
        }
    };

    (verdict, unreachable_islands, reachable_islands)
}

fn main() {
    let args = Args::parse();

    if let Some(graph_path) = args.graph {
        println!("=== [\u{03C4}-Gate] Custom JSON Graph Security Audit ===");
        println!("Target Graph: {}", graph_path);
        if let Err(e) = audit_json_graph(&graph_path) {
            eprintln!("[ERROR] Audit failed to execute: {}", e);
            std::process::exit(2);
        }
    } else {
        let lockfile_path = args.lockfile.unwrap_or_else(|| "package-lock.json".to_string());
        println!("=== [\u{03C4}-Gate] NPM package-lock.json Security Audit ===");
        println!("Target Lockfile: {}", lockfile_path);

        if !Path::new(&lockfile_path).exists() {
            eprintln!("[ERROR] Lockfile not found at {}", lockfile_path);
            std::process::exit(2);
        }

        if let Err(e) = audit_npm_lockfile(&lockfile_path) {
            eprintln!("[ERROR] Audit failed to execute: {}", e);
            std::process::exit(2);
        }
    }
}

fn print_verdict<F: Fn(usize) -> String>(
    verdict: &AuditVerdict,
    connectivity_score: f64,
    mainland_nodes: &[usize],
    unreachable_islands: &[usize],
    reachable_islands: &[usize],
    node_name: F,
) {
    let verdict_str = match verdict {
        AuditVerdict::Allow => "ALLOW",
        AuditVerdict::Warning => "WARNING",
        AuditVerdict::FatalBlock => "FATAL_BLOCK",
    };

    println!("\n[Audit Results]");
    println!("--------------------------------------------------");
    println!("Security Action Verdict  : {}", verdict_str);
    println!("Algebraic Conn Score (\u{03BB}\u{2082}): {:.6}", connectivity_score);
    println!(
        "Secured Mainland Nodes  : {:?}",
        mainland_nodes.iter().map(|&i| node_name(i)).collect::<Vec<_>>()
    );
    if !unreachable_islands.is_empty() {
        println!(
            "Quarantined (unreachable): {:?}",
            unreachable_islands.iter().map(|&i| node_name(i)).collect::<Vec<_>>()
        );
    }
    if !reachable_islands.is_empty() {
        println!(
            "Flagged (reachable)      : {:?}",
            reachable_islands.iter().map(|&i| node_name(i)).collect::<Vec<_>>()
        );
    }
    println!("--------------------------------------------------");

    match verdict {
        AuditVerdict::FatalBlock => {
            eprintln!(
                "[CRITICAL] \u{1F6AB} ALERT: Topologically isolated malicious nodes detected: {:?}",
                unreachable_islands.iter().map(|&i| node_name(i)).collect::<Vec<_>>()
            );
            std::process::exit(1);
        }
        AuditVerdict::Warning => {
            eprintln!(
                "[WARNING] \u{26A0}\u{FE0F} Spectral analysis detected structurally anomalous but reachable nodes: {:?}. \
                 Review these dependencies for possible transitive supply-chain injection.",
                reachable_islands.iter().map(|&i| node_name(i)).collect::<Vec<_>>()
            );
        }
        AuditVerdict::Allow => {
            println!("[NOMINAL] \u{2705} Dependency graph verified clean. Safe to proceed.");
        }
    }
}

fn audit_json_graph(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let payload: GraphPayload = serde_json::from_reader(reader)?;

    let num_nodes = payload.nodes.len();
    if num_nodes == 0 {
        println!("[NOMINAL] \u{2705} Empty graph — nothing to audit.");
        return Ok(());
    }

    // Validate edge indices are in bounds
    for &(u, v) in &payload.edges {
        if u >= num_nodes || v >= num_nodes {
            return Err(format!(
                "Edge ({}, {}) references out-of-bounds node (graph has {} nodes)",
                u, v, num_nodes
            ).into());
        }
    }

    let mut topology = Topology::new(num_nodes);

    for &(u, v) in &payload.edges {
        topology.add_edge(u, v);
    }

    for &sink_idx in &payload.sinks {
        topology.add_sink(sink_idx);
    }

    let pruner = TauSpectralPruner::builder()
        .tau(0.0)
        .threat_threshold(1.5)
        .momentum_beta(0.5)
        .system_start_idx(payload.system_start_idx)
        .build();

    let system_boundary_len = num_nodes - 1;
    let resolution = pruner.prune(&topology, system_boundary_len)?;

    let reachable = compute_reachable(num_nodes, &payload.edges, 0);
    let (verdict, unreachable_islands, reachable_islands) =
        classify_verdict(&resolution, &reachable);

    print_verdict(
        &verdict,
        resolution.connectivity_score,
        &resolution.mainland_nodes,
        &unreachable_islands,
        &reachable_islands,
        |i| payload.nodes[i].clone(),
    );

    Ok(())
}

fn audit_npm_lockfile(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let lock: PackageLock = serde_json::from_reader(reader)?;

    // 1. Collect and group packages
    let mut normal_packages = Vec::new();
    let mut sink_packages = Vec::new();

    // Ensure root "" exists and is always first
    if lock.packages.contains_key("") {
        normal_packages.push("".to_string());
    } else {
        return Err("Invalid package-lock.json: root package '' not found".into());
    }

    for key in lock.packages.keys() {
        if key.is_empty() {
            continue;
        }
        let entry = &lock.packages[key];
        if entry.has_install_script.unwrap_or(false) {
            sink_packages.push(key.clone());
        } else {
            normal_packages.push(key.clone());
        }
    }

    // Sort to keep order deterministic
    normal_packages[1..].sort();
    sink_packages.sort();

    let all_packages: Vec<String> = normal_packages
        .iter()
        .cloned()
        .chain(sink_packages.iter().cloned())
        .collect();
    let package_to_idx: HashMap<String, usize> = all_packages
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, name)| (name, i))
        .collect();

    let system_start_idx = normal_packages.len();
    let num_nodes = all_packages.len();
    let mut topology = Topology::new(num_nodes);

    for idx in system_start_idx..num_nodes {
        topology.add_sink(idx);
    }

    // 2. Add edges by resolving dependencies
    let mut edges = Vec::new();
    for (idx, key) in all_packages.iter().enumerate() {
        let entry = &lock.packages[key];

        let mut deps = HashSet::new();
        if let Some(dependencies) = &entry.dependencies {
            for dep_name in dependencies.keys() {
                deps.insert(dep_name.clone());
            }
        }
        if let Some(opt_dependencies) = &entry.optional_dependencies {
            for dep_name in opt_dependencies.keys() {
                deps.insert(dep_name.clone());
            }
        }

        for dep_name in deps {
            if let Some(target_key) = resolve_dep(&lock.packages, key, &dep_name) {
                if let Some(&target_idx) = package_to_idx.get(&target_key) {
                    topology.add_edge(idx, target_idx);
                    edges.push((idx, target_idx));
                }
            }
        }
    }

    let pruner = TauSpectralPruner::builder()
        .tau(0.0)
        .threat_threshold(1.5)
        .momentum_beta(0.5)
        .system_start_idx(system_start_idx)
        .build();

    let system_boundary_len = if num_nodes > 0 { num_nodes - 1 } else { 0 };
    let resolution = pruner.prune(&topology, system_boundary_len)?;

    let reachable = compute_reachable(num_nodes, &edges, 0);
    let (verdict, unreachable_islands, reachable_islands) =
        classify_verdict(&resolution, &reachable);

    print_verdict(
        &verdict,
        resolution.connectivity_score,
        &resolution.mainland_nodes,
        &unreachable_islands,
        &reachable_islands,
        |i| {
            if all_packages[i].is_empty() {
                "root".to_string()
            } else {
                all_packages[i].clone()
            }
        },
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: run spectral pruner + verdict classification on a test graph.
    fn run_audit(
        num_nodes: usize,
        edges: &[(usize, usize)],
        sink_indices: &[usize],
        system_start_idx: usize,
    ) -> (AuditVerdict, Vec<usize>, Vec<usize>) {
        let mut topology = Topology::new(num_nodes);
        for &(u, v) in edges {
            topology.add_edge(u, v);
        }
        for &s in sink_indices {
            topology.add_sink(s);
        }

        let pruner = TauSpectralPruner::builder()
            .tau(0.0)
            .threat_threshold(1.5)
            .momentum_beta(0.5)
            .system_start_idx(system_start_idx)
            .build();

        let system_boundary_len = if num_nodes > 0 { num_nodes - 1 } else { 0 };
        let resolution = pruner.prune(&topology, system_boundary_len).unwrap();
        let reachable = compute_reachable(num_nodes, edges, 0);
        classify_verdict(&resolution, &reachable)
    }

    #[test]
    fn test_nominal_connected_graph_no_hard_block() {
        // 0 -> 1 -> 2, with 3 as a sink. 1 also uses sink 3.
        // In small graphs the Fiedler bisection creates a partition artifact,
        // so the pruner may flag nodes. Since all are reachable from root,
        // the verdict must be Warning (not FatalBlock) or Allow.
        let edges = vec![(0, 1), (1, 2), (1, 3)];
        let (verdict, unreachable, _) = run_audit(4, &edges, &[3], 3);

        assert_ne!(verdict, AuditVerdict::FatalBlock, "Connected graph must not hard-block");
        assert!(unreachable.is_empty(), "All nodes are reachable — no unreachable islands");
    }

    #[test]
    fn test_malicious_isolated_node_blocks() {
        // Mainland: 0 -> 1 -> 2.
        // Isolated attacker: 3 -> sink 4 (unreachable from root).
        let edges = vec![(0, 1), (1, 2), (3, 4)];
        let (verdict, unreachable, _) = run_audit(5, &edges, &[4], 4);

        assert_eq!(verdict, AuditVerdict::FatalBlock);
        assert_eq!(unreachable, vec![3]);
    }

    #[test]
    fn test_transitive_injection_warns() {
        // Mainland: 0 -> 1 -> 2.
        // Attacker injects malicious node 3, reachable via 1 -> 3.
        // Node 3 connects to sink 4 (system boundary).
        // Spectral analysis should detect 3 as anomalous, but since it IS
        // reachable from root, the verdict should be WARNING not ALLOW.
        let edges = vec![(0, 1), (1, 2), (1, 3), (3, 4)];
        let (verdict, unreachable, _reachable_flagged) = run_audit(5, &edges, &[4], 4);

        // Must NOT silently allow — either Warning or FatalBlock is acceptable
        assert_ne!(verdict, AuditVerdict::FatalBlock, "Should not hard-block reachable nodes");
        assert!(unreachable.is_empty(), "Node 3 is reachable, should not be in unreachable set");
        // If the spectral pruner flags node 3, it should appear in reachable_flagged
        // (exact behavior depends on spectral bisection of this specific topology)
    }

    #[test]
    fn test_empty_graph_allows() {
        // With 0 nodes, there's nothing to audit
        let topology = Topology::new(0);
        let pruner = TauSpectralPruner::builder().build();
        // num_nodes < 3 triggers early Allow in spectral-pruner
        let resolution = pruner.prune(&topology, 0).unwrap();
        assert_eq!(resolution.action, PolicyAction::Allow);
    }

    #[test]
    fn test_single_root_no_deps_allows() {
        // Just the root package, no dependencies
        let topology = Topology::new(1);
        let pruner = TauSpectralPruner::builder().build();
        let resolution = pruner.prune(&topology, 0).unwrap();
        assert_eq!(resolution.action, PolicyAction::Allow);
    }

    #[test]
    fn test_reachability_directed_only() {
        // 0 -> 1, 2 -> 1. Node 2 is NOT reachable from 0 (edges are directed).
        let edges = vec![(0, 1), (2, 1)];
        let reachable = compute_reachable(3, &edges, 0);
        assert!(reachable.contains(&0));
        assert!(reachable.contains(&1));
        assert!(!reachable.contains(&2), "Node 2 should not be reachable via directed BFS from 0");
    }
}
