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

fn find_isolated_anomalies(
    num_nodes: usize,
    edges: &[(usize, usize)],
    island_nodes: &[usize],
    root_idx: usize,
) -> Vec<usize> {
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
    // Anomalies are island nodes that are completely unreachable from the root node
    island_nodes
        .iter()
        .copied()
        .filter(|i| !visited.contains(i))
        .collect()
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

fn audit_json_graph(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let payload: GraphPayload = serde_json::from_reader(reader)?;

    let num_nodes = payload.nodes.len();
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

    let system_boundary_len = if num_nodes > 0 { num_nodes - 1 } else { 0 };
    let resolution = pruner.prune(&topology, system_boundary_len)?;

    // Identify isolated anomalies by filtering out reachable nodes from root (index 0)
    let isolated_anomalies = find_isolated_anomalies(num_nodes, &payload.edges, &resolution.island_nodes, 0);
    let is_fatal = resolution.action == PolicyAction::FatalBlock && !isolated_anomalies.is_empty();

    println!("\n[Audit Results]");
    println!("--------------------------------------------------");
    println!("Security Action Verdict  : {}", if is_fatal { "FATAL_BLOCK" } else { "ALLOW" });
    println!("Algebraic Conn Score (\u{03BB}\u{2082}): {:.6}", resolution.connectivity_score);
    println!("Secured Mainland Nodes  : {:?}", resolution.mainland_nodes.iter().map(|&i| &payload.nodes[i]).collect::<Vec<_>>());
    println!("Quarantined Anomaly Set  : {:?}", isolated_anomalies.iter().map(|&i| &payload.nodes[i]).collect::<Vec<_>>());
    println!("--------------------------------------------------");

    if is_fatal {
        eprintln!(
            "[CRITICAL] \u{1F6AB} ALERT: Topologically isolated malicious nodes detected: {:?}",
            isolated_anomalies.iter().map(|&i| &payload.nodes[i]).collect::<Vec<_>>()
        );
        std::process::exit(1);
    } else {
        println!("[NOMINAL] \u{2705} Dependency graph verified clean or structurally unified. Safe to proceed.");
    }

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

    let all_packages: Vec<String> = normal_packages.iter().cloned().chain(sink_packages.iter().cloned()).collect();
    let package_to_idx: HashMap<String, usize> = all_packages.iter().cloned().enumerate().map(|(i, name)| (name, i)).collect();

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

    // Identify isolated anomalies by filtering out reachable nodes from root (index 0)
    let isolated_anomalies = find_isolated_anomalies(num_nodes, &edges, &resolution.island_nodes, 0);
    let is_fatal = resolution.action == PolicyAction::FatalBlock && !isolated_anomalies.is_empty();

    println!("\n[Audit Results]");
    println!("--------------------------------------------------");
    println!("Security Action Verdict  : {}", if is_fatal { "FATAL_BLOCK" } else { "ALLOW" });
    println!("Algebraic Conn Score (\u{03BB}\u{2082}): {:.6}", resolution.connectivity_score);
    println!("Secured Mainland Nodes  : {:?}", resolution.mainland_nodes.iter().map(|&i| if all_packages[i].is_empty() { "root" } else { &all_packages[i] }).collect::<Vec<_>>());
    println!("Quarantined Anomaly Set  : {:?}", isolated_anomalies.iter().map(|&i| &all_packages[i]).collect::<Vec<_>>());
    println!("--------------------------------------------------");

    if is_fatal {
        eprintln!(
            "[CRITICAL] \u{1F6AB} ALERT: Topologically isolated malicious nodes detected: {:?}",
            isolated_anomalies.iter().map(|&i| &all_packages[i]).collect::<Vec<_>>()
        );
        std::process::exit(1);
    } else {
        println!("[NOMINAL] \u{2705} Dependency graph verified clean or structurally unified. Safe to proceed.");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nominal_connected_graph_passes() {
        // A simple connected graph: 0 -> 1 -> 2, with 3 as a sink.
        // 0 (root) depends on 1, 1 depends on 2, 1 also uses sink 3.
        let num_nodes = 4;
        let edges = vec![(0, 1), (1, 2), (1, 3)];
        
        let mut topology = Topology::new(num_nodes);
        for &(u, v) in &edges {
            topology.add_edge(u, v);
        }
        topology.add_sink(3);

        let pruner = TauSpectralPruner::builder()
            .tau(0.0)
            .threat_threshold(1.5)
            .momentum_beta(0.5)
            .system_start_idx(3)
            .build();

        let resolution = pruner.prune(&topology, 3).unwrap();
        let isolated = find_isolated_anomalies(num_nodes, &edges, &resolution.island_nodes, 0);
        
        // Even if the pruner returns FatalBlock, our reachability filter should identify no isolated anomalies.
        assert!(isolated.is_empty(), "Nominal connected graph should have no isolated anomalies");
    }

    #[test]
    fn test_malicious_isolated_node_fails() {
        // A graph where 0 -> 1 -> 2.
        // Node 3 is an isolated package in the lockfile that is NOT reachable from 0.
        // Node 3 attempts to connect to sink 4.
        let num_nodes = 5;
        let edges = vec![(0, 1), (1, 2), (3, 4)];
        
        let mut topology = Topology::new(num_nodes);
        for &(u, v) in &edges {
            topology.add_edge(u, v);
        }
        topology.add_sink(4);

        let pruner = TauSpectralPruner::builder()
            .tau(0.0)
            .threat_threshold(1.5)
            .momentum_beta(0.5)
            .system_start_idx(4)
            .build();

        let resolution = pruner.prune(&topology, 4).unwrap();
        let isolated = find_isolated_anomalies(num_nodes, &edges, &resolution.island_nodes, 0);

        // Resolution should result in FatalBlock
        assert_eq!(resolution.action, PolicyAction::FatalBlock);
        // And the isolated anomaly set should contain Node 3
        assert_eq!(isolated, vec![3]);
    }
}

