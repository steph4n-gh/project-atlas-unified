#!/usr/bin/env python
"""Generate training datasets for the DevOps SysAdmin Micro-Expert Toolbox.

Generates diverse, syntactically valid datasets for:
1. SQL Expert (Text to SQL queries for Postgres, MySQL, SQLite)
2. Cron Expert (Natural language to Crontab expressions)
3. Git & CLI Expert (Natural language to git operations and shell pipelines)
4. YAML Config Expert (Natural language to Kubernetes/Docker configs)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

# --- 1. CRON GENERATOR ---
CRON_TIMES = [
    ("every minute", "* * * * *"),
    ("every hour", "0 * * * *"),
    ("every day at midnight", "0 0 * * *"),
    ("every Sunday at midnight", "0 0 * * 0"),
    ("every weekday at 9 AM", "0 9 * * 1-5"),
    ("every 15 minutes", "*/15 * * * *"),
    ("every 2 hours", "0 */2 * * *"),
    ("on the 1st of every month at midnight", "0 0 1 * *"),
    ("at 5 PM on Fridays", "0 17 * * 5"),
    ("every weekday at midnight", "0 0 * * 1-5")
]

CRON_ACTIONS = [
    ("run backup script", "/scripts/backup.sh"),
    ("clean up temp files", "rm -rf /tmp/*"),
    ("sync database replica", "/usr/bin/pg_dumpall"),
    ("rotate logs", "logrotate -f /etc/logrotate.conf"),
    ("check system health", "/usr/local/bin/healthcheck.sh"),
    ("update security patches", "apt-get update && apt-get upgrade -y"),
    ("restart web service", "systemctl restart nginx"),
    ("clear redis cache", "redis-cli flushall"),
    ("send daily reports", "/app/bin/send_reports.py"),
    ("vacuum postgres database", "vacuumdb -a -z")
]

def generate_cron_dataset(size: int) -> List[Dict[str, Any]]:
    samples = []
    for i in range(size):
        time_desc, time_cron = random.choice(CRON_TIMES)
        act_desc, act_cmd = random.choice(CRON_ACTIONS)
        
        # Add slight NL phrasing variations
        phrases = [
            f"{act_desc} {time_desc}",
            f"Schedule a cron job to {act_desc.lower()} {time_desc}",
            f"{time_desc.capitalize()}, {act_desc.lower()}"
        ]
        inst = random.choice(phrases)
        pattern = f"{time_cron} {act_cmd}"
        
        samples.append({
            "id": f"cron_{i:05d}",
            "instruction": inst,
            "dialect": "[CRON]",
            "pattern": pattern,
            "ast_breakdown": {
                "D2_steps": 1.0 if "*/" in time_cron else 0.0,
                "D1_ranges": 1.0 if "-" in time_cron else 0.0,
                "D3_weekdays": 1.0 if time_cron.split()[-1] != "*" else 0.0
            }
        })
    return samples

# --- 2. SQL GENERATOR ---
SQL_COLUMNS = ["*", "id, name", "email, status", "COUNT(*)", "SUM(amount)", "AVG(price)"]
SQL_TABLES = [
    ("users", "u"),
    ("orders", "o"),
    ("products", "p"),
    ("posts", "post"),
    ("comments", "c")
]
SQL_CONDITIONS = [
    ("status = 'active'", "active status"),
    ("created_at >= NOW() - INTERVAL '30 days'", "created in the last 30 days"),
    ("amount > 100", "amount greater than 100"),
    ("price <= 50", "price less than or equal to 50"),
    ("email LIKE '%@gmail.com'", "gmail users")
]

def generate_sql_dataset(size: int) -> List[Dict[str, Any]]:
    samples = []
    dialects = ["[POSTGRES]", "[MYSQL]", "[SQLITE]"]
    for i in range(size):
        dialect = random.choice(dialects)
        cols = random.choice(SQL_COLUMNS)
        table, alias = random.choice(SQL_TABLES)
        cond, cond_desc = random.choice(SQL_CONDITIONS)
        
        # Convert Postgres interval to MySQL/SQLite syntax if necessary
        if dialect == "[MYSQL]" and "NOW() - INTERVAL" in cond:
            cond = cond.replace("NOW() - INTERVAL '30 days'", "DATE_SUB(NOW(), INTERVAL 30 DAY)")
        elif dialect == "[SQLITE]" and "NOW() - INTERVAL" in cond:
            cond = cond.replace("NOW() - INTERVAL '30 days'", "datetime('now', '-30 days')")

        inst = f"get {cols} from {table} table where {cond_desc}"
        pattern = f"SELECT {cols} FROM {table} {alias} WHERE {cond};"
        
        samples.append({
            "id": f"sql_{i:05d}",
            "instruction": inst,
            "dialect": dialect,
            "pattern": pattern,
            "ast_breakdown": {
                "D0_depth": 0.0,
                "D1_joins": 0.0,
                "D6_aggregates": 1.0 if "COUNT" in cols or "SUM" in cols or "AVG" in cols else 0.0
            }
        })
    return samples

# --- 3. GIT/CLI GENERATOR ---
GIT_COMMANDS = [
    ("discard all uncommitted local changes", "git reset --hard HEAD"),
    ("undo the last commit but keep files", "git reset --soft HEAD~1"),
    ("delete a local branch named feature", "git branch -d feature"),
    ("force push current branch to origin", "git push origin HEAD --force"),
    ("cherry pick commit abc1234", "git cherry-pick abc1234"),
    ("staged all modified files for commit", "git add -u"),
    ("checkout a new branch named test", "git checkout -b test"),
    ("stash all local uncommitted changes", "git stash save"),
    ("pull origin main with rebase", "git pull --rebase origin main"),
    ("show commit logs simplified to one line", "git log --oneline")
]

CLI_COMMANDS = [
    ("find all files larger than 100M", "find . -type f -size +100M"),
    ("find text 'error' in log files", "grep -rn 'error' *.log"),
    ("delete all files ending in .tmp", "find . -name '*.tmp' -delete"),
    ("count total lines in all python files", "find . -name '*.py' | xargs wc -l"),
    ("replace text 'foo' with 'bar' in file", "sed -i 's/foo/bar/g' file.txt"),
    ("display memory usage in human-readable format", "free -h"),
    ("monitor system processes sorted by CPU", "top -o %CPU"),
    ("find unique lines in access log", "sort access.log | uniq"),
    ("print disk usage per directory", "du -sh *"),
    ("kill process running on port 8080", "lsof -t -i:8080 | xargs kill -9")
]

def generate_git_cli_dataset(size: int) -> List[Dict[str, Any]]:
    samples = []
    for i in range(size):
        is_git = random.choice([True, False])
        desc, cmd = random.choice(GIT_COMMANDS if is_git else CLI_COMMANDS)
        
        phrases = [
            f"{desc}",
            f"Run command to {desc.lower()}",
            f"Shell script to {desc.lower()}"
        ]
        inst = random.choice(phrases)
        dialect = "[GIT]" if is_git else "[CLI]"
        
        samples.append({
            "id": f"git_{i:05d}",
            "instruction": inst,
            "dialect": dialect,
            "pattern": cmd,
            "ast_breakdown": {
                "D1_pipe_count": float(cmd.count("|")),
                "D7_danger_level": 2.0 if "reset --hard" in cmd or "delete" in cmd or "kill" in cmd or "rm" in cmd else 1.0
            }
        })
    return samples

# --- 4. YAML GENERATOR ---
YAML_TEMPLATES = [
    (
        "kubernetes service exposing port 80 targeting app: web",
        "apiVersion: v1\nkind: Service\nmetadata:\n  name: web-service\nspec:\n  ports:\n  - port: 80\n    targetPort: 80\n  selector:\n    app: web"
    ),
    (
        "kubernetes pod running alpine container",
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: alpine-pod\nspec:\n  containers:\n  - name: alpine\n    image: alpine:latest\n    command: [\"sleep\", \"3600\"]"
    ),
    (
        "docker compose service running redis and mapping port 6379",
        "version: '3.8'\nservices:\n  redis:\n    image: redis:alpine\n    ports:\n      - \"6379:6379\"\n    restart: always"
    ),
    (
        "github actions job to run test commands",
        "name: Node CI\non: [push]\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n    - uses: actions/checkout@v3\n    - name: Run tests\n      run: npm test"
    )
]

def generate_yaml_dataset(size: int) -> List[Dict[str, Any]]:
    samples = []
    for i in range(size):
        desc, yaml_str = random.choice(YAML_TEMPLATES)
        
        # Add dialect tags based on YAML content
        dialect = "[K8S]"
        if "docker" in desc:
            dialect = "[DOCKER]"
        elif "github" in desc:
            dialect = "[GHA]"
            
        phrases = [
            f"Generate {desc}",
            f"{dialect} configuration for {desc.lower()}",
            f"YAML manifest to {desc.lower()}"
        ]
        inst = random.choice(phrases)
        
        samples.append({
            "id": f"yaml_{i:05d}",
            "instruction": inst,
            "dialect": dialect,
            "pattern": yaml_str,
            "ast_breakdown": {
                "D0_indent_levels": float(yaml_str.count("\n  ")),
                "D6_selectors": 1.0 if "selector" in yaml_str else 0.0
            }
        })
    return samples

# --- MAIN ---
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate datasets for all DevOps SysAdmin experts.")
    parser.add_argument("--out-dir", type=str, default="/Volumes/Storage/project_atlas/scratch/devops_data", help="Output directory for generated datasets.")
    parser.add_argument("--size-cron", type=int, default=1000, help="Size of the Cron dataset.")
    parser.add_argument("--size-sql", type=int, default=2000, help="Size of the SQL dataset.")
    parser.add_argument("--size-git", type=int, default=1500, help="Size of the Git/CLI dataset.")
    parser.add_argument("--size-yaml", type=int, default=1000, help="Size of the YAML dataset.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    args = parser.parse_args(argv)

    random.seed(args.seed)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Generating DevOps SysAdmin Expert Datasets ===")
    
    # 1. Cron
    cron_file = out_dir / "cron_dataset.jsonl"
    print(f"Generating Cron dataset ({args.size_cron} samples)...")
    cron_data = generate_cron_dataset(args.size_cron)
    with open(cron_file, "w") as f:
        for s in cron_data:
            f.write(json.dumps(s) + "\n")
    print(f"  Saved to {cron_file}")

    # 2. SQL
    sql_file = out_dir / "sql_dataset.jsonl"
    print(f"Generating SQL dataset ({args.size_sql} samples)...")
    sql_data = generate_sql_dataset(args.size_sql)
    with open(sql_file, "w") as f:
        for s in sql_data:
            f.write(json.dumps(s) + "\n")
    print(f"  Saved to {sql_file}")

    # 3. Git/CLI
    git_file = out_dir / "git_cli_dataset.jsonl"
    print(f"Generating Git/CLI dataset ({args.size_git} samples)...")
    git_data = generate_git_cli_dataset(args.size_git)
    with open(git_file, "w") as f:
        for s in git_data:
            f.write(json.dumps(s) + "\n")
    print(f"  Saved to {git_file}")

    # 4. YAML
    yaml_file = out_dir / "yaml_dataset.jsonl"
    print(f"Generating YAML dataset ({args.size_yaml} samples)...")
    yaml_data = generate_yaml_dataset(args.size_yaml)
    with open(yaml_file, "w") as f:
        for s in yaml_data:
            f.write(json.dumps(s) + "\n")
    print(f"  Saved to {yaml_file}")

    print("\nDataset generation completed successfully!")
    print(f"All files saved in: {out_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
