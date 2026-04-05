import os
import shutil
import subprocess

ROOT = r"e:\LLM\claude-code-main\codeclaw\codeclaw"

mapping = {
    # memory
    "core/memory.py": "memory/memory.py",
    "core/extract_memories.py": "memory/extract_memories.py",
    "core/session_memory.py": "memory/session_memory.py",
    
    # context
    "core/context.py": "context/context.py",
    "core/attachments.py": "context/attachments.py",
    "core/memory_files.py": "context/memory_files.py",
    "core/file_state_cache.py": "context/file_state_cache.py",
    "core/tool_results.py": "context/tool_results.py",
    
    # security
    "core/permissions.py": "security/permissions.py",
    "core/security_classifier.py": "security/security_classifier.py",
    "core/sandbox.py": "security/sandbox.py",
    
    # features
    "core/plans.py": "features/plans.py",
    "core/ultraplan.py": "features/ultraplan.py",
    "core/todos.py": "features/todos.py",
    "core/auto_commit.py": "features/auto_commit.py",
    "core/verification.py": "features/verification.py",
    "core/structured_output.py": "features/structured_output.py",
    "core/content_replacement.py": "features/content_replacement.py",
    "core/incremental_write_queue.py": "features/incremental_write_queue.py",
    "core/artifact_tracker.py": "features/artifact_tracker.py",
    "core/browser.py": "features/browser.py",
    
    # agents
    "core/team.py": "agents/team.py",
    "tools/builtin_agents.py": "agents/builtin_agents.py",
    "tools/agent_tool.py": "agents/agent_tool.py",
    
    # skills
    "tools/skill_tool.py": "skills/skill_tool.py",
    
    # utils
    "core/shell_tasks.py": "utils/shell_tasks.py",
    "core/vcr.py": "utils/vcr.py",
    "core/frc.py": "utils/frc.py",
}

dirs_to_create = ["memory", "context", "security", "features", "agents", "skills", "utils"]
for d in dirs_to_create:
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    init_path = os.path.join(ROOT, d, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w", encoding="utf-8") as f:
            f.write("")

def run_git_mv(src, dst):
    src_abs = os.path.join(ROOT, src)
    dst_abs = os.path.join(ROOT, dst)
    if os.path.exists(src_abs):
        try:
            subprocess.run(["git", "mv", src, dst], cwd=ROOT, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            print(f"Moved {src} -> {dst}")
        except subprocess.CalledProcessError as e:
            print(f"Git mv failed for {src}: {e.stderr.decode('utf-8').strip()}, using shutil.move")
            shutil.move(src_abs, dst_abs)

for src, dst in mapping.items():
    run_git_mv(src, dst)

import_replacements = []
for src, dst in mapping.items():
    src_mod = src.replace("/", ".").replace(".py", "")
    dst_mod = dst.replace("/", ".").replace(".py", "")
    import_replacements.append((f"codeclaw.{src_mod}", f"codeclaw.{dst_mod}"))

all_py_files = []
for root_dir, _, files in os.walk(ROOT):
    for str_file in files:
        if str_file.endswith(".py"):
            all_py_files.append(os.path.join(root_dir, str_file))

for py_file in all_py_files:
    try:
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        continue
        
    original = content
    for old_imp, new_imp in import_replacements:
        content = content.replace(old_imp, new_imp)
        
    if original != content:
        with open(py_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated imports in {os.path.relpath(py_file, ROOT)}")

print("Done refactoring.")
