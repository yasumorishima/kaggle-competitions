# --- idle-compact: 50 分放置で 1 回だけ自動コンパクト（全セッション共通） ---
IC=/opt/idle-compact
if [ ! -d "$IC/plugins/idle-compact" ]; then
  git clone -q https://github.com/takahirom/takahirom-claude-code-marketplace "$IC" \
    && git -C "$IC" checkout -q a8d452ebc0e95613237e162ce4ccba268cc9da70 || true
fi
mkdir -p ~/.claude
python3 - <<'PY'
import json, os
p = os.path.expanduser("~/.claude/settings.json")
s = json.load(open(p)) if os.path.exists(p) else {}
env = s.setdefault("env", {})
env["CLAUDE_CODE_ENABLE_FUNCTION_HOOKS"] = "1"
env["CLAUDE_CODE_PLUGIN_DIRS"] = "/opt/idle-compact/plugins/idle-compact"
json.dump(s, open(p, "w"), indent=2)
PY
# --- ここまで ---
