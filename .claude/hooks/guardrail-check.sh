#!/usr/bin/env bash
# PreToolUse guardrail for Claude Code. Reads the hook JSON on stdin.
# Exit 2 = hard block (reason on stderr). Exit 0 = allow.
set -u

payload=$(cat)

# Parse with python3 (jq is not installed on this machine). Prints: tool_name<TAB>value
parsed=$(printf '%s' "$payload" | python3 -I -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("PARSE_ERROR\t"); sys.exit(0)
t = d.get("tool_name", "")
i = d.get("tool_input", {}) or {}
v = i.get("command", "") if t == "Bash" else i.get("file_path", "")
print(t + "\t" + v.replace("\n", " "))
')

tool=${parsed%%$'\t'*}
value=${parsed#*$'\t'}

block() {
    echo "Blocked by .claude/hooks/guardrail-check.sh: $1" >&2
    exit 2
}

if [ "$tool" = "PARSE_ERROR" ]; then
    block "could not parse hook input"
fi

case "$tool" in
Bash)
    # Recursive force delete in any form: rm -rf, rm -fr, rm -r -f, /bin/rm -rf, --recursive --force
    if printf '%s' "$value" | grep -Eq '(^|[^A-Za-z0-9_-])(/usr)?(/bin/)?rm[[:space:]]+(.*[[:space:]])?(-[A-Za-z]*r[A-Za-z]*f|-[A-Za-z]*f[A-Za-z]*r|--recursive.*--force|--force.*--recursive|-r[[:space:]]+-f|-f[[:space:]]+-r)'; then
        block "recursive force delete (rm -rf / rm -fr)"
    fi
    if printf '%s' "$value" | grep -Eq 'chmod[[:space:]]+(-[A-Za-z]+[[:space:]]+)*0?777'; then
        block "chmod 777"
    fi
    if printf '%s' "$value" | grep -Eq 'git[[:space:]]+push.*([[:space:]]--force|[[:space:]]-f([[:space:]]|$)|[[:space:]]\+[A-Za-z0-9_/.-]+)'; then
        block "force push"
    fi
    # Reading or exfiltrating secrets: .env files, SSH keys, AWS credentials
    if printf '%s' "$value" | grep -Eq '(^|[^A-Za-z0-9_])\.env([^A-Za-z0-9_]|$)|\.ssh/|\.aws/|id_rsa|id_ed25519'; then
        if printf '%s' "$value" | grep -Eq '(^|[^A-Za-z0-9_])(cat|less|more|head|tail|echo|printf|base64|xxd|od|strings|grep|cp|scp|rsync|tar|zip|curl|wget|nc|source|\.)([[:space:]]|$)'; then
            block "access to .env / SSH / AWS credential files"
        fi
    fi
    ;;
Edit | Write)
    base=$(basename -- "$value")
    case "$base" in
    .env*) block "editing $base (secrets file)" ;;
    esac
    case "$value" in
    */.claude/settings.json | .claude/settings.json) block "editing .claude/settings.json (guardrail config)" ;;
    */.claude/hooks/* | .claude/hooks/*) block "editing .claude/hooks/ (guardrail scripts)" ;;
    esac
    ;;
esac

exit 0
