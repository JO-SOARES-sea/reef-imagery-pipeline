#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# set_deepseek_key.sh — store the DeepSeek API key for the "DeepSeek V4 for
# Copilot" VS Code extension from the command line, without ever pasting the
# key into a chat, a shell history line, or a process argument list.
#
# Why a script instead of `export DEEPSEEK_API_KEY=...`?
#   The extension (vizards.deepseek-v4-for-copilot) reads the key from VS Code
#   SecretStorage first, and only falls back to the `deepseek-copilot.apiKey`
#   user setting. It has no CLI flag and reads NO environment variable, so the
#   only command-line-visible place for the key is that setting.
#
#   Caveat: SecretStorage WINS. If a (stale) key is still stored there it
#   shadows the setting, so you must clear it once — see --clear-secret, or
#   run "DeepSeek: Clear API Key" from the Command Palette.
#
# Usage:
#   bash scripts/set_deepseek_key.sh                 # prompt (no echo), validate, write
#   pbpaste | bash scripts/set_deepseek_key.sh       # key from stdin (never echoed)
#   bash scripts/set_deepseek_key.sh --check         # validate only, write nothing
#   bash scripts/set_deepseek_key.sh --show          # report current state (redacted)
#   bash scripts/set_deepseek_key.sh --no-verify     # skip the API check (offline)
#   bash scripts/set_deepseek_key.sh --clear-secret  # delete the stale SecretStorage
#                                                    # entry (requires VS Code QUIT)
#
#   Any OpenAI-compatible endpoint (e.g. Unsloth Studio on loopback):
#   bash scripts/set_deepseek_key.sh --setting chat.openai.apiKey \
#        --base-url http://127.0.0.1:8888 --no-verify
#
#   --settings PATH   override settings.json   (or VSCODE_USER_SETTINGS)
#   --state-db PATH   override state.vscdb     (or VSCODE_STATE_DB)
#   --setting NAME    setting to write (default: deepseek-copilot.apiKey)
#   --base-url URL    endpoint probed by --check (default: https://api.deepseek.com)
#
# Exit codes:
#   0 — key validated and written (or --show/--clear-secret succeeded)
#   1 — bad usage
#   2 — settings.json missing / not writable / not plain JSON
#   3 — the endpoint rejected the key (HTTP 401)
#   4 — the endpoint answered with another error (402 balance, 429 rate limit, ...)
#   5 — VS Code is running; --clear-secret needs it fully quit
#   6 — aborted by the user
# ─────────────────────────────────────────────────────────────────────────────

set -u
# NOTE: deliberately NOT using `set -e` — we want controlled error handling
# and a precise exit code per failure mode.

# ── Paths ──────────────────────────────────────────────────────────────────
VSCODE_USER_DIR="${VSCODE_USER_DIR:-$HOME/Library/Application Support/Code/User}"
SETTINGS="${VSCODE_USER_SETTINGS:-$VSCODE_USER_DIR/settings.json}"
STATE_DB="${VSCODE_STATE_DB:-$VSCODE_USER_DIR/globalStorage/state.vscdb}"
BASE_URL="https://api.deepseek.com"
SETTING_KEY="deepseek-copilot.apiKey"

# ── Args ───────────────────────────────────────────────────────────────────
MODE="write"
VERIFY=1
while [ $# -gt 0 ]; do
  case "$1" in
    --show)         MODE="show"; shift ;;
    --check)        MODE="check"; shift ;;
    --clear-secret) MODE="clear-secret"; shift ;;
    --no-verify)    VERIFY=0; shift ;;
    --settings)     SETTINGS="${2:-}"; shift 2 ;;
    --state-db)     STATE_DB="${2:-}"; shift 2 ;;
    --setting)      SETTING_KEY="${2:-}"; shift 2 ;;
    --base-url)     BASE_URL="${2:-}"; shift 2 ;;
    -h|--help)
      sed -n '2,49p' "$0"
      exit 0
      ;;
    *)
      printf 'Unknown flag: %s (try --help)\n' "$1" >&2
      exit 1
      ;;
  esac
done

# ── Helpers ────────────────────────────────────────────────────────────────
# SecretStorage only ever holds the DeepSeek extension key; other settings
# (e.g. chat.openai.apiKey) live exclusively in settings.json.
secret_present () {
  [ "$SETTING_KEY" = "deepseek-copilot.apiKey" ] || return 1
  [ -f "$STATE_DB" ] || return 1
  command -v sqlite3 >/dev/null 2>&1 || return 1
  local n
  n="$(sqlite3 "file:${STATE_DB}?mode=ro" \
        "SELECT COUNT(*) FROM ItemTable WHERE key LIKE 'secret://%deepseek-copilot.apiKey%';" 2>/dev/null)"
  [ "${n:-0}" -gt 0 ] 2>/dev/null
}

setting_present () {
  [ -f "$SETTINGS" ] || return 1
  LC_ALL=C grep -q "\"deepseek-copilot.apiKey\"" "$SETTINGS" 2>/dev/null
}

# NOTE: `pgrep -f` does not reliably match the app's main binary on macOS
# (it is .../Contents/MacOS/Code, and only the Framework helpers match), so
# check the process list explicitly and, as a fallback, whether anything still
# holds state.vscdb open (WAL handle ⇒ VS Code is alive).
vscode_running () {
  if ps -Ao args= 2>/dev/null | LC_ALL=C grep -qE "Code(- Insiders)?\.app/Contents/MacOS/(Code|Electron)"; then
    return 0
  fi
  lsof -t -- "$STATE_DB" >/dev/null 2>&1
}

# Returns 0 when the key is accepted, 3 on 401, 4 on any other non-200.
validate_key () {
  local key="$1" code
  command -v curl >/dev/null 2>&1 || { printf 'curl not found — cannot verify\n' >&2; return 4; }
  # Authorization header is passed on stdin ('-H @-'), so the key never shows
  # up in the process list.
  code="$(printf 'Authorization: Bearer %s\n' "$key" \
    | curl -s -o /dev/null -w '%{http_code}' --max-time 15 -H @- "${BASE_URL}/v1/models")"
  case "$code" in
    200) printf '  ↳ key accepted by %s\n' "$BASE_URL"; return 0 ;;
    401) printf '  ↳ HTTP 401 — key invalid or revoked\n' >&2; return 3 ;;
    402) printf '  ↳ HTTP 402 — account has no balance\n' >&2; return 4 ;;
    429) printf '  ↳ HTTP 429 — rate limited, cannot verify right now\n' >&2; return 4 ;;
    000) printf '  ↳ no response from %s (offline?) — use --no-verify\n' "$BASE_URL" >&2; return 4 ;;
    *)   printf '  ↳ unexpected HTTP %s\n' "$code" >&2; return 4 ;;
  esac
}

read_key () {
  local key=""
  if [ -t 0 ]; then
    printf 'Paste the DeepSeek API key (not echoed, Ctrl-C to abort): '
    if ! IFS= read -rs key; then
      printf '\n[read error]\n' >&2
      return 1
    fi
    printf '\n'
  else
    # Non-interactive: one line from stdin, e.g. `pbpaste | ...`
    IFS= read -r key || true
  fi
  # Tolerate a trailing newline / CR from pbpaste or a password manager.
  key="${key%$'\r'}"
  if [ -z "$key" ]; then
    printf 'Empty key — nothing to do.\n' >&2
    return 1
  fi
  case "$key" in
    sk-*) ;;
    *) printf 'Warning: key does not start with "sk-" — continuing anyway.\n' >&2 ;;
  esac
  printf '  ↳ received %d character(s)\n' "${#key}"
  _secret_input="$key"
}

# ── --show ─────────────────────────────────────────────────────────────────
if [ "$MODE" = "show" ]; then
  printf 'API key status for %s (values redacted)\n' "$SETTING_KEY"
  printf '  settings.json        : %s\n' "$SETTINGS"
  if setting_present; then
    printf '  setting              : present (plaintext in settings.json)\n'
  else
    printf '  setting              : unset\n'
  fi
  if [ "$SETTING_KEY" = "deepseek-copilot.apiKey" ]; then
    printf '  state.vscdb          : %s\n' "$STATE_DB"
    if secret_present; then
      printf '  SecretStorage        : present → SHADOWS the setting above\n'
      printf '                         clear it once: bash %s --clear-secret\n' "$0"
    else
      printf '  SecretStorage        : empty\n'
    fi
    printf '  NOTE: a key cached in the RUNNING VS Code extension host is\n'
    printf '        invisible here and survives a DB edit until a window reload.\n'
  fi
  exit 0
fi

# ── --clear-secret ─────────────────────────────────────────────────────────
if [ "$MODE" = "clear-secret" ]; then
  if [ ! -f "$STATE_DB" ]; then
    printf 'No state.vscdb at %s — nothing to clear.\n' "$STATE_DB"
    exit 0
  fi
  if ! secret_present; then
    printf 'No SecretStorage entry for the DeepSeek key — nothing to clear.\n'
    exit 0
  fi
  if vscode_running; then
    printf 'VS Code is running — its in-memory copy would overwrite the deletion.\n' >&2
    printf 'Do ONE of these instead:\n' >&2
    printf '  a) Command Palette (Cmd+Shift+P) → "DeepSeek: Clear API Key"\n' >&2
    printf '  b) Quit VS Code (Cmd+Q), then re-run:  bash %s --clear-secret\n' "$0" >&2
    exit 5
  fi
  BACKUP="${STATE_DB}.bak_$(date -u +%Y%m%dT%H%M%SZ)"
  # Deleting this entry throws away a working key, so require an explicit
  # confirmation — only useful when the extension is failing with HTTP 401.
  printf 'This deletes the stored key%s. Only do this when the extension is\n' \
    " for $(basename "$STATE_DB")"
  printf 'failing with HTTP 401 (invalid key). Otherwise you lose a working key.\n'
  printf 'Type "clear" to confirm: '
  IFS= read -r _confirm
  if [ "$_confirm" != "clear" ]; then
    printf 'Aborted — nothing deleted.\n'
    exit 6
  fi
  cp -p "$STATE_DB" "$BACKUP" || { printf 'Backup failed — aborting.\n' >&2; exit 2; }
  chmod 600 "$BACKUP" 2>/dev/null || true
  if sqlite3 "$STATE_DB" \
      "DELETE FROM ItemTable WHERE key LIKE 'secret://%deepseek-copilot.apiKey%';" 2>/dev/null; then
    printf 'SecretStorage entry deleted (backup: %s)\n' "$BACKUP"
    printf 'Start VS Code again — the %s setting now takes effect.\n' "$SETTING_KEY"
    exit 0
  fi
  printf 'sqlite3 DELETE failed; state.vscdb untouched. Backup: %s\n' "$BACKUP" >&2
  exit 2
fi

# ── --check / write ────────────────────────────────────────────────────────
read_key || exit 6

KEY="$_secret_input"
_secret_input=""
trap 'unset KEY _secret_input 2>/dev/null || true' EXIT

if [ "$VERIFY" -eq 1 ]; then
  validate_key "$KEY"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    printf 'Key NOT stored (validation failed).\n' >&2
    exit "$rc"
  fi
else
  printf '  ↳ validation skipped (--no-verify)\n'
fi

if [ "$MODE" = "check" ]; then
  printf 'Key is valid. Nothing written (--check).\n'
  exit 0
fi

# ── Write the setting (atomic, with backup) ────────────────────────────────
# The helper goes to a 0600 temp file and the key travels through its stdin —
# neither the key nor the program is ever in argv (`ps` output).
_helpers="$(mktemp "${TMPDIR:-/tmp}/set_deepseek_key.XXXXXX")" || exit 2
chmod 600 "$_helpers"
cat > "$_helpers" <<'PY'
import datetime
import json
import os
import shutil
import sys
import tempfile

path, setting_key = sys.argv[1], sys.argv[2]
key = sys.stdin.readline().rstrip("\n")
if not key:
    sys.exit("no key on stdin")

data = {}
existed = os.path.exists(path)
if existed:
    with open(path, encoding="utf-8") as fh:
        raw = fh.read().strip()
    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            sys.exit(
                f"{path} is not plain JSON ({exc}); it has comments, so add this "
                f'line by hand: "{setting_key}": "<your key>"'
            )
    if not isinstance(data, dict):
        sys.exit(f"{path} does not contain a JSON object")

if existed:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = f"{path}.bak_{stamp}"
    shutil.copy2(path, backup)
    os.chmod(backup, 0o600)

data[setting_key] = key
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".settings.", suffix=".tmp")
try:
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4, ensure_ascii=False)
        fh.write("\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
except BaseException:
    os.unlink(tmp)
    raise

print(f"  wrote {setting_key} (length {len(key)}) to {path}")
PY
printf '%s' "$KEY" | python3 "$_helpers" "$SETTINGS" "$SETTING_KEY"
rc=$?
rm -f "$_helpers"
KEY=""
[ "$rc" -eq 0 ] || { printf 'Write failed.\n' >&2; exit 2; }

# ── Final state report ─────────────────────────────────────────────────────
printf '\n'
if secret_present; then
  printf 'WARNING: a key is still stored in VS Code SecretStorage, and it SHADOWS\n' >&2
  printf 'the setting just written. Clear it once:\n' >&2
  printf '  Command Palette → "DeepSeek: Clear API Key"   (fastest)\n' >&2
  printf '  or: quit VS Code, then bash %s --clear-secret\n' "$0" >&2
else
  printf 'Done. Reload the VS Code window (Cmd+R) so the extension re-reads the key.\n'
fi
exit 0
