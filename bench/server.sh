# Helpers for bench scripts that start their own TabbyAPI. Source it after setting ROOT; needs curl.
#   start_server LOG EXPECTED_MODEL [TabbyAPI args...]   launch serve/run_tabby.sh, wait until it serves EXPECTED_MODEL
#   stop_server                                          stop the server started by start_server
# Settings: TABBY_URL (default http://127.0.0.1:8096/v1), START_TIMEOUT seconds (default 900).

TABBY_URL=${TABBY_URL:-http://127.0.0.1:8096/v1}
while [[ $TABBY_URL == */ ]]; do TABBY_URL=${TABBY_URL%/}; done   # same as common.py's rstrip("/")
SERVER_PID=

api_key() {
    python -c 'import sys; sys.path.insert(0, sys.argv[1]); from common import api_key; print(api_key() or "")' "$ROOT/bench"
}

# Something already answering on the endpoint would be measured instead of the server we start
endpoint_busy() {
    curl -s -o /dev/null --max-time 5 "${TABBY_URL%/v1}/health"
}

# served_model: the loaded model's id (its directory name), empty if none
served_model() {
    local key
    key=$(api_key)
    curl -sf --max-time 10 ${key:+-H "Authorization: Bearer $key"} "$TABBY_URL/model" |
        python -c 'import json, sys; print(json.load(sys.stdin).get("id", ""))' 2>/dev/null
}

start_server() {
    local log=$1 expected=$2; shift 2
    if endpoint_busy; then
        echo "something is already serving on $TABBY_URL; stop it first (serve/stop_tabby.sh)" >&2
        return 1
    fi
    nohup "$ROOT/serve/run_tabby.sh" "$@" > "$log" 2>&1 < /dev/null &
    SERVER_PID=$!
    local deadline=$((SECONDS + ${START_TIMEOUT:-900})) model
    while [ $SECONDS -lt $deadline ]; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "TabbyAPI exited during startup; see $log" >&2
            SERVER_PID=; return 1
        fi
        model=$(served_model || true)
        if [ -n "$model" ]; then
            if [ "$model" = "$expected" ]; then return 0; fi
            echo "server came up with model '$model', expected '$expected'" >&2
            stop_server; return 1
        fi
        sleep 3
    done
    echo "TabbyAPI didn't come up within ${START_TIMEOUT:-900}s; see $log" >&2
    stop_server
    return 1
}

stop_server() {
    [ -n "$SERVER_PID" ] || return 0
    kill "$SERVER_PID" 2>/dev/null || true
    local i
    for i in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -0 "$SERVER_PID" 2>/dev/null && kill -9 "$SERVER_PID" 2>/dev/null
    SERVER_PID=
    return 0
}
