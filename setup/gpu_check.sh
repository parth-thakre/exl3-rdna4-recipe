# GPU settings check, sourced by container/run.sh, serve/run_tabby.sh and triage.sh. An undervolt or power limit that
# games run fine on can still hang a long prefill or turn the output into NaN (issue #2: -60 mV and a 275 W cap of
# 304 W), so the launchers warn before serving. Reads sysfs only; no root needed.

# gpu_not_stock: one line per AMD GPU with a voltage or clock offset, or a power cap below default
gpu_not_stock() {
    local c d h s v out
    for c in /sys/class/drm/card*; do
        [[ $c =~ card[0-9]+$ ]] && [ "$(cat "$c/device/vendor" 2>/dev/null)" = 0x1002 ] || continue
        d=$c/device out=""
        for s in VDDGFX:voltage SCLK:clock; do
            v=$(grep -A1 "^OD_${s%:*}_OFFSET:" "$d/pp_od_clk_voltage" 2>/dev/null | sed -n 2p | tr -d ' ') || true
            [ -n "$v" ] && [[ ! $v =~ ^0[A-Za-z]*$ ]] && out="$out, ${s#*:} offset $v"
        done
        for h in "$d"/hwmon/hwmon*; do
            [ -r "$h/power1_cap" ] && [ -r "$h/power1_cap_default" ] || continue
            [ "$(cat "$h/power1_cap")" -lt "$(cat "$h/power1_cap_default")" ] &&
                out="$out, power cap $(( $(cat "$h/power1_cap") / 1000000 )) W of $(( $(cat "$h/power1_cap_default") / 1000000 )) W"
        done
        [ -z "$out" ] || echo "$(basename "$(readlink -f "$d")"):${out#,}"
    done
    return 0
}

# gpu_stock_warning <script name>: warn on stderr if any GPU isn't at stock settings. In a terminal, ask whether to
# continue and return 1 on no; otherwise (the container, a service) only warn. ALLOW_NOT_STOCK=1 skips the question.
gpu_stock_warning() {
    local n ans; n=$(gpu_not_stock)
    [ -n "$n" ] || return 0
    {
        echo "$1: warning: GPU not at stock settings:"
        sed 's/^/  /' <<<"$n"
        echo "  Undervolts and power limits that games pass can hang long prompts or turn the output into a run of '!'."
        echo "  If that happens, set the card back to stock (LACT, CoreCtrl) first: README, \"GPU hangs and wrong output\"."
    } >&2
    [ -t 0 ] && [ -t 2 ] && [ -z "${ALLOW_NOT_STOCK:-}" ] || return 0
    read -r -p "Continue anyway? [y/N] " ans || ans=
    case $ans in [yY]*) return 0 ;; esac
    echo "$1: stopped (ALLOW_NOT_STOCK=1 skips this question)" >&2
    return 1
}
