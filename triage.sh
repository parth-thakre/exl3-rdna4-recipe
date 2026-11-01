#!/bin/bash
# Something didn't work? This collects what an AI coding agent needs to help (distro, kernel, GPU and device access,
# which GPU runs the display, clocks and overdrive, amdgpu driver settings and kernel-log errors, container runtime and
# settings, recipe version, the last server and build logs) and opens the first agent CLI you have installed
# (Claude Code, Codex or OpenCode) with it, in this directory. The agent sends it to its model provider, like
# anything you type into it; tokens and keys are removed first, and in a terminal you're asked before it opens.
#
#   ./triage.sh                       collect and open an agent
#   ./triage.sh "what went wrong"     add your own description or the error message
#   AGENT=codex ./triage.sh           pick the agent (claude, codex or opencode)
#   OpenCode uses one of its free models (OPENCODE_MODEL=provider/model to pick, OPENCODE_MODEL=keep for your default)
#   ./triage.sh --print               only write the prompt file and print its path (paste it into any assistant)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
print_only=false
[ "${1:-}" = --print ] && { print_only=true; shift; }
problem=${*:-}

# Tokens and keys: long hex strings, common token prefixes, key/password/secret/token values in key: value,
# key=value and "key": "value" form, and Bearer headers. Applied to the whole prompt, your own text included.
redact() {
    sed -E 's/[0-9a-fA-F]{32,}/<redacted>/g
        s/(hf_|sk-|sk_|ghp_|gho_|ghs_|github_pat_|xox[abp]-|glpat-|AKIA)[A-Za-z0-9_-]{8,}/<redacted>/g
        s/("?[A-Za-z_]*(api[_-]?key|token|secret|password|passwd)[A-Za-z_]*"?[[:space:]]*[:=][[:space:]]*)("[^"]*"|[^[:space:],}]+)/\1<redacted>/Ig
        s/(Bearer)[[:space:]]+[^[:space:]"]+/\1 <redacted>/Ig'
}
section() { printf '\n### %s\n```\n' "$1"; }
endsec() { printf '```\n'; }
run() { "$@" 2>&1 | head -n "${LINES_MAX:-40}"; }

# Each DRM card: PCI address and IDs, whether it's the boot display (boot_vga) and which outputs have a display, VRAM
# use, clocks, the power cap and the overdrive table (pp_od_clk_voltage: clock and voltage offsets, i.e. an undervolt
# or overclock; readable when overdrive is enabled)
gpu_state() {
    local c d h n s v conn
    for c in /sys/class/drm/card[0-9]*; do
        [[ $c =~ card[0-9]+$ ]] && [ -e "$c/device/vendor" ] || continue
        d=$c/device n=$(basename "$c")
        echo "$n: $(basename "$(readlink -f "$d")") $(cat "$d/vendor"):$(cat "$d/device" 2>/dev/null) boot_vga=$(cat "$d/boot_vga" 2>/dev/null || echo ?)"
        conn=""
        for s in /sys/class/drm/"$n"-*/status; do
            [ "$(cat "$s" 2>/dev/null)" = connected ] && conn="$conn ${s#/sys/class/drm/$n-}"
        done
        echo "  displays connected:${conn:- none}" | sed 's|/status||g'
        [ -r "$d/mem_info_vram_used" ] &&
            echo "  VRAM used: $(( $(cat "$d/mem_info_vram_used") >> 20 )) / $(( $(cat "$d/mem_info_vram_total") >> 20 )) MiB"
        [ -r "$d/power_dpm_force_performance_level" ] && echo "  performance level: $(cat "$d/power_dpm_force_performance_level")"
        [ -r "$d/pp_dpm_sclk" ] && echo "  core clock levels: $(paste -sd' ' "$d/pp_dpm_sclk")"
        [ -r "$d/pp_dpm_mclk" ] && echo "  memory clock levels: $(paste -sd' ' "$d/pp_dpm_mclk")"
        for h in "$d"/hwmon/hwmon*; do
            [ -r "$h/power1_cap" ] && echo "  power cap: $(( $(cat "$h/power1_cap") / 1000000 )) W (default $(( $(cat "$h/power1_cap_default" 2>/dev/null || echo 0) / 1000000 )) W)"
            for s in "$h"/temp*_input; do
                [ -r "$s" ] && echo "  $(cat "${s%_input}_label" 2>/dev/null || basename "$s"): $(( $(cat "$s") / 1000 )) C"
            done
        done
        [ -r "$d/pp_od_clk_voltage" ] && { echo "  overdrive (pp_od_clk_voltage):"; head -n 20 "$d/pp_od_clk_voltage" 2>&1 | sed 's/^/    /'; }
        # Undervolts and power limits look stable in games and can still hang long prefills (issue #2)
        for s in OD_VDDGFX_OFFSET OD_SCLK_OFFSET; do
            v=$(grep -A1 "^$s:" "$d/pp_od_clk_voltage" 2>/dev/null | sed -n 2p | tr -d ' ')
            [ -n "$v" ] && [[ ! $v =~ ^0[A-Za-z]*$ ]] && echo "  NOT STOCK: ${s#OD_} is $v"
        done
        for h in "$d"/hwmon/hwmon*; do
            [ -r "$h/power1_cap_default" ] && [ "$(cat "$h/power1_cap")" -lt "$(cat "$h/power1_cap_default")" ] &&
                echo "  NOT STOCK: power cap below default"
        done
    done
}

# Kernel command-line GPU options, the amdgpu module parameters that matter for hangs, the firmware package and the
# firmware versions each card loaded, and overclocking tools that are running
amdgpu_driver() {
    local p v
    echo "kernel command line: $(tr ' ' '\n' </proc/cmdline | grep -E 'amdgpu|radeon|iommu|pci' | paste -sd' ' -)"
    for p in lockup_timeout gpu_recovery cwsr_enable mes user_queue runpm ppfeaturemask; do
        v=$(cat /sys/module/amdgpu/parameters/$p 2>/dev/null) && echo "amdgpu.$p = $v"
    done
    # Loaded GPU firmware (MES schedules the queues; MEC runs compute), per card, readable without root
    for d in /sys/class/drm/card[0-9]*/device/fw_version; do
        [ -d "$d" ] || continue
        echo "firmware loaded on $(basename "$(readlink -f "$d/..")"): $(for f in mes mes_kiq mec rlc smc sos; do
            printf '%s=%s ' "$f" "$(cat "$d/${f}_fw_version" 2>/dev/null || echo -)"; done)"
    done
    echo "firmware: $( { rpm -q linux-firmware amd-gpu-firmware 2>/dev/null; dpkg-query -W -f '${Package} ${Version}\n' linux-firmware 2>/dev/null; pacman -Q linux-firmware linux-firmware-amdgpu 2>/dev/null; } | grep -v 'not installed' | paste -sd' ' -)"
    for p in lactd corectrl; do pgrep -x "$p" >/dev/null && echo "running: $p (GPU clock / voltage tool)"; done
}

# amdgpu, KFD and DRM errors from the kernel log of one boot (0 = this one, -1 = the previous, for when a hang ended in
# a reboot). Reading it may need sudo or the systemd-journal group.
kernel_log() {
    local out
    if out=$(journalctl -k -b "$1" --no-pager -q -o short-monotonic 2>/dev/null) && [ -n "$out" ]; then :
    elif [ "$1" = 0 ] && out=$(dmesg 2>/dev/null) && [ -n "$out" ]; then :
    else
        echo "(not readable as you; with sudo: sudo journalctl -k -b $1 | grep -iE 'amdgpu|kfd|drm')"; return
    fi
    out=$(grep -iE 'amdgpu|kfd|\[drm' <<<"$out" | grep -iE '\b(error|fail\w*|fault|reset|hang|timeout|timed out|wedged|lost|corrupt\w*)\b' | tail -n 40)
    echo "${out:-no amdgpu errors}"
}

collect() {
    section "System"
    . /etc/os-release 2>/dev/null; echo "distro: ${PRETTY_NAME:-unknown}"
    echo "kernel: $(uname -r)"
    echo "recipe: $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout') $(git status --porcelain 2>/dev/null | grep -q . && echo '(local changes)')"
    echo "RAM: $(free -g 2>/dev/null | awk '/^Mem:/{print $2 " GB total, " $7 " GB available"}')"
    echo "disk: $(df -h . 2>/dev/null | awk 'NR==2{print $4 " free here"}')"
    endsec
    section "GPU"
    run lspci -nn 2>/dev/null | grep -iE 'vga|display|3d' || echo "lspci not available"
    lsmod 2>/dev/null | grep -q '^amdgpu' && echo "amdgpu module: loaded" || echo "amdgpu module: NOT loaded"
    ls -l /dev/kfd /dev/dri/renderD* 2>&1
    echo "you: $(id)"
    command -v amd-smi >/dev/null && run amd-smi metric --mem-usage 2>/dev/null | grep -E 'GPU|VRAM'
    command -v hipconfig >/dev/null && echo "hipconfig: $(hipconfig --version 2>/dev/null)" || echo "hipconfig: not installed (fine for the container)"
    endsec
    section "GPUs, display and clocks"
    gpu_state
    endsec
    section "amdgpu driver"
    amdgpu_driver
    endsec
    section "Kernel log: amdgpu errors (this boot, then the previous one)"
    kernel_log 0
    echo "--- previous boot"
    kernel_log -1
    endsec
    section "Container runtime"
    for ct in docker podman; do
        if command -v $ct >/dev/null; then
            echo "$ct: $($ct --version 2>&1)"
            $ct info >/dev/null 2>&1 && echo "  $ct info: ok" || echo "  $ct info: FAILS ($($ct info 2>&1 | tail -1))"
            $ct ps -a --filter name=exl3-rdna4 --format '  container {{.Names}}: {{.Status}} ({{.Image}})' 2>/dev/null
            img=$($ct inspect --format '{{.Image}}' exl3-rdna4 2>/dev/null) &&
                echo "  image: $($ct image inspect --format '{{join .RepoDigests " "}}' "$img" 2>/dev/null) ($img)"
            $ct inspect --format '{{range .Config.Env}}{{println .}}{{end}}' exl3-rdna4 2>/dev/null |
                grep -E '^(CTX|EXL3_|AMD_|HIP_|ROCR_)' | sed 's/^/  env: /'
        else
            echo "$ct: not installed"
        fi
    done
    endsec
    for ct in docker podman; do
        if command -v $ct >/dev/null && $ct ps -a -q --filter name=exl3-rdna4 2>/dev/null | grep -q .; then
            section "Container log ($ct logs --tail 80 exl3-rdna4)"
            $ct logs --tail 80 exl3-rdna4 2>&1 | redact
            endsec
        fi
    done
    if [ -d .venv ] || [ -d exllamav3 ] || [ -d logs ]; then
        section "Host install"
        echo "venv: $([ -x .venv/bin/python ] && .venv/bin/python -c 'import sys, torch; print(sys.version.split()[0], "torch", torch.__version__)' 2>&1 | tail -1 || echo missing)"
        echo "exllamav3 checkout: $([ -d exllamav3 ] && git -C exllamav3 rev-parse --short HEAD 2>/dev/null || echo missing)"
        echo "built extension: $(ls exllamav3/exllamav3_ext*.so 2>/dev/null || echo none)"
        echo "models: $(ls models 2>/dev/null | tr '\n' ' ')"
        endsec
        [ -f logs/tabby.log ] && { section "Server log (logs/tabby.log, last 80 lines)"; tail -n 80 logs/tabby.log | redact; endsec; }
        [ -f logs/build_exllamav3.log ] && {
            section "Build log (logs/build_exllamav3.log: errors, then the last 30 lines)"
            grep -aE 'error|Error|fatal' logs/build_exllamav3.log | head -20 | redact
            echo ...; tail -n 30 logs/build_exllamav3.log | redact
            endsec
        }
    fi
}

prompt_file=$(mktemp "${TMPDIR:-/tmp}/exl3-rdna4-triage.XXXXXX.md")
raw=$(mktemp)
{
cat <<EOF
I'm trying to run exl3-rdna4-recipe (Qwen3.8-27B with ExLlamaV3 + TabbyAPI on an AMD RX 9070 / 9070 XT) and something
isn't working. You're in the recipe's directory. Please help me get it running.

What went wrong, in my words: ${problem:-"(not given; work it out from the details below)"}

How to help:
- Start from the details below, then read README.md, container/README.md (the container route) and the
  "Limitations" in notes/measurements.md. The setup scripts are in setup/, the container files in container/.
- Explain what you think is wrong before changing anything, and ask before anything that needs sudo, installs
  packages, or deletes files (never delete downloaded models or the .venv without asking).
- If the GPU hung or reset (my desktop died, \`GPU Hang\` or \`MODE1 reset\` in the logs) or the output turned to
  garbage (a run of \`!\`, 0% of the draft accepted), read "GPU hangs and wrong output" in README.md and take me
  through its tests one at a time, starting with any NOT STOCK line below (an undervolt or power limit). Diagnostics that need sudo (the full kernel log, firmware versions) are fine to
  suggest: show me the command and what it reads, and I'll run it.
- Don't post anything online for me. If it looks like a bug in the recipe, write a GitHub issue I can review and file
  myself at https://github.com/parth-thakre/exl3-rdna4-recipe/issues, with the details that matter.

Details collected by ./triage.sh on $(date -u '+%Y-%m-%d %H:%M UTC'):
EOF
collect
} > "$raw"
redact < "$raw" > "$prompt_file"
rm -f "$raw"

if $print_only; then echo "$prompt_file"; exit 0; fi

fallback() {
    echo "The details are in: $prompt_file"
    echo "Paste that file into any AI assistant, or attach it to an issue at https://github.com/parth-thakre/exl3-rdna4-recipe/issues"
    exit "${1:-0}"
}

agents=(claude codex opencode)
[ -n "${AGENT:-}" ] && agents=("$AGENT")
for a in "${agents[@]}"; do
    command -v "$a" >/dev/null || continue
    if [ -t 0 ] && [ -t 1 ]; then
        echo "The details are in $prompt_file (keys and tokens removed; have a look if you like)."
        read -r -p "Open $a with them? It sends them to $a's model provider. [Y/n] " ans || ans=n
        case $ans in [nN]*) fallback ;; esac
    fi
    p=$(cat "$prompt_file")
    case $a in
        claude|codex) exec "$a" "$p" ;;
        opencode)
            # A free model by default, so this needs no paid model or API key of your own. OpenCode's free lineup changes,
            # so take big-pickle (its own free model) if it's offered, else the first opencode/*-free one.
            m=${OPENCODE_MODEL:-}
            if [ -z "$m" ]; then
                list=$(timeout 30 opencode models 2>/dev/null || true)
                m=$(grep -x 'opencode/big-pickle' <<<"$list" || grep -m1 -E '^opencode/.*-free$' <<<"$list" || true)
                # Not silently your own default, which may be a paid model
                [ -n "$m" ] || { echo "No free OpenCode model found; set OPENCODE_MODEL=provider/model (or =keep for your default)."; fallback 1; }
            fi
            if [ "$m" != keep ]; then
                [[ $m =~ ^[A-Za-z0-9._:/#-]+$ ]] || { echo "OPENCODE_MODEL doesn't look like provider/model: $m" >&2; exit 2; }
                echo "OpenCode model: $m"
                export OPENCODE_CONFIG_CONTENT="{\"model\": \"$m\"}"
            fi
            exec opencode --prompt "$p" ;;
        *)            echo "AGENT must be claude, codex or opencode" >&2; exit 2 ;;
    esac
done
echo "No agent CLI found (claude, codex or opencode)."
fallback
