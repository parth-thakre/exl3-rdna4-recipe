# Open WebUI in front of TabbyAPI

We run Open WebUI as a rootless podman container on the same machine, with host networking so it can reach TabbyAPI
on 127.0.0.1. Replace `<tabby-api-key>` with the `api_key` from `tabbyAPI/api_tokens.yml`.

```bash
podman run -d --name open-webui --network host --restart unless-stopped \
  -v open-webui:/app/backend/data \
  -e HOST=127.0.0.1 -e PORT=3080 \
  -e OPENAI_API_BASE_URL=http://127.0.0.1:8096/v1 \
  -e OPENAI_API_KEY=<tabby-api-key> \
  -e ENABLE_OLLAMA_API=false \
  -e WEBUI_NAME="Qwen3.8 on 9070 XT" \
  ghcr.io/open-webui/open-webui:main
```

Then open http://127.0.0.1:3080 and create the admin account.

To use it from other devices on a tailnet, set `HOST` to the machine's Tailscale IP (`tailscale ip -4`). If TabbyAPI
is bound to that IP too (`network.host` in `tabbyAPI/config.yml`), point `OPENAI_API_BASE_URL` at it as well, since
TabbyAPI will no longer listen on 127.0.0.1.

`--restart unless-stopped` only restarts the container while podman is running. Neither TabbyAPI nor this container
starts on boot in our setup; use a systemd user unit or `podman generate systemd` if you want that.

## Suggested chat settings

For hard reasoning questions, set `reasoning_effort` to `medium` and use the anti-spiral system prompt from
`notes/measurements.md` (per chat, or in the model's advanced parameters). On GPQA it removed every thinking spiral that hit the token
cap and was faster at equal or better accuracy. For the hardest questions, `xhigh` with the same prompt and a
large `max_tokens` gets more right (28 vs 9-12 of the 48 GPQA Diamond questions `medium` missed) but thinks about 5x
longer. We haven't made
either the default in our own Open WebUI yet.
