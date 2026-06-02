# myvouvis na Kubernetes (e-infra.cz)

Nasazení **vouvis-api**: LM2 detekce štítku + vision LLM → DwC JSON.

LM2 váhy (`vendor/lm2/weights/best.pt`, ~268 MB) jsou **součástí Docker image** — PVC pro model nepotřebujete.

---

## Co kam patří

```text
┌──────────────────────────────────────────────┐
│  Deployment vouvis-api                       │
│  ┌─────────────────┐                         │
│  │ Docker image    │  vendor/lm2/weights/    │
│  │ myvouvis        │  best.pt (v image)      │
│  └────────┬────────┘                         │
│           ├──▶ e-infra LLM (vision)          │
│           └──▶ HTTP /v1/transcribe (volitelně│
│                batch přes jiný proces)       │
└──────────────────────────────────────────────┘
```

---

## Krok 1 — Secret s přihlašovacími údaji

Secret **`vouvis-creds`** ve vašem namespace (např. `pokorny1-ns`).

| Klíč | Popis |
|------|--------|
| `OPENAI_API_KEY` | Klíč k e-infra LLM |
| `OPENAI_LLM_PRESET` | např. `gpt-oss-120b` |
| `DB_*`, `S3_*` | Jen pokud spouštíte batch `herbarium-dwc` jinde |

`LM2_WEIGHTS_PATH` **není potřeba** — image používá `/app/vendor/lm2/weights/best.pt`.  
Volitelný override: `LM2_WEIGHTS_PATH=/jiná/cesta/best.pt`.

`OPENAI_BASE_URL` je v [`k8s/deployment.yaml`](k8s/deployment.yaml) natvrdo.

### Přes kubectl (ukázka)

```bash
kubectl create secret generic vouvis-creds -n pokorny1-ns \
  --from-literal=OPENAI_API_KEY='váš-klíč' \
  --from-literal=OPENAI_LLM_PRESET='gpt-oss-120b'
```

V Rancheru: **Storage → Secrets → vouvis-creds → Edit Config**.

---

## Krok 2 — Deployment API

1. Upravte namespace v manifestech, pokud je potřeba.
2. Image musí běžet **uvicorn**, ne batch scheduler — v deploymentu nastavte:

```yaml
command: ["uvicorn"]
args: ["api.app:app", "--host", "0.0.0.0", "--port", "8080"]
```

3. Nasazení:

```bash
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/ingress.yaml
```

4. Po pushi na `main` se image rebuildne v GitHub Actions (váhy v image díky Git LFS).

---

## Krok 3 — Test

```bash
curl -F "file=@arch.jpg" https://vase-url/v1/transcribe
```

Nebo uvnitř clusteru přes Service / Ingress.

---

## Kontrolní seznam

- [ ] Secret `vouvis-creds` obsahuje `OPENAI_API_KEY` (+ preset)
- [ ] Deployment běží, `/healthz` vrací OK
- [ ] `/v1/transcribe` vrací `dwc` + `validation`

---

## Časté problémy

| Problém | Co zkontrolovat |
|---------|------------------|
| `LM2 weights not found` | Starý image bez vah; rebuild po pushi s LFS; `git lfs pull` při buildu |
| `OPENAI_API_KEY is not set` | Secret nepřipojený (`envFrom`) |
| Image **PullBackOff** | oprávnění k `ghcr.io` |

---

## Git LFS (vývojáři)

Soubor `vendor/lm2/weights/best.pt` je v **Git LFS** (>100 MB limit GitHubu).

```bash
git lfs install
git lfs pull
```

CI: workflow `.github/workflows/docker-publish.yml` checkoutuje LFS automaticky.

Další info: [`docs/getting-started.md`](docs/getting-started.md).
