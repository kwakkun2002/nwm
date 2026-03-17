# Dev Container Workflow

This document summarizes a practical workflow for using the `nwm:cu126` Docker development container without installing AI tools such as Codex CLI or Claude inside the container.

## Goal

Keep AI tools on the host machine while running Python, Jupyter, training, and evaluation commands inside the Docker container.

This works well because the repository is mounted into the container with:

```bash
-v "$PWD":/workspace/nwm
```

That means:

- Code is edited from the host.
- The same files are visible inside the container.
- Runtime dependencies remain isolated inside Docker.

## Recommended Pattern

Use this split of responsibilities:

- Host:
  - Cursor / editor
  - Codex CLI / Claude
  - Git operations
  - File editing
- Container:
  - Python environment
  - PyTorch / CUDA execution
  - Jupyter Notebook
  - Training / inference / evaluation commands

In short, do not "work inside the container" unless necessary. Instead, send commands into the running container from the host.

## Start the Container

For one-off usage:

```bash
docker run --rm -it --gpus '"device=1"' -p 8888:8888 -v "$PWD":/workspace/nwm nwm:cu126
```

For a reusable named container:

```bash
docker run -it --name nwm_dev --gpus "device=1" -p 8888:8888 -v "$PWD":/workspace/nwm nwm:cu126
```

Useful lifecycle commands:

```bash
docker ps -a
docker stop nwm_dev
docker start -ai nwm_dev
docker start nwm_dev
docker exec -it nwm_dev bash
docker rm nwm_dev
docker rm -f nwm_dev
```

## Preferred Way to Execute Commands

Instead of attaching to the container and typing manually, execute commands directly from the host:

```bash
docker exec -it -w /workspace/nwm nwm_dev python train.py --config config/nwm_cdit_xl.yaml
docker exec -it -w /workspace/nwm nwm_dev python isolated_nwm_infer.py ...
docker exec -it -w /workspace/nwm nwm_dev python isolated_nwm_eval.py ...
```

You can also run a shell command string:

```bash
docker exec -it -w /workspace/nwm nwm_dev bash -lc "python train.py --config config/nwm_cdit_xl.yaml"
```

You can also use the wrapper script in this repository:

```bash
./scripts/nwm-run.sh "python train.py --config config/nwm_cdit_xl.yaml"
./scripts/nwm-run.sh "python isolated_nwm_eval.py --datasets recon ..."
```

## Jupyter Without Attaching to the Container

Start Jupyter from the host by executing it inside the container:

```bash
docker exec -it -w /workspace/nwm nwm_dev bash -lc \
  "jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root"
```

Then open this in the host browser:

```text
http://localhost:8888
```

This gives you container-based Python execution while keeping the browser and editor on the host.

## Why This Works Well for AI Tools

This setup avoids installing Codex CLI or Claude inside the container.

Benefits:

- AI tools stay on the host machine.
- No duplicate installation inside Docker.
- Container image remains focused on runtime dependencies.
- The AI can still edit the same mounted repository files.
- Execution can still happen in the container with `docker exec`.

Typical workflow:

1. Start the container once.
2. Open the repository on the host.
3. Let the AI edit code on the host-mounted files.
4. Run Python or notebook commands in the container via `docker exec`.

## Useful Shell Helpers

To reduce repeated typing, add one of these to your shell config.

If you prefer a repository-local wrapper instead of shell config, use:

```bash
chmod +x ./scripts/nwm-run.sh
./scripts/nwm-run.sh "python train.py --config config/nwm_cdit_xl.yaml"
```

The script:

- Uses `nwm_dev` by default
- Starts the container automatically if it exists but is stopped
- Runs commands in `/workspace/nwm`
- Supports overrides with `NWM_CONTAINER_NAME` and `NWM_CONTAINER_WORKDIR`

### Alias

```bash
alias nwm-exec='docker exec -it -w /workspace/nwm nwm_dev'
```

Usage:

```bash
nwm-exec python train.py --config config/nwm_cdit_xl.yaml
nwm-exec bash
```

### Function

```bash
function nwm-run() {
  docker exec -it -w /workspace/nwm nwm_dev bash -lc "$*"
}
```

Usage:

```bash
nwm-run "python train.py --config config/nwm_cdit_xl.yaml"
nwm-run "python isolated_nwm_eval.py --datasets recon ..."
nwm-run "jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root"
```

## Suggested Daily Workflow

```bash
docker start nwm_dev
```

Edit files on the host with your editor and AI tools.

Run commands in the container:

```bash
docker exec -it -w /workspace/nwm nwm_dev python train.py --config config/nwm_cdit_xl.yaml
```

When needed, start notebook services from the host:

```bash
docker exec -it -w /workspace/nwm nwm_dev bash -lc \
  "jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root"
```

Stop the container when finished:

```bash
docker stop nwm_dev
```

## When It Becomes Slightly Inconvenient

This pattern is simple, but there are a few trade-offs:

- Every runtime command needs a `docker exec` prefix.
- Tools that automatically run `python`, `pip`, or `pytest` on the host may use the wrong environment.
- Path, file ownership, or permission differences can occasionally matter.

For this repository, those issues are usually manageable, especially if you standardize command execution through an alias or wrapper script.

## Summary

The recommended workflow is:

- Keep editor and AI tools on the host.
- Keep Python and CUDA dependencies inside Docker.
- Edit files from the host.
- Run code inside the container with `docker exec`.
- Expose services such as Jupyter with `-p 8888:8888`.

This is the cleanest way to use a Docker-based development environment without installing extra AI tooling in the container itself.
