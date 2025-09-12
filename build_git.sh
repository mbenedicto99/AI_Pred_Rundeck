#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="AI_Pred_Rundeck"
VISIBILITY="private"   # mude para "public" se quiser repo público

# 1) Pré-requisitos (Ubuntu 24)
if ! command -v git >/dev/null 2>&1; then
  sudo apt update && sudo apt install -y git
fi
if ! command -v gh >/dev/null 2>&1; then
  sudo apt update && sudo apt install -y gh
fi
gh config set git_protocol ssh

# 2) Login no GitHub se necessário (abre o browser)
if ! gh auth status >/dev/null 2>&1; then
  gh auth login -s 'repo' -w
fi

# (Opcional) evitar erro de "dubious ownership" se estiver rodando como root
git config --global --add safe.directory "$PWD" || true

# 3) Inicializa o repositório Git (se ainda não existir)
if [ ! -d .git ]; then
  git init
  git checkout -b main
fi

# 4) Arquivos básicos (se não existirem)
[ -f README.md ] || cat > README.md <<'EOF'
# AI_Pred_Rundeck

Repositório para automação com Rundeck integrando OpenShift (ponte SSH via bastion).
EOF

[ -f .gitignore ] || cat > .gitignore <<'EOF'
# Python / gerais
__pycache__/
*.py[cod]
.env
.venv/
venv/
.ipynb_checkpoints/
.DS_Store
.idea/
.vscode/
dist/
build/
*.log
EOF

# 5) Commit inicial (se houver algo a commitar)
git add -A
git commit -m "chore: initial commit" || echo "Nada novo para commitar."

# 6) Cria (ou conecta) o remoto no GitHub e faz push
if gh repo view "$REPO_NAME" >/dev/null 2>&1; then
  # Já existe no GitHub — apenas conecta/push
  USERNAME="$(gh api user -q .login)"
  if ! git remote | grep -q '^origin$'; then
    git remote add origin "git@github.com:${USERNAME}/${REPO_NAME}.git"
  fi
else
  gh repo create "$REPO_NAME" --"$VISIBILITY" --source=. --remote=origin --push
fi

git push -u origin main
echo "✅ Repo criado/em sincronia: https://github.com/$(gh api user -q .login)/$REPO_NAME"

