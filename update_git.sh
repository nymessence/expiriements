#!/usr/bin/env bash
set -e

# === Git config setup ===
GIT_USER="humid_ray_0u@icloud.com"
GIT_EMAIL="nymessence"

git config user.name "$GIT_USER"
git config user.email "$GIT_EMAIL"

# === Load GitHub token ===
TOKEN_FILE="$HOME/.gh_token"
if [[ ! -f "$TOKEN_FILE" ]]; then
  echo "❌ GitHub token file not found at $TOKEN_FILE"
  echo "👉 Create a new token at: https://github.com/settings/tokens"
  exit 1
fi
GITHUB_TOKEN=$(<"$TOKEN_FILE")

# === Get current remote URL ===
REMOTE_URL=$(git config --get remote.origin.url)

# === Normalize and inject token ===
if [[ "$REMOTE_URL" == git@github.com:* ]]; then
  # SSH format -> convert to HTTPS
  REPO_PATH="${REMOTE_URL#git@github.com:}"
  REMOTE_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${REPO_PATH}"
elif [[ "$REMOTE_URL" == https://github.com/* ]]; then
  # HTTPS format without token
  REPO_PATH="${REMOTE_URL#https://github.com/}"
  REMOTE_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${REPO_PATH}"
elif [[ "$REMOTE_URL" == https://x-access-token:*@github.com/* ]]; then
  # Already contains a token — update it with current token
  REPO_PATH="${REMOTE_URL#*@github.com/}"
  REMOTE_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${REPO_PATH}"
else
  echo "❌ Unsupported remote URL format: $REMOTE_URL"
  exit 1
fi

git remote set-url origin "$REMOTE_URL"

# === Commit process ===
read -p "Enter your commit message: " COMMIT_MSG

git add .
echo
git status
echo

read -p "Proceed with commit and push? [y/N]: " CONFIRM
if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
  git commit -m "$COMMIT_MSG" || echo "⚠️ Nothing to commit."

  echo "🔁 Pushing to remote..."
  if git push origin HEAD 2>&1 | tee /tmp/git_push_output | grep -qi "Authentication failed"; then
    echo
    echo "❌ Authentication failed."
    echo "🔐 Your GitHub token may have expired."
    echo "👉 Generate a new one at: https://github.com/settings/tokens"
    echo "   Then save it to: $TOKEN_FILE"
  else
    echo "✅ Changes pushed successfully."
  fi
else
  echo "❌ Commit aborted."
fi

