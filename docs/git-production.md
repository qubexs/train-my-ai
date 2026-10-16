# Production Git Reference

## Daily ritual

`status -sb` (branch+ahead/behind!) → `diff` (unstaged!) → `add -p` (hunks
reviewed!) → `commit -m` (imperative + scope!) → `fetch` → `rebase`
origin/main (linear!) → `push -u` once (bare after!). Log graph (`--graph
--decorate --all`) read before complex merges. Branches per task (deleted
after landing merged!).

## Undo map (local vs shared!)

Unpushed: `amend` (last commit!), `reset --soft/mixed` (keep work!),
`restore --staged` (unstage!). Pushed/shared: `revert` (new commit, safe!),
NEVER reset+force (destroys others!). Lost work: `reflog` (90 days!) →
`checkout -b rescue <sha>`. Stash named (`-m`!), popped (not applied!),
dropped after verify (or worktrees avoid stashing entirely!).

## Collaborate/review/release

Fetch+rebase before PRs (linear!); tests green locally FIRST; `gh pr create
--fill` (templates!) + `checks --watch` (never request red reviews!);
squash+delete (clean main!); signed tags/releases (`-s`, supply chain!);
forks synced (stale forks conflict!); `CONTRIBUTING` read (conventions!).

## Files/identity/safety

`.gitignore` reviewed (secrets never! `!` re-includes examples, `rm --cached`
for tracked!); attributes normalize (EOL!) + filter archives; LFS binaries
(models/datasets!, never code!); GPG/SSH signing (tags mandatory!); credential
helpers (never plaintext!); hooks automate (lint/test, `--no-verify`
emergencies documented!); `clean -ndx` previews (unrecoverable otherwise!);
bundles archive air-gap (verified!).
