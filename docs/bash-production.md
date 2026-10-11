# Production Bash & Shell Scripting Reference

## Strict Mode & Error Handling

```bash
#!/usr/bin/env bash
set -euo pipefail
```

- `-e`: aborts on command failure (ignored in `if` tests, `&&`/`||` lists, traps — use explicit `|| return 1` for critical checks).
- `-u`: unset variables are fatal (guard empty arrays with `${arr[@]:-}`).
- `-o pipefail`: any pipeline stage failing fails the whole pipeline.

## Exit Codes & Pipelines

```bash
false | true
echo "first: ${PIPESTATUS[0]}"
```

Pipeline success masks intermediate failures. Conventional codes: `0` ok, `1` generic, `2` misuse, `126`/`127` exec problems.

## Tests, Regex, Loops

```bash
[[ -f "$f" && ! -s "$f" ]] && echo "empty file"
[[ "$d" =~ ^([0-9]{4})-([0-9]{2})-([0-9]{2})$ ]] && echo "year ${BASH_REMATCH[1]}"
shopt -s nullglob dotglob
for f in /var/log/*.log; do [[ -e "$f" ]] || continue; echo "$f"; done
while IFS= read -r line; do echo "$line"; done < input.txt
```

- `[[ ]]` avoids word-splitting surprises. `=~` fills `BASH_REMATCH` (0 full, 1+ groups).
- Quote `"$f"` always. Never `for f in $(ls)`. Redirect into `done <` (no subshell loss).

## Functions, Args, Expansions

```bash
check() { local t="${1:?need arg}"; systemctl is-active --quiet "$t" || return 1; }
while getopts ':vf:' o; do case "$o" in v) v=1;; f) f="$OPTARG";; :) exit 2;; \?) exit 2;; esac; done
shift $((OPTIND - 1))
echo "${PORT:-3000} ${#s} ${f%.log} ${s^^} ${s:0:4}"
pkgs=(a b c); for p in "${pkgs[@]}"; do echo "$p"; done
declare -A ports=([http]=80 [ssh]=22)
mapfile -t lines < input.txt
```

Functions return codes (0-255), data via stdout + `$(...)`. `local` everything.

## Advanced Built-ins

```bash
[[ "$d" =~ ^([0-9]{4})-([0-9]{2})$ ]] && echo "year ${BASH_REMATCH[1]}"
mapfile -t lines < f.txt
mapfile -d '' -t files < <(find . -print0)
mapfile -s 10 -n 20 sample < big.log
start=$EPOCHSECONDS; sleep 2; echo "$((EPOCHSECONDS - start))s"
echo "rand $SRANDOM"
trap 'echo "at ${FUNCNAME[0]:-main} (${BASH_SOURCE[0]}:$LINENO)"; caller' ERR
greet() { declare -n ref=$1; ref="hello"; }; v=""; greet v; echo "$v"
PS3='Choose: '; select o in start stop quit; do case $o in start) break;; esac; done
seq -w 1 10
```

- `=~` + `BASH_REMATCH` (0 full, 1+ groups): regex without grep forks.
- `mapfile`: `-t` strip newlines, `-d ''` NUL-delimited, `-s/-n` skip/take windows.
- `$EPOCHSECONDS` timestamps without forks; `$SRANDOM` 32-bit random (not crypto).
- `${FUNCNAME[@]}`, `${BASH_SOURCE[@]}`, `caller`: stack traces in ERR traps.
- `declare -n ref=$1`: pass-by-reference; validate (circular names recurse forever).
- `select` + `PS3`: numbered menus; `REPLY` holds raw input for validation.
- Empty-array guard under `set -u`: `${arr[@]:-}` (bare `${arr[@]}` errors when empty).

## Traps, Streams, Jobs

```bash
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
cmd > run.log 2>&1
cmd | tee -a run.log
cat <<'EOF' > c.yaml   # quoted heredoc = literal, no expansion
diff -u <(ls a) <(ls b)
task1 & task2 & wait
trap 'echo "fail line $LINENO in ${FUNCNAME[0]:-main}"' ERR
```

Process substitution feeds command output as files. `wait -n -p pid` names the first done.

## Text Processing & Diagnostics

```bash
find . -name '*.tmp' -print0 | xargs -0 -r -P4 -n1 gzip
awk '{print $1}' access.log | sort | uniq -c | sort -nr | head -10
cut -d: -f1,7 /etc/passwd
ss -tlnp
ssh -N -L 8080:127.0.0.1:80 user@host
ssh -N -D 1080 user@host
```

`xargs -0 -r` is NUL-safe and skips empty runs. `ss` replaces netstat.

## Text Tools: awk, sed, jq

```bash
awk -F, '{s+=$3} END {print s}' f.csv   # sum column; BEGIN/END frame the program
awk '$3 > 100' f.csv                    # filter rows without grep+cut chains
sed 's/old/new/g' f                     # substitute; -i.bak edits in place w/ backup
sed '/pattern/d' f                      # delete matching lines
jq '.users[].name' -r f.json            # extract; -r raw strings for shell loops
jq '.items[] | select(.stok < 10)' f.json
```

Use `awk` for columnar data, `sed` for line edits, `jq` for JSON — never parse JSON with grep.
