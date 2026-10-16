# Production Linux Reference

## Triage order

Disk (`df -h` + `df -i` + `du -xhd1|sort -rh` + `lsof|deleted` + journal size)
→ memory (`free` available!, `vmstat si/so/wa`) → CPU (`uptime÷cores`, `mpstat -P
ALL`, `pidstat -t`) → disk IO (`iostat -x await/util`) → net (`ss -s`, `iftop`)
→ logs (`journalctl -p err --since`). Five commands diagnose 90% pages.

## Users/auth/firewall

Provision (`useradd -m -G`, 700/600 SSH, `passwd -l` key-only); `sudoers`
scoped via `visudo` (syntax-checked!); UFW (SSH first, then enable!) /
firewalld zones / nftables atomic; fail2ban jails + whitelists; SELinux
enforcing (diagnose contexts, never permanently disable); AppArmor
complain-then-enforce; SSH (`ed25519`, hardened `sshd_config` validated with
`sshd -T` before restart!, tunnels `-L/-R/-D`, autossh).

## Storage/services/boot

Partition (`fdisk` GPT!) → format → UUID → fstab (`nofail`!) →
`findmnt --verify` before reboot; LVM extend (`-r` resizes FS!); swap
(file/zram, swappiness per workload); systemd (`enable --now`, `is-active`,
timers replace cron with logging, `analyze blame` boots); logrotate before
disks fill; sysctl persisted (`/etc/sysctl.d/`, reboot wipes runtime!);
limits (`ulimit`, `LimitNOFILE`); nice/ionice/taskset tame neighbors.

## Network/debug

`ip` suite (addr/route/link/neigh — ifconfig deprecated!); DNS (`dig +short`,
`+trace`, resolver pinned); path (`traceroute -T`, `mtr` loss patterns);
`nc -zv` probes; `tcpdump` precise captures; `strace -f` file access;
`opensnoop` live; backups (`rsync --link-dest` snapshots, borg/restic with
tested restores!); upgrades listed-then-applied with `needrestart` checks.

## Self-help first (KodeKloud method)

`cmd --help` (seconds, alphabetical flags!) before `man cmd` (minutes: NAME/
SYNOPSIS/DESCRIPTION/OPTIONS/EXAMPLES, pager `/`+`n`+`q`); sections qualify
(`man 1 printf` command vs `man 3 printf` C function!); `apropos -s 1,8`
finds by purpose (generic stems match more! + `mandb` on fresh VMs);
double-TAB lists options, TAB completes paths/names (bash-completion package!);
`info` for GNU depth, `/usr/share/doc` for package notes, `help` for builtins,
`type -a`/`command -v` for truth (scripts!), `tldr`/`navi` for examples.

## Kernel admin (docs.kernel.org map)

Oops decoded (`decode_stacktrace.sh` + `addr2line`, symbols required!) and
reported per admin-guide; SysRq `REISUB` before power buttons (`sync` before
`reboot`!); livepatch CVE-without-reboot (majors still scheduled); cmdline
params versioned (`update-grub` after!); THP per workload (DBs `never`!);
KSM for homogeneous fleets (metered!); zswap/zram tiers beat disk swap;
lockdown+Secure Boot (`mokutil`) for compliance; crashkernel reserved +
`kdumpctl` verified + test crash in maintenance; watchdogs on (NMI/softlockup/
hung-task investigated, never blanket-off); earlyoom/nohang/systemd-oomd beat
kernel lottery on desktops; `fstrim.timer` weekly (not `discard` mounts!);
unattended-upgrades security-only + reboot windows; MOTD informs (patch/reboot
status); Cockpit for occasional admins (CLI stays source of truth);
tracing (`ftrace`/`perf`/eBPF) with overhead measured; fault injection proves
error paths (ENOMEM handling!).
