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
