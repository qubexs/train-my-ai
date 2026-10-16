# Production PHP Reference

## Language core

`echo` output vs `print` value; superglobals read-only sources (`$_GET` URLs,
`$_POST` forms, `$_SERVER` routing, `$_FILES` uploads — validated copies used,
never raw); `include` optionals vs `require` essentials (autoload replaces
both); typed functions (`strict_types=1` enforces); strings byte-vs-char
(`mb_*` for Unicode!, `!== false` for strpos); arrays (`[]` append, `+` union
left-wins, references `unset` after loops!); `sprintf` formats; JSON with
`JSON_THROW_ON_ERROR` (silent nulls never).

## Files/dates/sessions

`file_get_contents` + `LOCK_EX` writes (streams for big!); `fopen` modes
(`x` race-safe create, `c+` no-truncate); `mkdir -p` recursive; `scandir/glob`
lists; `DateTimeImmutable` (+`modify`/`diff`, UTC stored, `setTimezone`
display); sessions (`start` before output!, regenerate on login, destroy+cookie
on logout); cookies (`httpOnly/Secure/SameSite`, validated reads);
`header()+exit` atomic pair (code runs on without exit!).

## Errors/DB/security

Ordered `try/catch (Specific → \Throwable) + finally`; custom domain
exceptions; `set_exception_handler` last resort. PDO baseline (exceptions,
assoc fetch, real prepares — the ONLY baseline); placeholders for ALL inputs;
transactions multi-write; `password_hash/verify/needs_rehash` (cost auto-rises);
`filter_var` types at edges; `htmlspecialchars(ENT_QUOTES|HTML5)` at OUTPUT
(context differs!); CSRF per-session tokens + `hash_equals` (GET never mutates!);
uploads (error→real MIME→`move_uploaded_file`→random names→outside webroot);
`mail()` basic, libraries queued otherwise.

## HTTP/clients/XML

cURL (timeout+errno checked every call!, Guzzle for retry/middleware/pools,
MockHandler tests); SimpleXML quick, DOM+XPath complex, XXE disabled always
(`LIBXML_NONET`); responses typed per consumer (`json/download/stream/file`).

## OOP/Composer/testing

Promotion (`private string $x`) kills boilerplate; `self::` vs `static::`
(LSB!); interfaces multi, abstracts share once, traits mix (`insteadof`);
namespaces mirror dirs (PSR-4, Linux case!); composer `install` (lockfile, CI!)
vs `update` (reviewed), `audit` gates; PHPUnit xUnit + Pest specs + data
providers; Dotenv never committed; Monolog JSON levels; Guzzle pools.

## Ecosystem/runtimes/deploy

Symfony (Console/Finder/Process/YAML/Validator/Mailer), Twig (autoescape!),
Slim/Lumen micros; opcache (preload + no-validate-timestamps prod!) + JIT
(measured compute only!); Fibers (AMPHP/ReactPHP, never manual threads);
attributes/enums/`match` (exhaustive!)/named/union/nullsafe; FPM pools
(RAM-sized children, max_requests recycle) + `ping/status` probes; php.ini
prod (display OFF, log ON, sane limits); Xdebug dev-only + Sentry context;
atomic symlink deploys + FPM reload + opcache reset; FrankenPHP workers
(stateless discipline); audit/outdated scheduled; Psalm/PHPStan leveled;
Box PHAR distribution.
