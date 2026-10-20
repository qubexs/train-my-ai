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

## Advanced II (SPL/FFI/async/crypto/i18n/integrations)

SPL streaming (`SplFileObject` CSV!), heaps/queues, recursive iterators;
generators bidirectional (`send/throw/getReturn`, `yield from`); FFI
(structs/callbacks/free/preload-gated!); Fibers + Revolt/Amp (never manual
threads!); WeakMap cycles; readonly deep-copy semantics; backed enums +
methods/traits; attributes reflected (cached!); FCC over string callables;
`match` guards exhaustive; interfaces (`Stringable/ArrayAccess/Iterator/
Countable/Observer`); CSV injection (`=+...@` prefixed!); Excel chunk filters;
PDF merge (FPDI) vs style (mPDF) vs quality (Chrome); QR/EAN check digits;
JWT JWK rotation + constraints; OAuth2 PKCE + state + refresh; SAML metadata
+ SLO; TOTP window + hashed backups; WebAuthn counter + UV; Casbin model/policy;
respect/rakit chains; PHP-DI compiled prod; PSR-14 stoppable; syslog central;
dotenv immutable + vault rotation; Symfony tables/progress/signals; Prophecy/
Codeception/Behat layers; Infection MSI + Deptrac layers + churn hotspots;
phpDocumentor guides; Blackfire compares; Amp/ReactPHP streams; Swoole
tables/timers; RoadRunner KV/Temporal; FrankenPHP workers; parallel Runtime
(no-GIL!); OPcache file-cache + preload + JIT counters; Redis sessions +
locks; pepper + dual-rotate; sodium sealed/nonces/memzero; mbstring strict;
ICU prices/plurals/dates; streaming hashes; CMS timestamps; GnuPG isolated;
KX directions split; zlib streaming; Zip AES (never ZipCrypto!); curl_multi
+ abort callbacks; socket servers framed; TLS verify (never off!); FTP
passive+binary; Gearman background; AMQP topology-as-code + DLX.

## Beginner path (W3Schools flow)

Install (XAMPP/MAMP or `php -S` built-in dev server!) → syntax/comments →
variables/scope → echo/print → datatypes → strings (+modify/concat/slice/escape)
→ numbers/casting/math → constants/magic → operators/if/switch/match → loops
(all 5 + break/continue) → functions → arrays (+10 sub-pages: create/access/
update/add/remove/sort/multidim/functions) → superglobals ($_GET/$_POST/
$_SERVER/$_REQUEST) → regex → forms (required/complete/URL-email) → date →
include → files (open/read/create/upload) → cookies/sessions → filters →
callbacks → JSON → exceptions → OOP (+14: classes/construct/destruct/modifiers/
inherit/constants/abstract/interface/traits/static/namespace/iterables) → MySQL
(connect/create/insert/lastid/multiple/prepared/select/where/orderby/delete/
update/limit) → XML (SimpleXML/DOM/expat) → AJAX (intro/db/XML/live-search/poll)
→ reference (array/calendar/date/directory/error/exception/filesystem/filter/
FTP/JSON/keywords/libxml/mail/math/misc/mysqli/network/output/regex/simplexml/
stream/string/variable).

## Projects (GFG ladder)

PDF download headers → auto-download → image resize (GD memory!) → uploader →
link extractor → signup+MySQL → admin login (guards/throttle) → chat polling →
covid tracker (API consume). Each: validate → store → PRG → test.
