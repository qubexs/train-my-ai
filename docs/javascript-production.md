# Production JavaScript Reference

## Language core

Placement (`defer` ordered vs `async` unordered, modules deferred);
semicolons always (ASI `return`-newline bug!); `const` default / `let`
reassign / `var` never (TDZ!); type table memorized (`typeof null` lies!);
precedence tiers + parens; ternary values (never nested twice); falsy list
(`[]`/`{}` truthy!); `for..of` values / `for..in` keys / `for await` streams;
labels escape nesting; slice (negatives!) vs legacy; pad/repeat/split;
trim + code points (emoji!); `toFixed` strings + `Number.isNaN` + EPSILON;
BigInt (`123n`, no mixing!); bitwise flags (commented!).

## Functions/objects/scope

Declarations hoist, expressions don't; defaults (not `||`!), rest (arrows lack
`arguments`!); arrows inherit `this` (never methods/constructors!); IIFE legacy;
closure loop fix (`let` per iteration!); timers pass args + clear + unref;
literals computed/spread; prototype chain (`hasOwnProperty`!); freeze/seal;
shadowing deliberate; strict everywhere; dates ISO-transport/local-display
(Temporal for logic!); holes never; comparators always; flat/flatMap depth;
find/some/every shaped; Sets dedupe + WeakSets annotate; Maps dynamic keys;
iterables custom via generators; `yield*` delegates (bounded!); random ranges
floored + crypto secrets; named regex groups; lookahead positions; destructuring
documents shape; typed errors + `cause`; throw objects + finally cleanup;
console levels/table/time; mistakes linted; DOM batched + measured.

## Web APIs

Loop order memorized (sync→micro→macro); debounce (idle) vs throttle (rate);
storage (JSON + private-mode try!); observers (intersection/mutation/resize,
disconnect!); fetch abortable (stale never renders!); sockets reconnect +
heartbeats; SSE auto-reconnect; workers (transferables!); service workers
(strategies per type + offline fallback); IndexedDB (transactions!);
canvas DPR-scaled + rAF; SVG namespaced; media needs gestures; permissions
pre-checked; history SPA (server renders same URLs!); hash for static hosts;
FormData native (no content-type!); blobs + revoke; DnD all-three events;
clipboard async + fallbacks; geolocation gated; notifications permission-first;
visibility sleeps work; beforeunload conditional; pointer unified + capture;
keyboard `code` shortcuts + `key` text; fullscreen gesture-bound; matchMedia
mirrors CSS; smooth scroll + focus moved; selection programmatic; contenteditable
sanitized (DOMPurify!); wakelock minimal; capability-adaptive loading;
prefs honored (color/motion/contrast); dialog top-layer; components
encapsulated; workers supervised; shared workers share; Workbox strategies;
FileReader content vs object-URL display; video events mirror states; audio
graphs declarative; speech async voices; CSSOM runtime theming; scroll-driven
declarative; view transitions native; WAAPI promises; eyedropper cancellable;
FS access permissioned; storage monitored; badges mirror counts; contacts
user-picked; vibration short; orientation per-experience; idle transparent;
pressure-aware; network-adapted; offline queued + visible; install UX designed;
share native + fallbacks; payments native + fallbacks; passkeys + fallbacks;
OTP formatted; detection native + fallbacks; hardware choosers explicit;
gamepads polled; WebGPU feature-detected; WebNN measured; WebCodecs
hardware-queried; WebRTC signaling custom + ICE monitored; datachannels per
concern; WebTransport with socket fallback; Trusted Types structural;
Sanitizer platform; COOP/COEP audited; reporting sampled; hints measured.

## Full apps (training seeds)

Todo (localStorage + textContent), validated forms, accessible modal, exact
countdown, ARIA tabs, debounced+aborted search, sortable table, DPR canvas
chart, quiz state machine, stopwatch deltas, fetch-with-retries, hash router.
