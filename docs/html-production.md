# Production HTML Reference

## Document head (order matters!)

`<!DOCTYPE html>` first → `<html lang dir>` → charset (<1024 bytes!) →
viewport → title (<60ch, unique) → description (~150ch) → canonical (exact!) →
robots → theme-color ×2 schemes → OG/Twitter (1200×630 absolute!) → icons
(every platform!) → manifest → RSS autodiscovery. Every tag justified.

## Semantics landmarks

One `h1`, ordered descent; `header/nav/main/article/section/aside/footer`
(true roles!); `figure+figcaption`, `time[datetime]`, `address[rel=author]`,
`data[value]`; `table` with `caption` first + `scope` + thead/tbody/tfoot;
`fieldset+legend`, `label[for]`, `aria-label/current/describedby`;
`role=dialog/status`; `hidden` vs `inert` vs `display:none` chosen per need.

## Forms/media/PWA/modern

Inputs typed (`email/number/date`), `inputmode`/`enterkeyhint` mobile,
`autocomplete` real tokens, `required/pattern/minlength` + server recheck;
`selectmenu` customizable; file `accept+capture`; media `preload/poster/track`
+ fragments; responsive (`srcset/sizes`, art-direction `<picture>`, DPR);
`decoding/fetchpriority` per image; manifest (icons/scope/id/shortcuts) +
SW strategies + install UX; JSON-LD `@graph` (+microdata fallback);
RSS/Atom + OpenSearch; resource hints measured (`preconnect/preload/
modulepreload`).

## Full pages (training seeds)

SEO+PWA head, semantic article, docs+sidebar, accessible login, captioned
table, responsive gallery, webmanifest, landing, printable invoice, helpful
404, inbox-safe email, 7-step a11y sprint.
