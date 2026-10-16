# Production CSS Reference

## Cascade architecture

`@layer reset, tokens, base, components, utilities` (order wins, not
specificity!); `@scope` bounds components (proximity!); nesting `&`
(readable, output audited!); `@supports` guards (fallbacks first!);
`@starting-style` entries; `revert-layer` undos. Specificity wars ended
by architecture (layers+scope), not `!important`.

## Modern layout/position

Container queries (`@container` size AND style!) + units (`cqw` typography!);
subgrid inherits tracks (aligned card internals!); anchor positioning
(`anchor-name` + `position-area` + `position-try-fallbacks`, no JS measuring!);
`position: sticky` needs offset+room; `inset` shorthands; `z-index` layered
intentionally (`isolate` stops leaks!); masonry: columns today (vertical
order!), grid-masonry when stable.

## Motion/scroll/top-layer

`transition-behavior: allow-discrete` + `overlay` + `display` trio (enter/exit!);
view timelines (`entry`→`cover` ranges!) + named scroll timelines; `starting-style`
entries; `popover` (`auto` exclusive/`hint` coexists/`manual` JS!) + `:popover-open`;
`dialog::backdrop` + `closedby`; `overlay` ordering; `prefers-reduced-motion`
always honored (static fallbacks!).

## Color/type/forms/tables/print

`light-dark()` pairs (no variable doubling!); `color-mix()` blends;
relative syntax derives (`hsl(from ...)`); oklch palettes (perceptually even!)
+ `color-contrast()` text; `color-scheme` adapts chrome; `display-p3` gated.
Type: `text-wrap: balance/pretty`, `hyphenate-limit-*` + `lang!`, `size-adjust`
fallbacks, `font-synthesis: none` brand type, variant caps/numeric, optical
auto. Forms: `accent-color` native branding, `user-valid` (not premature!),
`field-sizing: content` autosize, `color-scheme` dark inputs. Tables: model
chosen (collapse vs separate!), `caption-side`, `empty-cells`.
Print: hide nav/buttons, full URLs, avoid breaks, `@page` numbers (tested to PDF!).

## Full stylesheets (training seeds)

Layered architecture (oklch tokens!), container-query cards, anchor tooltips,
snap carousel (+scroll buttons!), view-timeline reveals, dialog/popover system,
`light-dark()` theme, print invoice, accessible forms, oklch palette, CSS
masonry (+fallback!), pure-CSS timeline.
