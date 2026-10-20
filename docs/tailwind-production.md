# Production Tailwind Reference

## Core loop

Utilities compose in HTML (`flex gap-4 p-4 md:grid-cols-3`); breakpoints layer
upward mobile-first (`sm md lg xl 2xl`); states stack (`hover: focus:
disabled:`); always pair hover with focus for keyboard users.

## Layout

Flex one dimension (`flex-col md:flex-row`, `items-center justify-center`,
`gap-*`, `grow/shrink/basis`); grid two (`grid-cols-N`, `auto-fit minmax()`);
`container mx-auto px-4` page shells; `min-h-screen` heroes; `max-w-prose`
readable text; `min-w-0` fixes flex truncation; container queries (`@container`
+ `@md:`) size components by their box, not viewport.

## Type/color/surface

Type scale + weights + `line-clamp-N` + `truncate`/`break-words`; palette
shades (`blue-600→700` on hover); theme tokens over arbitrary values;
`rounded/shadow/ring` surfaces; `bg-gradient-* bg-clip-text` headlines;
`aspect-*` + `object-cover` media without CLS.

## States/variants/a11y

`group-*/peer-*`/`has-*` parent/child/sibling states (CSS-only tabs with
named peers); `aria-*`/`data-*` mirror semantics; `motion-safe/reduce`
vestibular safety; `contrast-more`/`print:`/`dark:` (class strategy +
localStorage toggle); `sr-only`/`aria-busy`/`role=status` honest AT;
`focus-visible` rings; `invisible` vs `hidden` chosen per need.

## Tooling (v3 + v4)

`content:` paths (missing = missing classes in prod!) + `safelist` for dynamic
classes; `darkMode: 'class'`; `cva` + `clsx` + `tailwind-merge` variant APIs;
shadcn (owned components) / daisyUI (themes) / Headless (behavior) /
Radix (`data-state`); prettier sorting + eslint ordering; v4 = Oxide engine,
CSS-first (`@theme`), automatic content detection, `size-*` utilities.

## Components (full markup in training seeds)

Card, navbar, hero, modal (`role=dialog`), login form, dashboard sidebar,
striped table, toast (`role=status`), pricing grid, dark toggle, peer-tabs,
dropdown (`<details>`), skeleton loaders, shadcn setup, brand config.
