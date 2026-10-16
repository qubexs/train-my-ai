# Production TypeScript Reference

## Strict project baseline

`target: ES2022` + `lib` matched to runtime (drop DOM for Node!) +
`module/moduleResolution: nodenext` + `strict: true` + `noUncheckedIndexedAccess`
+ `exactOptionalPropertyTypes` + `noImplicitReturns` + `noUnusedLocals/Parameters`
+ `declaration`/`declarationMap`/`sourceMap` + `outDir/rootDir` +
`isolatedModules` + `verbatimModuleSyntax` + `erasableSyntaxOnly` +
`skipLibCheck` — strict from day one (loosen when needed, never reverse!).

## Handbook map

Start: TS-for-JS/OOP/functional primers → Basics → Everyday Types → Narrowing
(guards/predicates/discriminated/`never`) → Functions/Objects → Type
Manipulation (generics/keyof/typeof/indexed/conditional/mapped/template) →
Classes → Modules → Utility full set → Decorators (standard vs legacy) →
Declaration Merging (interfaces/namespaces!) → Enums vs unions → Mixins →
Namespaces vs modules → Symbols → Compatibility (structural!) → Inference
(annotate edges!) → Declarations → .d.ts templates/do-don'ts/publishing
(attw+publint!) → JSDoc migration → Watch config → Releases 5.0–5.9 + 6.0
preview → JSX tsx rules → Migration tutorial (allowJs→JSDoc→checkJs→rename→strict).

## Modern features (5.x)

`using`/`await using` (RAII!), import attributes (`with`), `satisfies`
(checks yet keeps literals!), `const` type params, `NoInfer`, template infer
parsing, `erasableSyntaxOnly` (future-proof!), `rewriteRelativeImportExtensions`,
`allowImportingExtensions`, `resolvePackageJsonExports`, `customConditions`,
`moduleDetection: force`, project references (`composite` + `tsc -b`).

## Ecosystem typings

React (explicit props, no `FC`!, typed refs/hooks), Vue (`ref` duality,
`defineProps/Emits`), Svelte (`$store` magic), Angular (signals, typed inputs),
Solid (fine-grained stores), Qwik (resumable tasks); tRPC (no-codegen
end-to-end!), Prisma generated + `select/include`, zod (validate+infer
unified!), Effect/ArkType/Valibot schemas; OpenAPI codegen (hey-api/orval);
GraphQL codegen (exact ops!); MikroORM forks; Auth.js/Clerk; testing
(vitest typecheck, `tsc --noEmit`, tsd, attw, publint, api-extractor,
type-aware ESLint catching floating promises!).

## Full apps (training seeds)

Strict tsconfig, published library (attw-verified), JS→TS migration, utility
transforms, standard decorators, generic fetch-with-retries, tsd type tests,
monorepo references, JSDoc migration, typed React, type-aware ESLint, safe upgrade.
