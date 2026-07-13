# Semantic GPU Engine V1 acceptance contract

Architecture-closure base: `712a16799a3b72fbae659fd6db59415dde56f36d`

This contract defines a finite compiler-style D3D12 foundation. It does not
claim completion of the longer geometric-algebra research programme.

## Final compilation boundary

`RenderRuntime` compiles a source `SemanticModule` into:

1. an immutable backend-only `CompiledRenderPackage`;
2. a separate `CompilationReport` for source-facing diagnostics.

The backend package contains no source module, compatibility module, or
`ExecutionPlan`. It contains resource and program blueprints, executable
specializations, normalized views, physical-instance and allocation data, and
one ordered `CompiledOperation` stream.

`FrameInvocation` carries only frame-dependent data. `IRenderBackend` receives
only package plus invocation.

## Unified backend execution contract

The operation stream must explicitly represent:

- work boundaries and queue assignment;
- cross-queue waits and signals;
- Temporal physical-instance waits and completion records;
- alias activation;
- Buffer or texture-subresource transitions;
- Copy-queue COMMON requirements and implicit promotion/decay;
- cyclic FrameLocal slot reuse;
- Raster, Compute, Copy, and Present commands.

The backend must not scan a work graph, queue-handoff table, cyclic-handoff
table, temporal-dependency table, or `ExecutionPlan` to infer execution policy.

## Native D3D12 package profile

The D3D12 package path must materialize directly from package blueprints:

- frame-local, temporal, persistent, and presentation instances;
- alias heaps and physical allocations;
- immutable Buffer initial data;
- immutable Texture mip/layer subresource data and preparation uploads;
- external Buffer/Texture slots with explicit ownership and state contracts;
- persistent compatible read envelopes;
- typed or typeless Texture resources and optimized clear values;
- programs, root signatures, PSOs, and range descriptors;
- package-sized descriptor heaps and expandable upload arenas.

The package path must never call
`EnsureCompiled(SemanticModule, ExecutionPlan)`. The concrete backend's old
source executor may remain temporarily for regression comparison, but it is
not exposed by `IRenderBackend` and cannot be reached from `RenderRuntime`.

## Existing V1 feature gate

The following remain required:

- pure `StructureHash()` and compiler-owned validation;
- canonical Program interfaces;
- Buffer and Texture range views with compatible reinterpretation;
- physical-instance and overlapping-range state analysis;
- Direct/Compute/Copy handoffs and cyclic FrameLocal reuse;
- multiple vertex streams, index streams, instancing, viewport, and scissor;
- executable Raster state, expanded formats, and shader reflection;
- memory/descriptor demand estimation and dynamic backend capacity;
- stable structured diagnostics;
- deterministic CPU tests and WARP package execution;
- common Classical/SDF experiment lowering and CSV metrics;
- Windows Debug/Release CI definitions.

## Release gate

V1 architecture closure is accepted only when:

- Debug x64 and Release x64 compile with Visual Studio 2026 / v145;
- all CPU package and operation-stream tests pass;
- package JSON is valid and records operations plus analysis counts;
- the D3D12 debug layer emits no error or corruption message;
- WARP executes the Copy-slot reuse path for at least five frames;
- WARP executes the native two-stream, Uint16-indexed, instanced Raster path;
- WARP executes the original multi-queue integration scene for 12 frames;
- repeated compilation produces identical package hashes and operation kinds;
- unsupported backend capabilities produce compile-time structured errors;
- source/module/plan materialization is absent from the package execution path.
- package-owned resources can be rematerialized after a device-epoch change;
- external resources require an explicit rebind after device recreation;
- frame submission exposes queue completion points;
- DRED device-removal diagnostics include package and operation identity.

The finite gate excludes Ray Work, bindless descriptors, DXC/Shader Model 6,
direct PGA/CGA GPU evaluation, and automatic Raster/Compute/Ray selection.
