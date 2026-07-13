# Semantic GPU Engine V1 acceptance contract

Base commit: `7de3fb647c36281483baf7601c586f2f33040f92`

This contract defines a finite completion target for the compiler-style D3D12
rendering foundation. It does not claim completion of the longer geometric
algebra research programme.

## Final compilation boundary

`RenderRuntime` compiles a source `SemanticModule` into one immutable
`CompiledRenderPackage`. The package contains:

- a canonical module snapshot with legacy Program parameters removed;
- normalized Buffer and Texture views;
- physical-instance and overlapping-range state requirements;
- compiled Raster, Compute, Copy, and Present commands;
- resource instance and physical allocation plans;
- explicit queue release/acquire handoffs, including Copy-queue crossings;
- executable specializations and package statistics;
- stable structured diagnostics.

`FrameInvocation` is separate and carries only frame-dependent data. The D3D12
backend's primary execution entry receives the package and invocation. The old
module-plus-plan call remains only as a transitional internal compatibility
hook used while materializing the already proven core resources and programs;
package-native command execution does not ask the application to resubmit the
source module.

## Fifteen implementation stages

1. `StructureHash()` is pure. Resource-instance state validation belongs to
   `RenderPackageCompiler::Validate()`.
2. Legacy `ProgramDeclaration::parameters` is canonicalized into
   `ProgramInterface`; canonical packages never retain both representations.
3. `CompiledRenderPackage` is the runtime compilation result.
4. Package plus `FrameInvocation` is the primary runtime/backend boundary.
5. `ResourceView` supports Buffer ranges and Texture mip/layer/plane ranges,
   with optional typeless-compatible format reinterpretation.
6. State conflicts are analyzed by physical instance and overlapping normalized
   range rather than only by ResourceId.
7. Cross-queue use is lowered to explicit release/acquire handoffs; Copy-queue
   crossings are represented instead of rejected as an unnamed special case.
8. Raster commands support multiple vertex streams, index streams, instancing,
   viewport, and scissor.
9. Raster state and expanded formats are executable identity; shader reflection
   validates bindings, vertex input signatures, and Raster outputs.
10. The compiler estimates physical memory and descriptor demand. The D3D12
    backend grows descriptor heaps and frame upload arenas from package demand.
11. Diagnostics have stable code, severity, Work/Program/Resource/View
    locations, notes, and JSON output.
12. CPU acceptance tests cover deterministic compilation, range overlap,
    generalized Raster, backend capability rejection, Copy handoff, memory
    budget, and common-scene lowering. WARP covers native package execution.
13. Classical and SDF lower from one representation-neutral
    `ExperimentScene`.
14. `ExperimentHarness` records lowering and compile time, resource and instance
    counts, executable/descriptor/barrier/wait counts, and estimated memory.
15. Windows Debug/Release builds and WARP tests are defined in GitHub Actions.

## Native D3D12 package profile

The D3D12 backend consumes compiled package works and implements:

- Raster, Compute, Copy, and Present commands;
- Direct, Compute, and Copy queue timelines;
- explicit package handoffs and temporal waits;
- frame-local, temporal, persistent, and external physical instances;
- placed-resource aliasing and persistent compatible read envelopes;
- Buffer and Texture range SRV/UAV descriptors;
- mip/layer/plane/depth-slice RTV/DSV and subresource barriers;
- multiple vertex streams, indexed draw, and instancing;
- viewport, scissor, cull/fill/front-face state, MRT, depth, and typed formats;
- package-sized descriptor heaps and expandable per-frame upload arenas.

The compatibility module and original `ExecutionPlan` remain embedded in the
package so the old proven allocation/materialization passes can be reused while
the execution boundary is migrated. They are an implementation detail, not the
public runtime contract.

## Release gate

V1 is accepted only when all of the following hold:

- Debug x64 and Release x64 compile with Visual Studio 2026 / v145;
- all CPU package tests pass;
- the D3D12 debug layer emits no error or corruption message;
- WARP executes the package-native two-stream, Uint16-indexed, instanced Raster
  acceptance scene;
- WARP then executes the original multi-queue integration scene for 12 frames;
- Classical and SDF package compilation succeed from the same experiment scene;
- repeated compilation produces identical package hashes and dependency plans;
- a backend that does not advertise a required feature receives a compile-time
  structured error rather than silent feature degradation.

The finite gate deliberately excludes Ray Work, bindless descriptors, DXC/
Shader Model 6, direct PGA/CGA GPU execution, and automatic selection among
Raster, Compute, and Ray implementations.
