# Semantic GPU Engine

Semantic GPU Engine is a compiler-style DirectX 12 rendering foundation for
Visual Studio 2026 and MSVC Platform Toolset v145.

Application and mathematical frontends produce an API-independent
`SemanticModule`. `RenderPackageCompiler` validates and canonicalizes that
source, runs dependency and scheduling analysis, then produces two different
artifacts:

- `CompiledRenderPackage`: the immutable backend-ready artifact;
- `CompilationReport`: source mapping, analysis plan, handoff explanations, and
  human-facing diagnostics context.

```text
Classical / SDF / future mathematical frontend
    -> SemanticModule
    -> RenderPackageCompiler
       |- CompiledRenderPackage -> RenderRuntime + FrameInvocation -> D3D12
       `- CompilationReport     -> text / DOT / JSON diagnostics
```

`CompiledRenderPackage` contains no `SemanticModule`, compatibility module, or
`ExecutionPlan`. The D3D12 backend materializes resources, programs, alias
heaps, descriptors, and PSOs directly from package blueprints. It executes one
linear compiled operation stream and does not rediscover queue, state, cyclic,
or temporal policy at runtime.

## Build and run

1. Open `SemanticGpuEngine.sln` in Visual Studio 2026.
2. Select `Debug | x64`.
3. Set `13_Launcher` as the startup project.
4. Press F5.

Controls:

- `1`: Classical Mesh frontend
- `2`: SDF frontend
- `Esc`: quit
- command line `--sdf`: start in SDF mode
- command line `--experiment-report`: write `experiment_report.csv`

Runtime diagnostics:

- `execution_plan.txt`
- `work_graph.dot`
- `compiled_package.json`
- optionally `experiment_report.csv`

## Backend-ready package

A package owns:

- resource blueprints, physical-instance selectors, allocation identities,
  reconstructible Buffer/Texture subresource data, external binding slots,
  persistent read envelopes, typeless requirements, and optimized clear values;
- program blueprints and executable specializations;
- normalized Buffer and Texture views;
- compiled Raster, Compute, Copy, and Present commands;
- a linear `CompiledOperation` stream;
- backend feature requirements, hashes, and package statistics.

`FrameInvocation` is separate from the static package. It supplies
frame-dependent CPU data and borrowed external resources with incoming and
outgoing state contracts. `FrameSubmission` returns per-queue completion
points and the current device epoch.

### Unified operation stream

Queue synchronization and resource-state policy are lowered before backend
execution. The stream consists of operations such as:

```text
WaitForWork / WaitForTemporal
BeginWork
ActivateAlias
Transition / RequireCommon
ExecuteCommand
SubmitWork
```

`RequireCommon` represents Copy-queue implicit promotion/decay and cyclic
FrameLocal slot reuse. Ordinary cross-queue release/acquire, Copy crossings,
Temporal waits, alias activation, subresource transitions, and cyclic frame
handoffs therefore share one ordered backend contract.

### Resource views and Raster execution

Buffer views use byte offset, byte size, and stride. Texture views use mip,
array-layer, plane, and Texture3D depth-slice ranges, with optional
 typeless-compatible format reinterpretation. The compiler rejects invalid or
overlapping read/write ranges while allowing independent texture
subresources.

Package-native Raster commands support multiple vertex and per-instance
streams, Uint16/Uint32 indices, indexed and non-indexed instancing, viewport,
scissor, cull/fill/front-face state, depth, blending, sample count, multiple
color attachments, expanded formats, and reflected shader interface checks.

### Native D3D12 materialization

The D3D12 backend creates alias heaps, physical resource instances, programs,
root signatures, PSOs, package-sized descriptor heaps, and expandable upload
arenas directly from the package. It no longer invokes the old
`EnsureCompiled(SemanticModule, ExecutionPlan)` path for package preparation.
The old source execution function may remain in the concrete backend only as an
unreachable regression implementation; it is not part of `IRenderBackend`.

Immutable Texture data is canonicalized by mip/layer/plane and uploaded through
`GetCopyableFootprints`/`CopyTextureRegion`. Borrowed D3D12 resources are
validated against package slots and can be safely rebound; the reference path
uses a conservative idle descriptor recycle when a bound object changes.

Device removal enables DRED before device creation, records
`device_removed.json`, increments a device epoch, reconstructs package-owned
objects from the package, resets temporal history, and requests a fresh bind
for external resources. The implementation remains a finite reference rather
than a bindless allocator or production residency manager.

## Diagnostics and reproducible experiments

Structured diagnostics carry stable code, severity, Work/Program/Resource/View
location, and notes. Package JSON exposes backend operations separately from
the compiler analysis report.

`ExperimentScene` is shared by Classical and SDF frontends. The experiment
harness records lowering and package compilation time, resource and instance
counts, executable/descriptor/operation/barrier/wait counts, and estimated
memory to CSV. GPU timestamps and image-error metrics remain explicit future
measurement extensions.

## Acceptance and CI

`11_Tests` covers deterministic compilation, canonicalization, texture-range
conflicts, generalized Raster, capability rejection, Copy handoff, cyclic
FrameLocal reuse, memory budget, common-scene lowering, operation-stream
lowering, and D3D12/WARP execution.

`.github/workflows/windows-warp.yml` builds Debug x64, runs the acceptance test
binary with WARP, and builds Release x64 using `cmd` and MSBuild.

Ray Work, bindless descriptors, DXC/Shader Model 6, direct PGA/CGA GPU
evaluation, and automatic Raster/Compute/Ray selection remain later research,
not hidden V1 requirements.
