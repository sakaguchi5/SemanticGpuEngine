# Semantic GPU Engine

Semantic GPU Engine is a compiler-style DirectX 12 rendering foundation for
Visual Studio 2026 and MSVC Platform Toolset v145.

Application and mathematical frontends produce an API-independent
`SemanticModule`. `RenderPackageCompiler` validates and canonicalizes that
source, then compiles dependencies, scheduling, lifetime, physical instances,
allocation, resource-view ranges, state requirements, queue handoffs, program
layouts, and executable specializations into one immutable
`CompiledRenderPackage`.

```text
Classical / SDF / future mathematical frontend
    -> SemanticModule
    -> RenderPackageCompiler
    -> CompiledRenderPackage
    -> RenderRuntime + FrameInvocation
    -> D3D12 backend
```

D3D12 resources, descriptors, command lists, barriers, PSOs, and fences remain
below the backend boundary. The backend consumes the package-native compiled
works; the old module-plus-plan entry remains only as a transitional
compatibility hook for the already proven core materialization path.

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

## V1 compiler package

A package owns a canonical immutable module snapshot, compiled Raster/Compute/
Copy/Present commands, normalized Buffer and Texture views, range state
requirements, queue handoffs, resource and program blueprints, executable
specializations, physical-instance and allocation plans, statistics, and
structured diagnostics.

`FrameInvocation` is separate from the static package. It supplies only
frame-dependent CPU resource data and the frame number, so changing constants
does not change the compiled package identity.

### Resource views and states

Buffer views use byte offset, byte size, and stride. Texture views use mip,
array-layer, plane, and Texture3D depth-slice ranges, with optional typeless-compatible format
reinterpretation. The compiler rejects invalid or overlapping read/write and
multiple-write ranges on the same physical instance, while allowing independent
non-overlapping texture subresources.

The D3D12 backend creates range-specific SRV/UAV/RTV/DSV descriptors and emits
subresource barriers. Cross-queue ownership changes become explicit package
handoffs, including Copy-queue crossings. Persistent immutable read-only
resources continue to use a compiled compatible read-state envelope.

### Raster executable

Package-native Raster commands support:

- multiple vertex streams and per-instance streams;
- Uint16 and Uint32 index buffers;
- indexed and non-indexed instanced draw;
- viewport and scissor;
- cull mode, fill mode, front-face convention, depth, blending, and sample
  count;
- multiple color attachments and expanded typed color/depth formats;
- shader reflection checks for resource bindings, vertex inputs, color outputs,
  and depth output compatibility.

The WARP acceptance test executes a two-stream, Uint16-indexed, instanced Raster
package before running the original 12-frame multi-queue integration path.

### Scalable physical backend

Descriptor heaps and per-frame upload arenas are sized from package demand and
can grow across package recompilation. The compiler estimates committed memory
from resource dimensions and physical-instance counts and rejects a known local
memory-budget overflow before execution.

The implementation remains a finite V1 reference, not a bindless allocator or
full production residency manager. Texture arrays, multisampling, and expanded
format paths are represented and lowered, while the acceptance scene exercises
the most important generalized Buffer/Raster path and CPU tests cover texture
subresource compilation.

## Diagnostics and reproducible experiments

Structured diagnostics carry stable code, severity, Work/Program/Resource/View
location, and notes. Package JSON exposes hashes, requirements, statistics,
handoffs, and diagnostics.

`ExperimentScene` is shared by Classical and SDF frontends. The experiment
harness records frontend lowering time, package compilation time, resource and
instance counts, executable/descriptor/barrier/wait counts, and estimated
memory to CSV. GPU timestamp and image-error metrics remain explicit backend
measurement extensions rather than fabricated CPU estimates.

## Acceptance and CI

`11_Tests` includes deterministic package compilation, canonicalization,
texture-range conflict, generalized Raster, capability rejection, Copy
handoff, memory-budget, common-scene experiment, and D3D12/WARP tests.

`.github/workflows/windows-warp.yml` builds Debug x64, runs the acceptance test
binary with WARP, and builds Release x64 using `cmd` and MSBuild.

See `V1_ACCEPTANCE.md` for the finite V1 completion contract and its mapping to
the fifteen implementation stages. Ray Work, bindless descriptors, DXC/Shader
Model 6, direct PGA/CGA GPU evaluation, and automatic Raster/Compute/Ray choice
remain later research rather than hidden V1 requirements.
