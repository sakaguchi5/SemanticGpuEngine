# Architecture

## 目的

この実装は次の4領域を混ぜないことを目的にしています。

### 1. GPU実行に本質的な制約

- 実行するWorkが必要
- Workが読むResourceと書くResourceを区別する必要
- RAW / WAR / WAW hazardを守る必要
- Programと入力・出力の契約が必要
- 有限メモリ上でResource寿命を管理する必要
- CPUとGPUの完了関係を観測する必要
- 表示画像をPresentation systemへ渡す必要

これらは `02_GpuSemantics` と `03_RenderIR` で表現します。

### 2. D3D12が採用した表現

- PSO
- Root Signature
- Descriptor Heap
- Resource State / Resource Barrier
- Command Allocator / Command List / Command Queue
- ID3D12Fence
- DXGI Swap Chain

これらは `07_D3D12Backend` の外へ出しません。

### 3. エンジンが便宜上採用した設計

- 安定トポロジカルスケジュール
- Structure HashによるPlan Cache
- Executable KeyによるPSO Cache
- 3 Frames in Flight
- 1 MiB Frame Upload Arena
- Transient Resourceのallocation slot
- 自動Resource State追跡

これらはCompilerまたはRuntimeのPolicyです。
GPUの数学的必然とは扱いません。

### 4. アプリケーション固有の理論

- Cubeの定義
- 古典的な4x4行列
- PGAの6境界平面
- SDF
- カメラ
- 色
- 座標平面
- 回転則

これらは `08_ClassicalRasterFrontend`、`09_ExperimentalGeometry`、
`12_CubeLab` に閉じ込めます。

## 一方向依存

```text
Application Theory
        ↓
Frontend
        ↓
RenderIR / GpuSemantics
        ↓
RenderCompiler
        ↓
RenderRuntime
        ↓
D3D12Backend
        ↓
Driver / GPU
```

D3D12BackendはCube、Scene、GameObject、Material、PGA、SDFを知りません。

CubeLabはDevice、PSO、Barrier、Descriptor、Fenceを知りません。

## main.cppの役割

`13_Launcher/main.cpp` が行うのは次だけです。

- Platformを選ぶ
- Backendを選ぶ
- Runtime Policyを選ぶ
- Experimentを選ぶ

D3D12の初期化・描画命令・同期はmainに存在しません。

## SemanticModule

Frontendが出力する中間表現です。

```text
Resources
Programs
Works
```

各WorkはResource Accessを宣言します。

```text
DrawCube
  reads  Geometry
  reads  CubeConstants
  writes PresentationColor
  writes MainDepth
```

Compilerはこの意味から依存関係と状態遷移を導出します。

## ExecutionPlan

RenderCompilerの出力で、Backendとの正式な国境です。

```text
Dependencies
ScheduledWorks
ResourceLifetimes
AllocationSlots
AbstractTransitions
NormalizedExecutables
```

Backendはこの計画をD3D12へ物理変換します。

## Shader Interface

HLSLのBindingをC++側で盲目的に二重管理しません。

1. ProgramDeclarationが意味的なParameterを宣言
2. HLSLをコンパイル
3. D3DReflectで実際のBindingを取得
4. 宣言とReflectionを検証
5. ProgramDeclarationからRoot Signatureを生成

Cube版ではVertex Shaderの `b0` Constant BufferをRoot CBVへ変換します。

## Classical / PGA / SDF

現在表示に使う三角形はClassicalRasterFrontendが生成します。

ただしCubeのhalf extentはPGA境界面モデルから取得しており、
PGAとSDFは別プロジェクトの独立した理論表現として存在します。

将来は同じSemanticModuleへ次の経路を追加できます。

```text
PGA → CPU triangulation
PGA → GPU geometry generation
SDF → Ray marching
SDF → Compute-generated mesh
```
