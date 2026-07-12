# Semantic GPU Engine

Visual Studio 2026（version 18）と MSVC Platform Toolset v145 を対象にした、
DirectX 12のコンパイラ型レンダリング基盤です。

これは単純なCubeサンプルではありません。

アプリケーション固有の理論をAPI非依存の意味IRへ変換し、
依存関係、Resource寿命、抽象状態遷移、実行構成をコンパイルした後、
D3D12のRoot Signature、PSO、Resource Barrier、Command List、Fenceへ
変換する端から端までの参照実装です。

## 必要環境

- Visual Studio 2026
- Desktop development with C++
- MSVC Platform Toolset v145
- Windows 10またはWindows 11 SDK
- DirectX 12対応GPU
  - 対応GPUが見つからない場合はWARPへフォールバックします

## 実行方法

1. `SemanticGpuEngine.sln` を開く
2. 構成を `Debug | x64` にする
3. `13_Launcher` がスタートアッププロジェクトであることを確認する
4. F5を押す

表示内容:

- 回転する色付き立方体
- XY / YZ / ZXの半透明平面
- 三平面のグリッド
- X軸: 赤、Y軸: 緑、Z軸: 青
- 深度テスト
- アルファ合成
- ウィンドウリサイズ
- 3 frames in flight

Escキーで終了します。

## 実行時に生成される診断ファイル

実行フォルダーに次が生成されます。

- `execution_plan.txt`
  - Workの実行順
  - Resource依存関係
  - Resource寿命
  - 抽象状態遷移
  - 正規化されたExecutable
- `work_graph.dot`
  - Graphviz形式のWork依存グラフ

## ソリューション構成

- `00_Foundation`
  - 型付きID、Hash、契約
- `01_Platform`
  - Win32ウィンドウ、イベントループ、時刻
- `02_GpuSemantics`
  - Work、Resource、Access、Program、AbstractState
- `03_RenderIR`
  - API非依存のSemanticModule
- `04_RenderCompiler`
  - 検証、hazard解析、DAG、schedule、寿命、allocation slot、状態計画
- `05_RenderRuntime`
  - ExecutionPlanキャッシュ、Backend境界
- `06_ShaderSystem`
  - HLSLコンパイル、Shader Reflection、ProgramInterface検証
- `07_D3D12Backend`
  - Device、Swap Chain、Resource、Root Signature、PSO、Barrier、Fence
- `08_ClassicalRasterFrontend`
  - 古典的なfloatベクトル、4x4行列、頂点・三角形表現
- `09_ExperimentalGeometry`
  - PGA境界平面によるBox、SDF Box
- `10_Diagnostics`
  - ExecutionPlanと依存グラフの出力
- `11_Tests`
  - GPUを必要としないCompiler・Frontendテスト
- `12_CubeLab`
  - 複数の幾何表現を比較する受け入れ実験
- `13_Launcher`
  - Platform、Backend、Runtime、Experimentの組み立てだけを行うmain

## 実装された垂直経路

1. CubeLabがSemanticModuleを生成
2. SemanticValidatorがID、参照、Resourceデータ、Device能力を検証
3. DependencyAnalyzerがRAW / WAR / WAW hazardから依存DAGを作成
4. 安定トポロジカルソートがWork順序を決定
5. LifetimeAnalyzerがResourceの最初と最後の使用位置を計算
6. Transient Resourceへ物理allocation slotを計画
7. StatePlannerがAbstractState遷移を生成
8. ProgramとRasterStateからExecutableを正規化
9. Shader ReflectionでHLSL BindingとProgramDeclarationを照合
10. ProgramInterfaceからD3D12 Root Signatureを生成
11. ExecutableからD3D12 PSOを生成・キャッシュ
12. Static BufferをDefault HeapへUpload
13. Dynamic ConstantsをFrame Upload Arenaへ配置
14. AbstractStateからD3D12 Resource Barrierを生成
15. 3つのFrameContextとFenceでCPU/GPUを並行化
16. Swap ChainへPresent

## 境界規則

`07_D3D12Backend` より上の層には、次のD3D12型を公開しません。

- `ID3D12Resource`
- `ID3D12PipelineState`
- `D3D12_RESOURCE_STATES`
- `D3D12_RESOURCE_BARRIER`
- Descriptor Handle
- Command List
- Fence
- DXGI Format

一方で、メモリ予算や実行能力など、本物のハードウェア制約は
`DeviceCapabilities`として意味的に公開します。

## 現在の完成範囲

Cube受け入れテストに必要な垂直経路は実装されています。

将来追加できるが、この受け入れテストには未使用の機能:

- Compute WorkのD3D12実行
- Ray Work
- SRV / UAV / Samplerの一般Binding
- Descriptor Table / Bindless
- Placed Resourceによる実メモリエイリアス
- Async Compute Queue
- Shader Model 6 / DXC
- PGA / CGAをGPU上で直接評価するFrontend
- SDF Ray Marching Frontend

これらはSemanticModuleとExecutionPlanの国境を壊さずに追加できます。
