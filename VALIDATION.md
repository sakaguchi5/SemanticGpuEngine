# Validation

## 実施済み

- 14個の `.vcxproj` XML構文検査
- 14個の `.vcxproj.filters` XML構文検査
- Solution内Project GUIDとProjectReferenceの整合検査
- Projectが参照する全 `.cpp` / `.h` / `.hlsl` の存在検査
- 非Windows層をGCC C++20で `/W4` 相当かつ警告をエラー扱いでビルド
- Compiler / Classical Frontend / PGA / SDF / CubePlanテスト実行
- ExecutionPlanとGraphviz DOTの生成テスト
- ZIP CRC検査

## 環境上実施できない検査

生成環境はWindowsではないため、Visual Studio 2026 / MSVC v145での
実ビルドとDirectX 12実機描画は実行できません。

そのため、Windows専用層については次を行っています。

- Microsoft公式API定義との照合
- ヘッダー依存の静的確認
- D3D12 / DXGI / D3DCompilerのLink Library確認
- Resource State遷移とFrame Fenceのコードレビュー
- Shader Model 5.1とD3DReflectの組合せ確認

Visual Studioで最初にビルドした結果が最終的な実機検証になります。
