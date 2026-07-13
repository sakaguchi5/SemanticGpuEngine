#include "10_Diagnostics/StructuredDiagnostics.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace
{
    std::string EscapeJson(const std::string& value)
    {
        std::ostringstream output;
        for (const unsigned char character : value)
        {
            switch (character)
            {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20)
                {
                    output << "\\u"
                           << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned>(character)
                           << std::dec;
                }
                else
                {
                    output << static_cast<char>(character);
                }
                break;
            }
        }
        return output.str();
    }

    template<class OptionalId>
    void WriteOptionalId(
        std::ostream& output,
        const char* name,
        const OptionalId& value,
        bool& first)
    {
        if (!value)
        {
            return;
        }
        if (!first)
        {
            output << ',';
        }
        first = false;
        output << '"' << name << "\":" << value->Value();
    }

    void WriteNormalizedView(
        std::ostream& output,
        const sge::compiler::NormalizedResourceView& view)
    {
        output << '{'
               << "\"resource\":" << view.resource.Value()
               << ",\"kind\":" << static_cast<unsigned>(view.kind)
               << ",\"byteOffset\":" << view.byteOffset
               << ",\"byteSize\":" << view.byteSize
               << ",\"strideBytes\":" << view.strideBytes
               << ",\"firstMip\":" << view.textureRange.firstMip
               << ",\"mipCount\":" << view.textureRange.mipCount
               << ",\"firstArrayLayer\":"
               << view.textureRange.firstArrayLayer
               << ",\"arrayLayerCount\":"
               << view.textureRange.arrayLayerCount
               << ",\"firstPlane\":"
               << static_cast<unsigned>(view.textureRange.firstPlane)
               << ",\"planeCount\":"
               << static_cast<unsigned>(view.textureRange.planeCount)
               << ",\"firstDepthSlice\":"
               << view.textureRange.firstDepthSlice
               << ",\"depthSliceCount\":"
               << view.textureRange.depthSliceCount
               << ",\"format\":\""
               << sge::gpu::ToString(view.format)
               << "\"}";
    }

    void WriteOperation(
        std::ostream& output,
        const sge::compiler::CompiledOperation& operation)
    {
        using namespace sge;
        output << "{\"type\":\""
               << compiler::ToString(operation) << '"';
        std::visit([&](const auto& value)
        {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T,
                compiler::BeginWorkOperation>)
            {
                output << ",\"workIndex\":" << value.workIndex
                       << ",\"work\":" << value.work.Value()
                       << ",\"name\":\""
                       << EscapeJson(value.name) << '"'
                       << ",\"queue\":"
                       << static_cast<unsigned>(value.queue);
            }
            else if constexpr (std::is_same_v<T,
                compiler::WaitForWorkOperation>)
            {
                output << ",\"waitingQueue\":"
                       << static_cast<unsigned>(value.waitingQueue)
                       << ",\"signalWork\":"
                       << value.signalWorkIndex
                       << ",\"signalQueue\":"
                       << static_cast<unsigned>(value.signalQueue);
            }
            else if constexpr (std::is_same_v<T,
                compiler::WaitForTemporalOperation>)
            {
                output << ",\"waitingQueue\":"
                       << static_cast<unsigned>(value.waitingQueue)
                       << ",\"resource\":"
                       << value.access.resource.Value()
                       << ",\"frameLag\":" << value.access.frameLag;
            }
            else if constexpr (std::is_same_v<T,
                compiler::ActivateAliasOperation>)
            {
                output << ",\"resource\":" << value.resource.Value()
                       << ",\"frameLag\":" << value.frameLag;
            }
            else if constexpr (std::is_same_v<T,
                compiler::TransitionOperation>)
            {
                output << ",\"view\":";
                WriteNormalizedView(output, value.view);
                output << ",\"state\":\""
                       << gpu::ToString(value.state) << '"'
                       << ",\"frameLag\":" << value.frameLag;
            }
            else if constexpr (std::is_same_v<T,
                compiler::RequireCommonOperation>)
            {
                output << ",\"view\":";
                WriteNormalizedView(output, value.view);
                output << ",\"frameLag\":" << value.frameLag
                       << ",\"implicitCopyState\":\""
                       << gpu::ToString(value.implicitCopyState) << '"'
                       << ",\"cyclicReuse\":"
                       << (value.cyclicReuse ? "true" : "false");
            }
            else if constexpr (std::is_same_v<T,
                compiler::ExecuteCommandOperation>)
            {
                output << ",\"commandKind\":"
                       << value.command.index();
            }
            else if constexpr (std::is_same_v<T,
                compiler::SubmitWorkOperation>)
            {
                output << ",\"workIndex\":" << value.workIndex
                       << ",\"queue\":"
                       << static_cast<unsigned>(value.queue)
                       << ",\"temporalAccesses\":"
                       << value.temporalAccesses.size();
            }
        }, operation);
        output << '}';
    }
}

namespace sge::diagnostics
{
    std::string DiagnosticToJson(
        const compiler::Diagnostic& diagnostic)
    {
        std::ostringstream output;
        output << "{\"code\":\""
               << compiler::ToString(diagnostic.code)
               << "\",\"severity\":\""
               << compiler::ToString(diagnostic.severity)
               << "\",\"message\":\""
               << EscapeJson(diagnostic.message)
               << "\",\"location\":{";

        bool first = true;
        WriteOptionalId(output, "resource", diagnostic.location.resource, first);
        WriteOptionalId(output, "program", diagnostic.location.program, first);
        WriteOptionalId(output, "work", diagnostic.location.work, first);
        if (diagnostic.location.view)
        {
            if (!first)
            {
                output << ',';
            }
            const auto& view = *diagnostic.location.view;
            output << "\"view\":{"
                   << "\"resource\":" << view.resource.Value()
                   << ",\"offsetBytes\":" << view.offsetBytes
                   << ",\"sizeBytes\":" << view.sizeBytes
                   << ",\"strideBytes\":" << view.strideBytes
                   << ",\"baseMip\":" << view.textureRange.baseMip
                   << ",\"mipCount\":" << view.textureRange.mipCount
                   << ",\"baseArrayLayer\":"
                   << view.textureRange.baseArrayLayer
                   << ",\"arrayLayerCount\":"
                   << view.textureRange.arrayLayerCount
                   << ",\"basePlane\":"
                   << static_cast<unsigned>(view.textureRange.basePlane)
                   << ",\"planeCount\":"
                   << static_cast<unsigned>(view.textureRange.planeCount)
                   << ",\"baseDepthSlice\":"
                   << view.textureRange.baseDepthSlice
                   << ",\"depthSliceCount\":"
                   << view.textureRange.depthSliceCount
                   << '}';
        }
        output << "},\"notes\":[";
        for (std::size_t index = 0; index < diagnostic.notes.size(); ++index)
        {
            if (index != 0)
            {
                output << ',';
            }
            output << '"' << EscapeJson(diagnostic.notes[index]) << '"';
        }
        output << "]}";
        return output.str();
    }

    void WriteCompiledPackageJson(
        const compiler::CompiledRenderPackage& package,
        const compiler::CompilationReport& report,
        std::span<const compiler::Diagnostic> diagnostics,
        const std::filesystem::path& outputPath)
    {
        std::ofstream output(outputPath, std::ios::trunc);
        if (!output)
        {
            throw std::runtime_error(
                "Could not write compiled package diagnostics.");
        }

        const auto& statistics = package.statistics;
        output << "{\n"
               << "  \"sourceHash\": " << package.sourceHash << ",\n"
               << "  \"packageHash\": " << package.packageHash << ",\n"
               << "  \"backendReady\": true,\n"
               << "  \"planningUsedCompatibilitySnapshot\": "
               << (report.planningUsedCompatibilitySnapshot
                    ? "true" : "false") << ",\n"
               << "  \"statistics\": {\n"
               << "    \"logicalResources\": "
               << statistics.logicalResourceCount << ",\n"
               << "    \"physicalInstances\": "
               << statistics.physicalInstanceCount << ",\n"
               << "    \"works\": " << statistics.workCount << ",\n"
               << "    \"executables\": "
               << statistics.executableCount << ",\n"
               << "    \"descriptorViews\": "
               << statistics.descriptorViewCount << ",\n"
               << "    \"operations\": "
               << statistics.operationCount << ",\n"
               << "    \"barriers\": " << statistics.barrierCount << ",\n"
               << "    \"queueWaits\": "
               << statistics.queueWaitCount << ",\n"
               << "    \"estimatedCommittedBytes\": "
               << statistics.estimatedCommittedBytes << "\n"
               << "  },\n"
               << "  \"requirements\": {\n"
               << "    \"textureSubresourceViews\": "
               << (package.requirements.textureSubresourceViews
                    ? "true" : "false") << ",\n"
               << "    \"multipleVertexStreams\": "
               << (package.requirements.multipleVertexStreams
                    ? "true" : "false") << ",\n"
               << "    \"indexedDraw\": "
               << (package.requirements.indexedDraw
                    ? "true" : "false") << ",\n"
               << "    \"instancedDraw\": "
               << (package.requirements.instancedDraw
                    ? "true" : "false") << ",\n"
               << "    \"explicitCopyQueueHandoffs\": "
               << (package.requirements.explicitCopyQueueHandoffs
                    ? "true" : "false") << ",\n"
               << "    \"dynamicDescriptorGrowth\": "
               << (package.requirements.dynamicDescriptorGrowth
                    ? "true" : "false") << ",\n"
               << "    \"advancedRasterState\": "
               << (package.requirements.advancedRasterState
                    ? "true" : "false") << ",\n"
               << "    \"customViewportScissor\": "
               << (package.requirements.customViewportScissor
                    ? "true" : "false") << ",\n"
               << "    \"expandedResourceFormats\": "
               << (package.requirements.expandedResourceFormats
                    ? "true" : "false") << "\n"
               << "  },\n"
               << "  \"operations\": [\n";

        for (std::size_t index = 0;
             index < package.operations.size(); ++index)
        {
            output << "    ";
            WriteOperation(output, package.operations[index]);
            output << (index + 1 == package.operations.size()
                ? "\n" : ",\n");
        }

        output << "  ],\n  \"analysis\": {\n"
               << "    \"dependencies\": "
               << report.analysisPlan.dependencies.size() << ",\n"
               << "    \"queueHandoffs\": "
               << report.queueHandoffs.size() << ",\n"
               << "    \"cyclicFrameHandoffs\": "
               << report.cyclicFrameHandoffs.size() << "\n"
               << "  },\n"
               << "  \"diagnostics\": [\n";
        for (std::size_t index = 0; index < diagnostics.size(); ++index)
        {
            output << "    " << DiagnosticToJson(diagnostics[index]);
            output << (index + 1 == diagnostics.size()
                ? "\n" : ",\n");
        }
        output << "  ]\n}\n";
    }
}
