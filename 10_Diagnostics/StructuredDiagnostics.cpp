#include "10_Diagnostics/StructuredDiagnostics.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

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
               << "  \"legacyExecutable\": "
               << (package.legacyExecutable ? "true" : "false") << ",\n"
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
               << "    \"barriers\": " << statistics.barrierCount << ",\n"
               << "    \"queueWaits\": "
               << statistics.queueWaitCount << ",\n"
               << "    \"estimatedCommittedBytes\": "
               << statistics.estimatedCommittedBytes << "\n"
               << "  },\n"
               << "  \"requirements\": {\n"
               << "    \"textureSubresourceViews\": "
               << (package.requirements.textureSubresourceViews ? "true" : "false")
               << ",\n"
               << "    \"multipleVertexStreams\": "
               << (package.requirements.multipleVertexStreams ? "true" : "false")
               << ",\n"
               << "    \"indexedDraw\": "
               << (package.requirements.indexedDraw ? "true" : "false")
               << ",\n"
               << "    \"instancedDraw\": "
               << (package.requirements.instancedDraw ? "true" : "false")
               << ",\n"
               << "    \"explicitCopyQueueHandoffs\": "
               << (package.requirements.explicitCopyQueueHandoffs
                    ? "true" : "false")
               << ",\n"
               << "    \"advancedRasterState\": "
               << (package.requirements.advancedRasterState
                    ? "true" : "false")
               << ",\n"
               << "    \"customViewportScissor\": "
               << (package.requirements.customViewportScissor
                    ? "true" : "false")
               << ",\n"
               << "    \"expandedResourceFormats\": "
               << (package.requirements.expandedResourceFormats
                    ? "true" : "false")
               << "\n"
               << "  },\n"
               << "  \"queueHandoffs\": [\n";

        for (std::size_t index = 0;
             index < package.queueHandoffs.size(); ++index)
        {
            const auto& handoff = package.queueHandoffs[index];
            output << "    {\"releaseWork\":"
                   << handoff.releaseScheduledWork
                   << ",\"acquireWork\":"
                   << handoff.acquireScheduledWork
                   << ",\"resource\":" << handoff.resource.Value()
                   << ",\"frameLag\":" << handoff.frameLag
                   << ",\"releaseQueue\":"
                   << static_cast<unsigned>(handoff.releaseQueue)
                   << ",\"acquireQueue\":"
                   << static_cast<unsigned>(handoff.acquireQueue)
                   << ",\"crossesCopyQueue\":"
                   << (handoff.crossesCopyQueue ? "true" : "false") << '}';
            output << (index + 1 == package.queueHandoffs.size() ? "\n" : ",\n");
        }

        output << "  ],\n  \"diagnostics\": [\n";
        for (std::size_t index = 0;
             index < package.diagnostics.size(); ++index)
        {
            output << "    " << DiagnosticToJson(package.diagnostics[index]);
            output << (index + 1 == package.diagnostics.size() ? "\n" : ",\n");
        }
        output << "  ]\n}\n";
    }
}
