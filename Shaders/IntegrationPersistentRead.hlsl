struct IntegrationVertex
{
    float3 position;
    float4 color;
};

StructuredBuffer<IntegrationVertex> SharedVertices : register(t0);
RWByteAddressBuffer OutputBuffer : register(u0);

[numthreads(1, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const IntegrationVertex vertex = SharedVertices[0];
    OutputBuffer.Store3(0, asuint(vertex.position));
    OutputBuffer.Store4(12, asuint(vertex.color));
}
