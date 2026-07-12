RWByteAddressBuffer OutputBuffer : register(u0);

[numthreads(3, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint vertex = dispatchThreadId.x;

    float3 position;
    float4 color;

    if (vertex == 0)
    {
        position = float3(0.0f, 2.35f, 0.0f);
        color = float4(1.0f, 0.15f, 0.85f, 0.90f);
    }
    else if (vertex == 1)
    {
        position = float3(-0.65f, 1.25f, 0.0f);
        color = float4(0.15f, 0.85f, 1.0f, 0.90f);
    }
    else
    {
        position = float3(0.65f, 1.25f, 0.0f);
        color = float4(1.0f, 0.85f, 0.15f, 0.90f);
    }

    const uint byteOffset = vertex * 28;
    OutputBuffer.Store3(byteOffset, asuint(position));
    OutputBuffer.Store4(byteOffset + 12, asuint(color));
}
